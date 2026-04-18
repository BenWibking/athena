//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator.cpp
//! \brief Idealized galaxy precipitator problem generator (hydrostatic + gravity only)
//========================================================================================

// C headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// C++ headers
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../units/units.hpp"

namespace {

// Simple monotone interpolant over tabulated profile data
class PrecipitatorProfile {
 public:
  explicit PrecipitatorProfile(const std::string &filename) { LoadProfile(filename); }

  Real Density(Real r) const { return Interp(rho_table_, r); }
  Real Pressure(Real r) const { return Interp(pressure_table_, r); }
  Real Gravity(Real r) const { return Interp(gravity_table_, r); }
  Real Phi(Real r) const { return Interp(phi_table_, r); }
  Real BField(Real r) const { return Interp(bfield_table_, r); }

  Real MinRadius() const { return radius_table_.front(); }
  Real MaxRadius() const { return radius_table_.back(); }

 private:
  void LoadProfile(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Failed to open precipitator profile file: " << filename;
      ATHENA_ERROR(msg);
    }

    std::string header;
    std::getline(file, header);

    for (std::string line; std::getline(file, line);) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream iss(line);
      std::vector<Real> values;
      for (Real value = 0.0; iss >> value;) {
        values.push_back(value);
      }
      if (values.size() < 8) {
        continue;
      }
      radius_table_.push_back(values[0]);
      rho_table_.push_back(values[1]);
      pressure_table_.push_back(values[2]);
      gravity_table_.push_back(values[3]);
      phi_table_.push_back(values[6]);
      bfield_table_.push_back(values[7]);
    }

    if (radius_table_.empty()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Precipitator profile file contains no data rows.";
      ATHENA_ERROR(msg);
    }
  }

  Real Interp(const std::vector<Real> &table, Real r) const {
    if (r <= radius_table_.front()) {
      return table.front();
    }
    if (r >= radius_table_.back()) {
      return table.back();
    }
    auto it = std::lower_bound(radius_table_.begin(), radius_table_.end(), r);
    std::size_t idx = static_cast<std::size_t>(std::distance(radius_table_.begin(), it));
    const Real r1 = radius_table_[idx - 1];
    const Real r2 = radius_table_[idx];
    const Real f = (r - r1) / (r2 - r1);
    return table[idx - 1] * (1.0 - f) + table[idx] * f;
  }

  std::vector<Real> radius_table_;
  std::vector<Real> rho_table_;
  std::vector<Real> pressure_table_;
  std::vector<Real> gravity_table_;
  std::vector<Real> phi_table_;
  std::vector<Real> bfield_table_;
};

std::unique_ptr<PrecipitatorProfile> g_profile;
int g_uniform_init = 0;
Real g_uniform_height = 0.0;
Real g_gamma = 0.0;
Real g_gm1 = 0.0;
bool g_enable_powerlaw_cooling = false;
Real g_powerlaw_lambda_code = 0.0;
bool g_enable_magic_heating = false;
Real g_magic_target_temperature = 0.0;
Real g_magic_Kp = 0.0;
Real g_magic_h_smooth = 1.0;
Real g_magic_mu = 0.0;
Real g_magic_mmw_cgs = 0.0;
Real g_magic_mmw_code = 0.0;
Real g_magic_c_v = 0.0;
int g_magic_profile_bins = 0;
Real g_magic_profile_x1min = 0.0;
Real g_magic_profile_inv_dr = 0.0;
Real g_magic_profile_time = std::numeric_limits<Real>::quiet_NaN();
Real g_magic_profile_dt = std::numeric_limits<Real>::quiet_NaN();
bool g_magic_profile_ready = false;
std::vector<Real> g_magic_error_profile;

std::string g_force_free_param_file;  // NOLINT(runtime/string)
bool g_force_free_loaded = false;
Real g_force_free_alpha = 0.0;
Real g_force_free_amplitude = 1.0;
bool g_force_free_bfield_override = false;
Real g_force_free_bfield_override_value = 1.0;

bool g_enable_density_perturbations = false;
Real g_pert_sigma = 0.0;
int g_pert_lmax = 0;
int g_pert_radial_modes = 0;
Real g_pert_kmin = 0.0;
Real g_pert_kmax = 0.0;
Real g_pert_small_kr_threshold = 1.0;
std::uint64_t g_pert_seed = 88172645463393265ULL;
std::uint64_t g_pert_state = 0;
Real g_pert_radius_min = 0.0;
Real g_pert_radius_max = 1.0;
bool g_pert_ready = false;

std::vector<Real> g_pert_coeff_cos;
std::vector<Real> g_pert_coeff_sin;
std::vector<Real> g_pert_k_values;
std::vector<Real> g_pert_radial_weight;

Real g_inner_sponge_radius = 0.0;
Real g_inner_sponge_tau = 0.0;
Real g_outer_sponge_inner_radius = 0.0;
Real g_outer_sponge_tau = 0.0;
bool g_enable_outer_buffer_halo = false;
Real g_outer_buffer_inner_radius_code = 0.0;
Real g_outer_buffer_inner_radius_cgs = 0.0;
Real g_outer_buffer_entropy_slope = 0.0;
Real g_outer_buffer_match_density_cgs = 0.0;
Real g_outer_buffer_match_pressure_cgs = 0.0;
Real g_outer_buffer_match_entropy_cgs = 0.0;
Real g_outer_buffer_match_enthalpy_cgs = 0.0;

constexpr const char *kDensityContrastName = "delta_rho_over_rho_bar";
constexpr const char *kTemperatureOutputName = "temperature_K";
int g_radial_profile_bins = 0;
Real g_radial_profile_rmin = 0.0;
Real g_radial_profile_rmax = 0.0;
Real g_radial_profile_inv_dr = 0.0;
std::vector<Real> g_radial_density_profile;
std::vector<Real> g_radial_pressure_profile;
std::vector<Real> g_radial_entropy_profile;
std::vector<Real> g_radial_temperature_profile;
std::vector<Real> g_radial_v1_profile;
std::vector<Real> g_radial_v2_profile;
std::vector<Real> g_radial_v3_profile;
Real g_radial_density_time = std::numeric_limits<Real>::quiet_NaN();
int g_radial_density_cycle = -1;
bool g_radial_density_ready = false;

Real PrecipitatorThetaGrid(Real x2, RegionSize rs);
void ApplyInnerSponge(MeshBlock *pmb);
void ApplyOuterSponge(MeshBlock *pmb);
Real ComputeCellTemperature(Real rho_code, Real pressure_code, const Units &units);
Real HistoryMagicHeatingRate(MeshBlock *pmb, int iout);
Real CellCenterRadiusCode(const Coordinates *coord, int k, int j, int i);
Real NominalOuterRadiusCode(const RegionSize &size);
Real RadialProfileMaxCode(const RegionSize &size);
bool InOuterBufferHalo(Real radius_code);
Real OuterBufferHeatCoolTaper(Real radius_code, Real outer_radius_code);

struct CartesianPoint {
  Real x;
  Real y;
  Real z;
};

struct SphericalPoint {
  Real r;
  Real theta;
  Real phi;
  Real rhat_x;
  Real rhat_y;
  Real rhat_z;
};

bool UsingCartesianCoordinates() {
#if defined(COORDINATE_SYSTEM)
  return std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0;
#else
  return false;
#endif
}

bool UsingSphericalPolarCoordinates() {
#if defined(COORDINATE_SYSTEM)
  return std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0;
#else
  return false;
#endif
}

Real CellCenteredCoord2(const Coordinates *coord, int j) {
  return coord->x2v(j);
}

Real CellCenteredCoord3(const Coordinates *coord, int k) {
  return coord->x3v(k);
}

CartesianPoint MeshCoordinatesToCartesian(Real x1, Real x2, Real x3) {
  if (UsingCartesianCoordinates()) {
    return {x1, x2, x3};
  }
  if (UsingSphericalPolarCoordinates()) {
    const Real sin_theta = std::sin(x2);
    return {x1 * sin_theta * std::cos(x3), x1 * sin_theta * std::sin(x3),
            x1 * std::cos(x2)};
  }
  return {x1, x2, x3};
}

CartesianPoint CellCenterPosition(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1v(i), CellCenteredCoord2(coord, j),
                                    CellCenteredCoord3(coord, k));
}

CartesianPoint Face1CenterPosition(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1f(i), CellCenteredCoord2(coord, j),
                                    CellCenteredCoord3(coord, k));
}

CartesianPoint Face2CenterPosition(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1v(i), coord->x2f(j),
                                    CellCenteredCoord3(coord, k));
}

CartesianPoint Face3CenterPosition(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1v(i), CellCenteredCoord2(coord, j),
                                    coord->x3f(k));
}

CartesianPoint Edge1Position(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1v(i), coord->x2f(j), coord->x3f(k));
}

CartesianPoint Edge2Position(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1f(i), CellCenteredCoord2(coord, j),
                                    coord->x3f(k));
}

CartesianPoint Edge3Position(const Coordinates *coord, int k, int j, int i) {
  return MeshCoordinatesToCartesian(coord->x1f(i), coord->x2f(j),
                                    CellCenteredCoord3(coord, k));
}

SphericalPoint CartesianToSpherical(const CartesianPoint &pos) {
  const Real r = std::sqrt(SQR(pos.x) + SQR(pos.y) + SQR(pos.z));
  if (!(r > 0.0)) {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  const Real cos_theta = std::max(static_cast<Real>(-1.0),
                                  std::min(static_cast<Real>(1.0), pos.z / r));
  Real phi = std::atan2(pos.y, pos.x);
  if (phi < 0.0) {
    phi += 2.0 * PI;
  }
  return {r, std::acos(cos_theta), phi, pos.x / r, pos.y / r, pos.z / r};
}

SphericalPoint CellCenterSpherical(const Coordinates *coord, int k, int j, int i) {
  return CartesianToSpherical(CellCenterPosition(coord, k, j, i));
}

Real MaxAbsBound(Real a, Real b) {
  return std::max(std::abs(a), std::abs(b));
}

Real NominalOuterRadiusCode(const RegionSize &size) {
  if (!UsingCartesianCoordinates()) {
    return size.x1max;
  }

  Real r_outer = MaxAbsBound(size.x1min, size.x1max);
  if (size.nx2 > 1) {
    r_outer = std::min(r_outer, MaxAbsBound(size.x2min, size.x2max));
  }
  if (size.nx3 > 1) {
    r_outer = std::min(r_outer, MaxAbsBound(size.x3min, size.x3max));
  }
  return r_outer;
}

Real RadialProfileMaxCode(const RegionSize &size) {
  if (!UsingCartesianCoordinates()) {
    return size.x1max;
  }

  const Real rx = MaxAbsBound(size.x1min, size.x1max);
  const Real ry = (size.nx2 > 1) ? MaxAbsBound(size.x2min, size.x2max) : 0.0;
  const Real rz = (size.nx3 > 1) ? MaxAbsBound(size.x3min, size.x3max) : 0.0;
  return std::sqrt(SQR(rx) + SQR(ry) + SQR(rz));
}

Real RadialProfileMinCode(const RegionSize &size) {
  if (UsingCartesianCoordinates()) {
    return 0.0;
  }
  return size.x1min;
}

int RadialProfileBinCount(const RegionSize &size) {
  return std::max(size.nx1, 1);
}

int ClampRadialBinIndex(int num_bins, Real radius, Real rmin, Real inv_dr) {
  if (num_bins <= 1 || inv_dr == 0.0) {
    return 0;
  }
  int idx = static_cast<int>((radius - rmin) * inv_dr);
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= num_bins) {
    idx = num_bins - 1;
  }
  return idx;
}

Real SampleRadialProfile(const std::vector<Real> &profile, Real radius) {
  if (!g_radial_density_ready || profile.empty()) {
    return 0.0;
  }
  if (g_radial_profile_bins <= 1 || g_radial_profile_inv_dr == 0.0) {
    return profile.front();
  }
  Real idx_f = (radius - g_radial_profile_rmin) * g_radial_profile_inv_dr;
  if (idx_f <= 0.0) {
    return profile.front();
  }
  const Real max_index = static_cast<Real>(g_radial_profile_bins - 1);
  if (idx_f >= max_index) {
    return profile.back();
  }
  const int idx = static_cast<int>(idx_f);
  const Real frac = idx_f - static_cast<Real>(idx);
  const Real a = profile[static_cast<std::size_t>(idx)];
  const Real b = profile[static_cast<std::size_t>(idx + 1)];
  return a + frac * (b - a);
}

void ComputeRadialProfiles(Mesh *mesh) {
  if (mesh == nullptr) {
    return;
  }
  if (mesh->multilevel) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Density contrast output is not implemented for multilevel meshes.";
    ATHENA_ERROR(msg);
  }

  const int num_bins = RadialProfileBinCount(mesh->mesh_size);
  if (num_bins <= 0) {
    g_radial_density_profile.clear();
    g_radial_pressure_profile.clear();
    g_radial_entropy_profile.clear();
    g_radial_temperature_profile.clear();
    g_radial_v1_profile.clear();
    g_radial_v2_profile.clear();
    g_radial_v3_profile.clear();
    g_radial_profile_bins = 0;
    g_radial_profile_rmin = 0.0;
    g_radial_profile_rmax = 0.0;
    g_radial_profile_inv_dr = 0.0;
    g_radial_density_ready = false;
    return;
  }

  Units *units = mesh->punit;
#if NON_BAROTROPIC_EOS
  if (units == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Units object must be configured before computing diagnostic profiles.";
    ATHENA_ERROR(msg);
  }
#else
  (void)units;
#endif

  const Real rmin = RadialProfileMinCode(mesh->mesh_size);
  const Real rmax = RadialProfileMaxCode(mesh->mesh_size);
  const bool has_extent = (num_bins > 1) && (rmax > rmin);
  const Real inv_dr = has_extent ? static_cast<Real>(num_bins) / (rmax - rmin) : 0.0;

  const std::size_t vec_size = static_cast<std::size_t>(num_bins);
  std::vector<Real> rho_sum(vec_size, 0.0);
#if NON_BAROTROPIC_EOS
  std::vector<Real> pressure_sum(vec_size, 0.0);
  std::vector<Real> entropy_sum(vec_size, 0.0);
  std::vector<Real> temperature_sum(vec_size, 0.0);
#endif
  std::vector<Real> v1_sum(vec_size, 0.0);
  std::vector<Real> v2_sum(vec_size, 0.0);
  std::vector<Real> v3_sum(vec_size, 0.0);
  std::vector<Real> volume(vec_size, 0.0);

  for (int block = 0; block < mesh->nblocal; ++block) {
    MeshBlock *pmb = mesh->my_blocks(block);
    if (pmb == nullptr || pmb->phydro == nullptr) {
      continue;
    }
    Coordinates *coord = pmb->pcoord;
    auto &prim = pmb->phydro->w;

    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          const Real radius = CellCenterRadiusCode(coord, k, j, i);
          const std::size_t idx = static_cast<std::size_t>(
              ClampRadialBinIndex(num_bins, radius, rmin, inv_dr));
          const Real cell_volume = coord->GetCellVolume(k, j, i);
          const Real rho = prim(IDN, k, j, i);
          rho_sum[idx] += rho * cell_volume;
#if NON_BAROTROPIC_EOS
          const Real pressure = prim(IPR, k, j, i);
          pressure_sum[idx] += pressure * cell_volume;
          Real entropy = 0.0;
          if (pressure > 0.0 && rho > 0.0) {
            entropy = pressure / std::pow(rho, g_gamma);
          }
          entropy_sum[idx] += entropy * cell_volume;
          const Real temperature = ComputeCellTemperature(rho, pressure, *units);
          temperature_sum[idx] += temperature * cell_volume;
#endif
          const Real v1 = prim(IVX, k, j, i);
          const Real v2 = prim(IVY, k, j, i);
          const Real v3 = prim(IVZ, k, j, i);
          v1_sum[idx] += v1 * cell_volume;
          v2_sum[idx] += v2 * cell_volume;
          v3_sum[idx] += v3 * cell_volume;
          volume[idx] += cell_volume;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, rho_sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#if NON_BAROTROPIC_EOS
  MPI_Allreduce(MPI_IN_PLACE, pressure_sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, entropy_sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, temperature_sum.data(), num_bins, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  MPI_Allreduce(MPI_IN_PLACE, v1_sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, v2_sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, v3_sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, volume.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  g_radial_density_profile.assign(vec_size, 0.0);
#if NON_BAROTROPIC_EOS
  g_radial_pressure_profile.assign(vec_size, 0.0);
  g_radial_entropy_profile.assign(vec_size, 0.0);
  g_radial_temperature_profile.assign(vec_size, 0.0);
#endif
  g_radial_v1_profile.assign(vec_size, 0.0);
  g_radial_v2_profile.assign(vec_size, 0.0);
  g_radial_v3_profile.assign(vec_size, 0.0);

  for (int i = 0; i < num_bins; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    if (volume[idx] > 0.0) {
      g_radial_density_profile[idx] = rho_sum[idx] / volume[idx];
#if NON_BAROTROPIC_EOS
      g_radial_pressure_profile[idx] = pressure_sum[idx] / volume[idx];
      g_radial_entropy_profile[idx] = entropy_sum[idx] / volume[idx];
      g_radial_temperature_profile[idx] = temperature_sum[idx] / volume[idx];
#endif
      g_radial_v1_profile[idx] = v1_sum[idx] / volume[idx];
      g_radial_v2_profile[idx] = v2_sum[idx] / volume[idx];
      g_radial_v3_profile[idx] = v3_sum[idx] / volume[idx];
    } else {
      g_radial_density_profile[idx] = 0.0;
#if NON_BAROTROPIC_EOS
      g_radial_pressure_profile[idx] = 0.0;
      g_radial_entropy_profile[idx] = 0.0;
      g_radial_temperature_profile[idx] = 0.0;
#endif
      g_radial_v1_profile[idx] = 0.0;
      g_radial_v2_profile[idx] = 0.0;
      g_radial_v3_profile[idx] = 0.0;
    }
  }

  g_radial_profile_bins = num_bins;
  g_radial_profile_rmin = rmin;
  g_radial_profile_rmax = rmax;
  g_radial_profile_inv_dr = has_extent ? inv_dr : 0.0;
  g_radial_density_ready = true;
  g_radial_density_time = mesh->time;
  g_radial_density_cycle = mesh->ncycle;
}

const std::vector<Real> &GetRadiallyAveragedDensity(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_density_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_density_profile;
}

#if NON_BAROTROPIC_EOS
const std::vector<Real> &GetRadiallyAveragedPressure(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_pressure_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_pressure_profile;
}

const std::vector<Real> &GetRadiallyAveragedEntropy(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_entropy_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_entropy_profile;
}

const std::vector<Real> &GetRadiallyAveragedTemperature(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_temperature_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_temperature_profile;
}
#endif

const std::vector<Real> &GetRadiallyAveragedV1(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_v1_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_v1_profile;
}

const std::vector<Real> &GetRadiallyAveragedV2(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_v2_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_v2_profile;
}

const std::vector<Real> &GetRadiallyAveragedV3(Mesh *mesh) {
  static const std::vector<Real> kEmptyProfile;
  if (mesh == nullptr) {
    return kEmptyProfile;
  }
  const bool size_changed =
      static_cast<int>(g_radial_v3_profile.size()) !=
      RadialProfileBinCount(mesh->mesh_size);
  if (!g_radial_density_ready || size_changed || g_radial_density_cycle != mesh->ncycle
      || g_radial_density_time != mesh->time) {
    ComputeRadialProfiles(mesh);
  }
  return g_radial_v3_profile;
}

constexpr Real kForceFreeSeriesLimit = 1.0e-6;
constexpr Real kSqrtTwo = 1.41421356237309504880;
constexpr Real kInvSqrt4Pi = 0.28209479177387814347;
constexpr Real kPerturbationWarningThreshold = 1.0e6;

constexpr std::array<Real, 7> kGaussNodes = {
    {-0.94910791234275852453, -0.74153118559939443986, -0.40584515137739716691,
     0.0, 0.40584515137739716691, 0.74153118559939443986, 0.94910791234275852453}};

constexpr std::array<Real, 7> kGaussWeights = {
    {0.12948496616886969327, 0.27970539148927666790, 0.38183005050511894495,
     0.41795918367346938776, 0.38183005050511894495, 0.27970539148927666790,
     0.12948496616886969327}};

template <typename Func>
Real AverageProfile(const Func &func, Real a, Real b,
                    const PrecipitatorProfile &profile) {
  if (b <= a) {
    const Real clamped = std::min(std::max(a, profile.MinRadius()), profile.MaxRadius());
    return func(clamped);
  }
  Real half = 0.5 * (b - a);
  Real mid = 0.5 * (b + a);
  Real integral = 0.0;
  for (std::size_t n = 0; n < kGaussNodes.size(); ++n) {
    Real sample = mid + half * kGaussNodes[n];
    sample = std::min(std::max(sample, profile.MinRadius()), profile.MaxRadius());
    integral += kGaussWeights[n] * func(sample);
  }
  return integral * half / (b - a);
}

Real CodeRadiusToCgs(Real radius_code, const Units *units) {
  if (units == nullptr) {
    return 0.0;
  }
  return radius_code * units->code_length_cgs;
}

Real PotentialInCodeUnits(Real radius_code, const Units *units) {
  const Real length_cgs = units->code_length_cgs;
  const Real time_cgs = units->code_time_cgs;
  const Real code_potential_cgs = (length_cgs / (time_cgs * time_cgs)) * length_cgs;
  const Real r_cgs = radius_code * length_cgs;
  return g_profile->Phi(r_cgs) / code_potential_cgs;
}

Real GravityInCodeUnits(Real radius_code, const Units *units) {
  if (!g_profile || units == nullptr) {
    return 0.0;
  }
  const Real code_accel_cgs =
      units->code_length_cgs / (units->code_time_cgs * units->code_time_cgs);
  if (!(code_accel_cgs > 0.0)) {
    return 0.0;
  }
  const Real r_cgs = CodeRadiusToCgs(radius_code, units);
  return g_profile->Gravity(r_cgs) / code_accel_cgs;
}

Real SampleBackgroundRadiusCgs(Real radius_code, const Units *units) {
  if (g_uniform_init == 1) {
    return g_uniform_height;
  }
  if (units == nullptr) {
    return 0.0;
  }
  return radius_code * units->code_length_cgs;
}

Real SampleBackgroundDensityCode(Real radius_code, const Units *units) {
  if (!g_profile || units == nullptr) {
    return 0.0;
  }
  const Real r_cgs = SampleBackgroundRadiusCgs(radius_code, units);
  const Real rho_cgs = g_profile->Density(r_cgs);
  return rho_cgs / units->code_density_cgs;
}

Real SampleBackgroundPressureCode(Real radius_code, const Units *units) {
  if (!g_profile || units == nullptr) {
    return 0.0;
  }
  const Real r_cgs = SampleBackgroundRadiusCgs(radius_code, units);
  const Real pressure_cgs = g_profile->Pressure(r_cgs);
  return pressure_cgs / units->code_pressure_cgs;
}

Real OuterBufferEntropyCgs(Real radius_cgs) {
  if (!(g_outer_buffer_match_entropy_cgs > 0.0)
      || !(g_outer_buffer_inner_radius_cgs > 0.0)) {
    return 0.0;
  }
  if (g_outer_buffer_entropy_slope == 0.0) {
    return g_outer_buffer_match_entropy_cgs;
  }
  const Real radius_ratio =
      std::max(radius_cgs, g_outer_buffer_inner_radius_cgs) /
      g_outer_buffer_inner_radius_cgs;
  return g_outer_buffer_match_entropy_cgs *
         std::pow(radius_ratio, g_outer_buffer_entropy_slope);
}

Real OuterBufferEnthalpyDerivativeCgs(Real radius_cgs, Real enthalpy_cgs) {
  Real deriv = -g_profile->Gravity(radius_cgs);
  if (g_outer_buffer_entropy_slope != 0.0 && radius_cgs > 0.0) {
    deriv += (g_outer_buffer_entropy_slope / g_gamma) * (enthalpy_cgs / radius_cgs);
  }
  return deriv;
}

Real SolveOuterBufferEnthalpyCgs(Real radius_cgs) {
  if (!g_enable_outer_buffer_halo
      || radius_cgs <= g_outer_buffer_inner_radius_cgs) {
    return g_outer_buffer_match_enthalpy_cgs;
  }

  const Real dr_total = radius_cgs - g_outer_buffer_inner_radius_cgs;
  const Real dr_limit =
      0.01 * std::max(g_outer_buffer_inner_radius_cgs, TINY_NUMBER);
  const int nsteps = std::max(
      1, static_cast<int>(std::ceil(dr_total / std::max(dr_limit, TINY_NUMBER))));
  const Real dr = dr_total / static_cast<Real>(nsteps);

  Real r = g_outer_buffer_inner_radius_cgs;
  Real h = g_outer_buffer_match_enthalpy_cgs;
  for (int n = 0; n < nsteps; ++n) {
    const Real k1 = OuterBufferEnthalpyDerivativeCgs(r, h);
    const Real k2 = OuterBufferEnthalpyDerivativeCgs(r + 0.5 * dr, h + 0.5 * dr * k1);
    const Real k3 = OuterBufferEnthalpyDerivativeCgs(r + 0.5 * dr, h + 0.5 * dr * k2);
    const Real k4 = OuterBufferEnthalpyDerivativeCgs(r + dr, h + dr * k3);
    h += (dr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    r += dr;
  }
  return h;
}

void SampleOuterBufferHaloStateCgs(Real radius_cgs, Real *rho_cgs, Real *pressure_cgs) {
  const Real enthalpy_cgs = SolveOuterBufferEnthalpyCgs(radius_cgs);
  if (!(enthalpy_cgs > 0.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Outer buffer halo produced non-positive enthalpy at r=" << radius_cgs
        << " cm. Reduce outer_buffer_inner_radius, shrink the box, or lower "
        << "outer_buffer_entropy_slope.";
    ATHENA_ERROR(msg);
  }

  const Real entropy_cgs = OuterBufferEntropyCgs(radius_cgs);
  if (!(entropy_cgs > 0.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Outer buffer halo produced a non-positive entropy constant.";
    ATHENA_ERROR(msg);
  }

  const Real rho_pow = ((g_gm1 / g_gamma) * enthalpy_cgs) / entropy_cgs;
  if (!(rho_pow > 0.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Outer buffer halo produced a non-positive density factor at r="
        << radius_cgs << " cm.";
    ATHENA_ERROR(msg);
  }

  *rho_cgs = std::pow(rho_pow, 1.0 / g_gm1);
  *pressure_cgs = (g_gm1 / g_gamma) * (*rho_cgs) * enthalpy_cgs;
}

void SampleInitialBackgroundStateCgs(Real radius_code, const Units *units,
                                     Real *rho_cgs, Real *pressure_cgs) {
  if (rho_cgs == nullptr || pressure_cgs == nullptr) {
    return;
  }

  if (g_uniform_init == 1) {
    *rho_cgs = g_profile->Density(g_uniform_height);
    *pressure_cgs = g_profile->Pressure(g_uniform_height);
    return;
  }

  const Real radius_cgs = CodeRadiusToCgs(radius_code, units);
  if (g_enable_outer_buffer_halo && radius_cgs > g_outer_buffer_inner_radius_cgs) {
    SampleOuterBufferHaloStateCgs(radius_cgs, rho_cgs, pressure_cgs);
  } else {
    *rho_cgs = g_profile->Density(radius_cgs);
    *pressure_cgs = g_profile->Pressure(radius_cgs);
  }
}

bool InOuterBufferHalo(Real radius_code) {
  return g_enable_outer_buffer_halo &&
         (radius_code > g_outer_buffer_inner_radius_code);
}

Real OuterBufferHeatCoolTaper(Real radius_code, Real outer_radius_code) {
  if (!g_enable_outer_buffer_halo) {
    return 1.0;
  }
  if (radius_code <= g_outer_buffer_inner_radius_code) {
    return 1.0;
  }
  if (!(outer_radius_code > g_outer_buffer_inner_radius_code)) {
    return 0.0;
  }

  Real frac = (outer_radius_code - radius_code) /
              (outer_radius_code - g_outer_buffer_inner_radius_code);
  frac = std::max(static_cast<Real>(0.0), std::min(frac, static_cast<Real>(1.0)));
  // Smoothstep taper keeps the source term C1-continuous at both ends.
  return frac * frac * (3.0 - 2.0 * frac);
}

void ApplyInnerSponge(MeshBlock *pmb) {
  if (pmb == nullptr || pmb->phydro == nullptr) {
    return;
  }
  if (!g_profile || g_inner_sponge_tau <= 0.0) {
    return;
  }
  if (UsingCartesianCoordinates()) {
    return;
  }
  Mesh *mesh = pmb->pmy_mesh;
  if (mesh == nullptr) {
    return;
  }
  if (mesh->multilevel) {
    return;
  }
  const Real dt = mesh->dt;
  if (!(dt > 0.0)) {
    return;
  }

  const Real r_inner = mesh->mesh_size.x1min;
  const Real r_limit = g_inner_sponge_radius;
  if (!(r_limit > r_inner)) {
    return;
  }
  const Real inv_extent = 1.0 / (r_limit - r_inner);

  auto &prim = pmb->phydro->w;
  auto &cons = pmb->phydro->u;
  Coordinates *pcoord = pmb->pcoord;

  const int is = pmb->is;
  const int ie = pmb->ie;
  const int js = pmb->js;
  const int je = pmb->je;
  const int ks = pmb->ks;
  const int ke = pmb->ke;

  bool modified = false;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real radius_code = CellCenterRadiusCode(pcoord, k, j, i);
        if (radius_code >= r_limit) {
          continue;
        }
        Real radial_weight = (r_limit - radius_code) * inv_extent;
        radial_weight = std::max(static_cast<Real>(0.0),
                                 std::min(radial_weight, static_cast<Real>(1.0)));
        if (radial_weight <= 0.0) {
          continue;
        }
        Real alpha = radial_weight * dt / g_inner_sponge_tau;
        alpha = std::max(static_cast<Real>(0.0), std::min(alpha, static_cast<Real>(1.0)));
        if (alpha <= 0.0) {
          continue;
        }
        prim(IVX, k, j, i) *= (1.0 - alpha);
        prim(IVY, k, j, i) *= (1.0 - alpha);
        prim(IVZ, k, j, i) *= (1.0 - alpha);
        modified = true;
      }
    }
  }

  if (!modified) {
    return;
  }

#if MAGNETIC_FIELDS_ENABLED
  AthenaArray<Real> &bcc = pmb->pfield->bcc;
#else
  AthenaArray<Real> &bcc = prim; // placeholder, bc not used in hydro builds
#endif
  pmb->peos->PrimitiveToConserved(prim, bcc, cons, pcoord, is, ie, js, je, ks, ke);
}

void ApplyOuterSponge(MeshBlock *pmb) {
  if (pmb == nullptr || pmb->phydro == nullptr) {
    return;
  }
  if (!g_profile || g_outer_sponge_tau <= 0.0) {
    return;
  }
  Mesh *mesh = pmb->pmy_mesh;
  if (mesh == nullptr) {
    return;
  }
  if (mesh->multilevel) {
    return;
  }
  const Real dt = mesh->dt;
  if (!(dt > 0.0)) {
    return;
  }

  const Real r_start = g_outer_sponge_inner_radius;
  const Real r_outer = NominalOuterRadiusCode(mesh->mesh_size);
  if (!(r_outer > r_start)) {
    return;
  }
  const Real inv_extent = 1.0 / (r_outer - r_start);

  auto &prim = pmb->phydro->w;
  auto &cons = pmb->phydro->u;
  Coordinates *pcoord = pmb->pcoord;

  const int is = pmb->is;
  const int ie = pmb->ie;
  const int js = pmb->js;
  const int je = pmb->je;
  const int ks = pmb->ks;
  const int ke = pmb->ke;

  bool modified = false;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real radius_code = CellCenterRadiusCode(pcoord, k, j, i);
        if (radius_code <= r_start) {
          continue;
        }
        Real radial_weight = (radius_code - r_start) * inv_extent;
        radial_weight = std::max(static_cast<Real>(0.0),
                                 std::min(radial_weight, static_cast<Real>(1.0)));
        if (radial_weight <= 0.0) {
          continue;
        }
        Real alpha = radial_weight * dt / g_outer_sponge_tau;
        alpha = std::max(static_cast<Real>(0.0), std::min(alpha, static_cast<Real>(1.0)));
        if (alpha <= 0.0) {
          continue;
        }
        prim(IVX, k, j, i) *= (1.0 - alpha);
        prim(IVY, k, j, i) *= (1.0 - alpha);
        prim(IVZ, k, j, i) *= (1.0 - alpha);
        modified = true;
      }
    }
  }

  if (!modified) {
    return;
  }

#if MAGNETIC_FIELDS_ENABLED
  AthenaArray<Real> &bcc = pmb->pfield->bcc;
#else
  AthenaArray<Real> &bcc = prim; // placeholder, bc not used in hydro builds
#endif
  pmb->peos->PrimitiveToConserved(prim, bcc, cons, pcoord, is, ie, js, je, ks, ke);
}

Real ComputeCellTemperature(Real rho_code, Real pressure_code, const Units &units) {
  if (rho_code <= 0.0) {
    return 0.0;
  }
  const Real rho_cgs = rho_code * units.code_density_cgs;
  if (rho_cgs <= 0.0) {
    return 0.0;
  }
  const Real pressure_cgs = pressure_code * units.code_pressure_cgs;
  return (pressure_cgs * g_magic_mmw_cgs) / (Constants::k_boltzmann_cgs * rho_cgs);
}

Real MagicHeatingTaper(Real coord_value) {
  if (g_magic_h_smooth <= 0.0) {
    return 1.0;
  }
  const Real arg = std::abs(coord_value) / g_magic_h_smooth;
  if (arg <= 0.0) {
    return 0.0;
  }
  const Real th = std::tanh(arg);
  return SQR(SQR(th));
}

void UpdateMagicHeatingProfile(Mesh *mesh, Real time, Real dt) {
  if (!g_enable_magic_heating || g_magic_profile_bins <= 0) {
    return;
  }
  if (g_magic_profile_ready && time == g_magic_profile_time && dt == g_magic_profile_dt) {
    return;
  }

  Units *units = mesh->punit;
  if (units == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Units object must be configured before magic heating can run.";
    ATHENA_ERROR(msg);
  }

  const int num_bins = g_magic_profile_bins;
  if (static_cast<int>(g_magic_error_profile.size()) != num_bins) {
    g_magic_error_profile.assign(static_cast<std::size_t>(num_bins), 0.0);
  }

  std::vector<Real> sum(static_cast<std::size_t>(num_bins), 0.0);
  std::vector<Real> volume(static_cast<std::size_t>(num_bins), 0.0);

  const Real rmin = RadialProfileMinCode(mesh->mesh_size);
  const Real rmax = RadialProfileMaxCode(mesh->mesh_size);
  const Real extent = rmax - rmin;
  const bool has_extent = (num_bins > 1) && (extent > 0.0);
  const Real inv_dr = has_extent ? static_cast<Real>(num_bins) / extent : 0.0;
  const Real outer_radius_code = NominalOuterRadiusCode(mesh->mesh_size);

  for (int block = 0; block < mesh->nblocal; ++block) {
    MeshBlock *pmb = mesh->my_blocks(block);
    if (pmb == nullptr || pmb->phydro == nullptr) {
      continue;
    }
    auto &prim = pmb->phydro->w;
    Coordinates *coord = pmb->pcoord;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          const Real r = CellCenterRadiusCode(coord, k, j, i);
          const Real outer_taper = OuterBufferHeatCoolTaper(r, outer_radius_code);
          if (outer_taper <= 0.0) {
            continue;
          }
          const int idx = ClampRadialBinIndex(num_bins, r, rmin, inv_dr);
          const Real rho = prim(IDN, k, j, i);
          const Real pressure = prim(IPR, k, j, i);
          const Real temperature = ComputeCellTemperature(rho, pressure, *units);
          const Real err = temperature - g_magic_target_temperature;
          const Real cell_volume = outer_taper * coord->GetCellVolume(k, j, i);
          sum[static_cast<std::size_t>(idx)] += err * cell_volume;
          volume[static_cast<std::size_t>(idx)] += cell_volume;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, sum.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, volume.data(), num_bins, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  for (int n = 0; n < num_bins; ++n) {
    if (volume[static_cast<std::size_t>(n)] > 0.0) {
      g_magic_error_profile[static_cast<std::size_t>(n)] =
          sum[static_cast<std::size_t>(n)] / volume[static_cast<std::size_t>(n)];
    } else {
      g_magic_error_profile[static_cast<std::size_t>(n)] = 0.0;
    }
  }

  g_magic_profile_ready = true;
  g_magic_profile_time = time;
  g_magic_profile_dt = dt;
  g_magic_profile_x1min = rmin;
  g_magic_profile_inv_dr = has_extent ? inv_dr : 0.0;
}

Real SampleMagicHeatingError(Real radius) {
  if (!g_magic_profile_ready || g_magic_error_profile.empty()) {
    return 0.0;
  }
  if (g_magic_profile_bins <= 1 || g_magic_profile_inv_dr == 0.0) {
    return g_magic_error_profile.front();
  }
  Real idx_f = (radius - g_magic_profile_x1min) * g_magic_profile_inv_dr;
  if (idx_f <= 0.0) {
    return g_magic_error_profile.front();
  }
  const Real max_index = static_cast<Real>(g_magic_profile_bins - 1);
  if (idx_f >= max_index) {
    return g_magic_error_profile.back();
  }
  const int idx = static_cast<int>(idx_f);
  const Real frac = idx_f - static_cast<Real>(idx);
  const Real a = g_magic_error_profile[static_cast<std::size_t>(idx)];
  const Real b = g_magic_error_profile[static_cast<std::size_t>(idx + 1)];
  return a + frac * (b - a);
}

Real HistoryMagicHeatingRate(MeshBlock *pmb, int iout) {
  (void)iout;
  const bool heating_enabled = g_enable_magic_heating && (g_powerlaw_lambda_code > 0.0) &&
                               (g_magic_profile_bins > 0) && (g_magic_c_v > 0.0) &&
                               (g_magic_Kp != 0.0) && g_magic_profile_ready;
  if (!heating_enabled || pmb == nullptr || pmb->phydro == nullptr) {
    return 0.0;
  }

  Units *units = pmb->pmy_mesh->punit;
  if (units == nullptr) {
    return 0.0;
  }

  Coordinates *pcoord = pmb->pcoord;
  AthenaArray<Real> &cons = pmb->phydro->u;
  const Real outer_radius_code = NominalOuterRadiusCode(pmb->pmy_mesh->mesh_size);

  Real heating_rate = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real radius_code = CellCenterRadiusCode(pcoord, k, j, i);
        const Real outer_taper = OuterBufferHeatCoolTaper(radius_code, outer_radius_code);
        if (outer_taper <= 0.0) {
          continue;
        }
        const Real err = SampleMagicHeatingError(radius_code);
        if (err == 0.0) {
          continue;
        }
        const Real taper = outer_taper * MagicHeatingTaper(radius_code);
        if (taper <= 0.0) {
          continue;
        }

        const Real rho = cons(IDN, k, j, i);
        if (rho <= 0.0) {
          continue;
        }

        const Real rho_bg = SampleBackgroundDensityCode(radius_code, units);
        const Real pressure_bg = SampleBackgroundPressureCode(radius_code, units);
        Real inv_t_cool = 0.0;
        if (rho_bg > 0.0 && pressure_bg > 0.0 && g_powerlaw_lambda_code > 0.0) {
          const Real thermal_bg = pressure_bg / g_gm1;
          const Real cooling_strength = g_powerlaw_lambda_code * rho_bg * rho_bg;
          if (thermal_bg > 0.0 && cooling_strength > 0.0) {
            const Real t_cool = thermal_bg / cooling_strength;
            if (t_cool > 0.0) {
              inv_t_cool = 1.0 / t_cool;
            }
          }
        }
        if (inv_t_cool <= 0.0) {
          continue;
        }

        const Real dE_dt = -taper * rho * g_magic_c_v * inv_t_cool * (g_magic_Kp * err);
        heating_rate += dE_dt * pcoord->GetCellVolume(k, j, i);
      }
    }
  }
  return heating_rate;
}

std::string Trim(const std::string &input) {
  const auto first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

void LoadForceFreeParameters() {
  if (g_force_free_loaded) {
    return;
  }
  std::ifstream file(g_force_free_param_file);
  if (!file.is_open()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Unable to open force-free parameter file: " << g_force_free_param_file;
    ATHENA_ERROR(msg);
  }

  bool alpha_found = false;
  std::string line;
  while (std::getline(file, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }
    const std::string key = Trim(line.substr(0, eq_pos));
    const std::string value_str = Trim(line.substr(eq_pos + 1));
    if (value_str.empty()) {
      continue;
    }
    const Real value = std::stod(value_str);
    if (key == "alpha") {
      g_force_free_alpha = value;
      alpha_found = true;
    }
  }
  file.close();

  if (!alpha_found) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Force-free parameter file must provide alpha=" << std::endl
        << "Checked: " << g_force_free_param_file;
    ATHENA_ERROR(msg);
  }
  if (g_force_free_bfield_override) {
    const Real small_radius_bfield_per_amplitude =
        (2.0 / 3.0) * std::abs(g_force_free_alpha);
    if (!(small_radius_bfield_per_amplitude > 0.0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "force_free_bfield_gauss requires a non-zero alpha to define the "
             "small-radius characteristic field.";
      ATHENA_ERROR(msg);
    }
    g_force_free_amplitude =
        g_force_free_bfield_override_value / small_radius_bfield_per_amplitude;
  }
  g_force_free_loaded = true;
}

Real SeriesJ1(Real x) {
  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  return x / 3.0 - x * x2 / 30.0 + x4 * x / 840.0;
}

Real SeriesJ1Derivative(Real x) {
  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  return 1.0 / 3.0 - x2 / 10.0 + x4 / 280.0;
}

#if defined(__cpp_lib_math_special_functions)
inline Real StdSphericalBessel(int l, Real x) {
  return static_cast<Real>(
      std::sph_bessel(static_cast<unsigned>(l), static_cast<double>(x)));
}
#endif

Real SphericalBesselJ0(Real x) {
  const Real ax = std::abs(x);
  if (ax < kForceFreeSeriesLimit) {
    const Real x2 = x * x;
    const Real x4 = x2 * x2;
    return 1.0 - x2 / 6.0 + x4 / 120.0;
  }
#if defined(__cpp_lib_math_special_functions)
  return StdSphericalBessel(0, x);
#else
  return std::sin(x) / x;
#endif
}

Real SphericalBesselJ1(Real x) {
  const Real ax = std::abs(x);
  if (ax < kForceFreeSeriesLimit) {
    return SeriesJ1(x);
  }
#if defined(__cpp_lib_math_special_functions)
  return StdSphericalBessel(1, x);
#else
  return std::sin(x) / (x * x) - std::cos(x) / x;
#endif
}

void ForceFreeRadialTerms(Real r, Real *S, Real *S_over_r, Real *dSdr) {
#if !defined(COORDINATE_SYSTEM)
  (void)r;
  (void)S;
  (void)S_over_r;
  (void)dSdr;
  return;
#else
  LoadForceFreeParameters();
  const Real x = g_force_free_alpha * r;
  Real s_val = 0.0;
  Real dsdr_val = 0.0;
  if (std::abs(x) < kForceFreeSeriesLimit) {
    s_val = g_force_free_amplitude * SeriesJ1(x);
    dsdr_val = g_force_free_amplitude * g_force_free_alpha * SeriesJ1Derivative(x);
  } else {
    const Real j1 = SphericalBesselJ1(x);
    const Real j0 = SphericalBesselJ0(x);
    const Real dj1dx = j0 - 2.0 * j1 / x;
    s_val = g_force_free_amplitude * j1;
    dsdr_val = g_force_free_amplitude * g_force_free_alpha * dj1dx;
  }
  Real s_over_r_val = 0.0;
  if (r > 0.0) {
    s_over_r_val = s_val / r;
  } else {
    s_over_r_val = g_force_free_amplitude * g_force_free_alpha / 3.0;
  }
  if (S != nullptr) {
    *S = s_val;
  }
  if (S_over_r != nullptr) {
    *S_over_r = s_over_r_val;
  }
  if (dSdr != nullptr) {
    *dSdr = dsdr_val;
  }
#endif
}

Real ForceFreeVectorPotentialR(Real r, Real theta) {
  Real S = 0.0;
  ForceFreeRadialTerms(r, &S, nullptr, nullptr);
  return g_force_free_alpha * r * S * std::cos(theta);
}

Real ForceFreeVectorPotentialPhi(Real r, Real theta) {
  Real S = 0.0;
  ForceFreeRadialTerms(r, &S, nullptr, nullptr);
  return S * std::sin(theta);
}

void ForceFreeVectorPotentialCartesian(const CartesianPoint &pos, Real *a1, Real *a2,
                                       Real *a3) {
  const Real r = std::sqrt(SQR(pos.x) + SQR(pos.y) + SQR(pos.z));
  Real s_over_r = 0.0;
  ForceFreeRadialTerms(r, nullptr, &s_over_r, nullptr);
  const Real alpha_z = g_force_free_alpha * pos.z;
  if (a1 != nullptr) {
    *a1 = s_over_r * (alpha_z * pos.x - pos.y);
  }
  if (a2 != nullptr) {
    *a2 = s_over_r * (alpha_z * pos.y + pos.x);
  }
  if (a3 != nullptr) {
    *a3 = s_over_r * (alpha_z * pos.z);
  }
}

void InitializeForceFreeFieldSpherical(MeshBlock *pmb) {
  auto *pcoord = pmb->pcoord;
  auto *pbval = pmb->pbval;
  auto &b = pmb->pfield->b;

  const int nx1 = pmb->block_size.nx1 + 2 * NGHOST + 1;
  const int nx2 = pmb->block_size.nx2 + 2 * NGHOST + 1;
  const int nx3 = pmb->block_size.nx3 + 2 * NGHOST + 1;

  AthenaArray<Real> a1, a3;
  a1.NewAthenaArray(nx3, nx2, nx1);
  a3.NewAthenaArray(nx3, nx2, nx1);

  for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
    for (int j = pmb->js; j <= pmb->je + 1; ++j) {
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        a1(k, j, i) = ForceFreeVectorPotentialR(pcoord->x1v(i), pcoord->x2f(j));
        a3(k, j, i) = ForceFreeVectorPotentialPhi(pcoord->x1f(i), pcoord->x2f(j));
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        b.x1f(k, j, i) = 0.0;
      }
    }
  }
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je + 1; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        b.x2f(k, j, i) = 0.0;
      }
    }
  }
  for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        b.x3f(k, j, i) = 0.0;
      }
    }
  }

  const int ncells1 = pmb->block_size.nx1 + 2 * NGHOST + 2;
  AthenaArray<Real> area, len, len_p1;
  area.NewAthenaArray(ncells1);
  len.NewAthenaArray(ncells1);
  len_p1.NewAthenaArray(ncells1);

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    int jl = pmb->js;
    int ju = pmb->je + 1;
    if (pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
      jl = pmb->js + 1;
    }
    if (pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
      ju = pmb->je;
    }
    for (int j = jl; j <= ju; ++j) {
      pcoord->Face2Area(k, j, pmb->is, pmb->ie, area);
      pcoord->Edge3Length(k, j, pmb->is, pmb->ie + 1, len);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        b.x2f(k, j, i) = -(len(i + 1) * a3(k, j, i + 1) - len(i) * a3(k, j, i)) /
                         area(i);
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pcoord->Face1Area(k, j, pmb->is, pmb->ie + 1, area);
      pcoord->Edge3Length(k, j, pmb->is, pmb->ie + 1, len);
      pcoord->Edge3Length(k, j + 1, pmb->is, pmb->ie + 1, len_p1);
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        if (area(i) > 0.0) {
          b.x1f(k, j, i) =
              (len_p1(i) * a3(k, j + 1, i) - len(i) * a3(k, j, i)) / area(i);
        } else {
          b.x1f(k, j, i) = 0.0;
        }
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pcoord->Face3Area(k, j, pmb->is, pmb->ie, area);
      pcoord->Edge1Length(k, j, pmb->is, pmb->ie, len);
      pcoord->Edge1Length(k, j + 1, pmb->is, pmb->ie, len_p1);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (area(i) > 0.0) {
          b.x3f(k, j, i) -=
              (len_p1(i) * a1(k, j + 1, i) - len(i) * a1(k, j, i)) / area(i);
        } else {
          b.x3f(k, j, i) = 0.0;
        }
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    int jl = pmb->js;
    int ju = pmb->je + 1;
    if (pbval->block_bcs[BoundaryFace::inner_x2] == BoundaryFlag::polar) {
      jl = pmb->js + 1;
    }
    if (pbval->block_bcs[BoundaryFace::outer_x2] == BoundaryFlag::polar) {
      ju = pmb->je;
    }
    for (int j = jl; j <= ju; ++j) {
      pcoord->Face2Area(k, j, pmb->is, pmb->ie, area);
      pcoord->Edge1Length(k, j, pmb->is, pmb->ie, len);
      pcoord->Edge1Length(k + 1, j, pmb->is, pmb->ie, len_p1);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        if (area(i) > 0.0) {
          b.x2f(k, j, i) +=
              (len_p1(i) * a1(k + 1, j, i) - len(i) * a1(k, j, i)) / area(i);
        } else {
          b.x2f(k, j, i) = 0.0;
        }
      }
    }
  }
}

void InitializeForceFreeFieldCartesian(MeshBlock *pmb) {
  auto *pcoord = pmb->pcoord;
  auto &b = pmb->pfield->b;

  const int nx1 = pmb->block_size.nx1 + 2 * NGHOST + 1;
  const int nx2 = pmb->block_size.nx2 + 2 * NGHOST + 1;
  const int nx3 = pmb->block_size.nx3 + 2 * NGHOST + 1;

  AthenaArray<Real> a1, a2, a3;
  a1.NewAthenaArray(nx3, nx2, nx1);
  a2.NewAthenaArray(nx3, nx2, nx1);
  a3.NewAthenaArray(nx3, nx2, nx1);

  for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
    for (int j = pmb->js; j <= pmb->je + 1; ++j) {
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        if (i != pmb->ie + 1) {
          const CartesianPoint pos = Edge1Position(pcoord, k, j, i);
          ForceFreeVectorPotentialCartesian(pos, &a1(k, j, i), nullptr, nullptr);
        }
        if (j != pmb->je + 1) {
          const CartesianPoint pos = Edge2Position(pcoord, k, j, i);
          ForceFreeVectorPotentialCartesian(pos, nullptr, &a2(k, j, i), nullptr);
        }
        if (k != pmb->ke + 1) {
          const CartesianPoint pos = Edge3Position(pcoord, k, j, i);
          ForceFreeVectorPotentialCartesian(pos, nullptr, nullptr, &a3(k, j, i));
        }
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
        b.x1f(k, j, i) = (a3(k, j + 1, i) - a3(k, j, i)) / pcoord->dx2f(j) -
                         (a2(k + 1, j, i) - a2(k, j, i)) / pcoord->dx3f(k);
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je + 1; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        b.x2f(k, j, i) = (a1(k + 1, j, i) - a1(k, j, i)) / pcoord->dx3f(k) -
                         (a3(k, j, i + 1) - a3(k, j, i)) / pcoord->dx1f(i);
      }
    }
  }

  for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        b.x3f(k, j, i) = (a2(k, j, i + 1) - a2(k, j, i)) / pcoord->dx1f(i) -
                         (a1(k, j + 1, i) - a1(k, j, i)) / pcoord->dx2f(j);
      }
    }
  }
}

void InitializeForceFreeField(MeshBlock *pmb) {
#if MAGNETIC_FIELDS_ENABLED
  LoadForceFreeParameters();
  if (UsingCartesianCoordinates()) {
    InitializeForceFreeFieldCartesian(pmb);
  } else {
    InitializeForceFreeFieldSpherical(pmb);
  }
#else
  (void)pmb;
#endif
}

int PerturbationCoefficientCount() {
  return (g_pert_lmax + 1) * (g_pert_lmax + 2) / 2;
}

inline std::size_t PerturbationCoeffIndex(int l, int m) {
  const std::size_t ls = static_cast<std::size_t>(l);
  const std::size_t ms = static_cast<std::size_t>(m);
  return ls * (ls + 1u) / 2u + ms;
}

void ComputeNormalizedAssociatedLegendre(int lmax, Real cos_theta,
                                         std::vector<Real> &output) {
  if (lmax < 0) {
    output.clear();
    return;
  }
  const std::size_t required =
      static_cast<std::size_t>(lmax + 1) * static_cast<std::size_t>(lmax + 2) / 2;
  if (output.size() != required) {
    output.resize(required);
  }

  const Real sin_theta =
      std::sqrt(std::max(static_cast<Real>(0.0),
                         static_cast<Real>(1.0) - cos_theta * cos_theta));
  output[PerturbationCoeffIndex(0, 0)] = kInvSqrt4Pi;

  Real prev_diag = output[PerturbationCoeffIndex(0, 0)];
  for (int m = 1; m <= lmax; ++m) {
    const std::size_t idx = PerturbationCoeffIndex(m, m);
    const Real factor = -std::sqrt((2.0 * m + 1.0) / (2.0 * m));
    output[idx] = factor * sin_theta * prev_diag;
    prev_diag = output[idx];
  }

  for (int m = 0; m < lmax; ++m) {
    const std::size_t idx_diag = PerturbationCoeffIndex(m, m);
    const std::size_t idx_next = PerturbationCoeffIndex(m + 1, m);
    output[idx_next] = std::sqrt(2.0 * m + 3.0) * cos_theta * output[idx_diag];
  }

  for (int m = 0; m <= lmax; ++m) {
    for (int l = m + 2; l <= lmax; ++l) {
      const Real ll = static_cast<Real>(l);
      const Real mm = static_cast<Real>(m);
      const Real denom = ll - mm;
      const Real term1 = (2.0 * ll - 1.0) / denom;
      const Real term2 = (ll + mm - 1.0) / denom;
      const Real ratio1_term = ((2.0 * ll + 1.0) / (2.0 * ll - 1.0)) *
                               ((ll - mm) / (ll + mm));
      const Real ratio1 = std::sqrt(std::max(static_cast<Real>(0.0), ratio1_term));
      const Real ratio2_term = ((2.0 * ll + 1.0) / (2.0 * ll - 3.0)) *
                               ((ll - mm) * (ll - mm - 1.0)) /
                               ((ll + mm) * (ll + mm - 1.0));
      const Real ratio2 = std::sqrt(std::max(static_cast<Real>(0.0), ratio2_term));
      const std::size_t idx = PerturbationCoeffIndex(l, m);
      const std::size_t idx_lm1 = PerturbationCoeffIndex(l - 1, m);
      const std::size_t idx_lm2 = PerturbationCoeffIndex(l - 2, m);
      output[idx] = term1 * ratio1 * cos_theta * output[idx_lm1] -
                    term2 * ratio2 * output[idx_lm2];
    }
  }
}

bool PerturbationsActive() {
  return g_enable_density_perturbations && (g_pert_sigma != 0.0) &&
         (g_pert_radial_modes > 0);
}

Real ScaleRadius(Real r) {
  const Real denom = g_pert_radius_max - g_pert_radius_min;
  if (denom <= 0.0) {
    return 0.0;
  }
  Real scaled = (r - g_pert_radius_min) / denom;
  if (scaled < 0.0) {
    scaled = 0.0;
  } else if (scaled > 1.0) {
    scaled = 1.0;
  }
  return scaled;
}

Real RandomUniform01() {
  constexpr double inv = 1.0 / 9007199254740992.0;
  g_pert_state = g_pert_state * 6364136223846793005ULL + 1ULL;
  const std::uint64_t mantissa = (g_pert_state >> 11) & 0x1fffffffffffffULL;
  return static_cast<Real>(static_cast<double>(mantissa) * inv);
}

Real RandomSymmetric() { return 2.0 * RandomUniform01() - 1.0; }

Real SmallXSphericalBessel(int l, Real x) {
  if (l == 0) {
    const Real x2 = x * x;
    return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
  }
  if (l == 1) {
    const Real x2 = x * x;
    return x / 3.0 - x * x2 / 30.0 + x2 * x2 * x / 840.0;
  }
  Real result = 1.0;
  for (int k = 1; k <= l; ++k) {
    result *= static_cast<Real>(2 * k + 1);
  }
  result = std::pow(std::abs(x), static_cast<Real>(l)) / result;
  if (x < 0.0 && (l % 2) == 1) {
    result = -result;
  }
  return result;
}

Real NoiseSphericalBessel(int l, Real x) {
  const Real ax = std::abs(x);
  if (ax < g_pert_small_kr_threshold) {
    return SmallXSphericalBessel(l, x);
  }
#if defined(__cpp_lib_math_special_functions)
  return StdSphericalBessel(l, x);
#else
  if (l == 0) {
    return SphericalBesselJ0(x);
  }
  if (l == 1) {
    return SphericalBesselJ1(x);
  }
  Real jm1 = SphericalBesselJ0(x);
  Real jcurr = SphericalBesselJ1(x);
  for (int ell = 1; ell < l; ++ell) {
    const Real jp1 = ((2.0 * ell + 1.0) / x) * jcurr - jm1;
    if (!std::isfinite(jp1) || std::abs(jp1) > 1.0e12) {
      std::cout << "### Warning in precipitator.cpp: spherical Bessel recurrence "
                << "overflow (l=" << l << ", ell=" << ell << ", x=" << x
                << ", jm1=" << jm1 << ", jcurr=" << jcurr << ", jp1=" << jp1
                << ")\n";
    }
    jm1 = jcurr;
    jcurr = jp1;
  }
  return jcurr;
#endif
}

Real EvalSphHarmNoise(Real r, Real theta, Real phi) {
  if (!g_pert_ready) {
    return 0.0;
  }
  const Real cos_t = std::cos(theta);
  const Real r_scaled = ScaleRadius(r);
  Real noise = 0.0;
  const std::size_t num_coeff =
      static_cast<std::size_t>(PerturbationCoefficientCount());
  std::size_t idx_base = 0;
  static thread_local std::vector<Real> legendre_values;
  ComputeNormalizedAssociatedLegendre(g_pert_lmax, cos_t, legendre_values);

  for (int n = 0; n < g_pert_radial_modes; ++n) {
    std::size_t idx = idx_base;
    const Real kr = g_pert_k_values[n] * r_scaled;
    for (int l = 0; l <= g_pert_lmax; ++l) {
      const Real level_scale = 1.0 / std::sqrt(2.0 * l + 1.0);
      const Real radial_fn = NoiseSphericalBessel(l, kr);
      const Real radial_val = radial_fn * g_pert_radial_weight[n];
      if (!std::isfinite(radial_fn) ||
          std::abs(radial_val) > kPerturbationWarningThreshold) {
        std::cout << "### Warning in precipitator.cpp: large radial term "
                  << "(l=" << l << ", mode=" << n << ", kr=" << kr
                  << ", r=" << r << ", r_scaled=" << r_scaled
                  << ", radial_fn=" << radial_fn
                  << ", weight=" << g_pert_radial_weight[n]
                  << ", radial_val=" << radial_val << ")\n";
      }
      const Real plm0 = legendre_values[PerturbationCoeffIndex(l, 0)];
      const Real coeff0 = g_pert_coeff_cos[idx++];
      noise += coeff0 * level_scale * plm0 * radial_val;

      for (int m = 1; m <= l; ++m) {
        const Real plm = legendre_values[PerturbationCoeffIndex(l, m)];
        const Real base = level_scale * plm * radial_val;
        const Real coeff_c = g_pert_coeff_cos[idx];
        const Real coeff_s = g_pert_coeff_sin[idx];
        const Real phase = static_cast<Real>(m) * phi;
        noise += kSqrtTwo * base *
                 (coeff_c * std::cos(phase) + coeff_s * std::sin(phase));
        ++idx;
      }
    }
    idx_base += static_cast<std::size_t>(num_coeff);
  }
  return noise;
}

void InitializePerturbationTables(const Mesh &mesh) {
  if (!PerturbationsActive()) {
    return;
  }
  g_pert_radius_min = RadialProfileMinCode(mesh.mesh_size);
  g_pert_radius_max = RadialProfileMaxCode(mesh.mesh_size);
  if (g_pert_ready) {
    return;
  }

  const int num_coeff = PerturbationCoefficientCount();
  if (num_coeff <= 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Invalid perturbation configuration: lmax=" << g_pert_lmax;
    ATHENA_ERROR(msg);
  }

  const std::size_t total_coeff =
      static_cast<std::size_t>(g_pert_radial_modes) *
      static_cast<std::size_t>(num_coeff);
  g_pert_coeff_cos.assign(total_coeff, 0.0);
  g_pert_coeff_sin.assign(total_coeff, 0.0);
  g_pert_k_values.assign(g_pert_radial_modes, 0.0);
  g_pert_radial_weight.assign(g_pert_radial_modes, 0.0);

  g_pert_state = g_pert_seed;
  Real kmin = g_pert_kmin;
  Real kmax = g_pert_kmax;
  if (kmin <= 0.0) {
    kmin = 0.5;
  }
  if (kmax <= kmin) {
    kmax = kmin + 1.0;
  }
  const Real span = kmax - kmin;
  const Real denom =
      (g_pert_radial_modes > 1) ? static_cast<Real>(g_pert_radial_modes - 1)
                                : 1.0;
  Real dk_eff = span / denom;
  if (dk_eff <= 0.0) {
    dk_eff = kmin;
  }

  for (int n = 0; n < g_pert_radial_modes; ++n) {
    const Real kval =
        (g_pert_radial_modes > 1) ? (kmin + n * dk_eff) : kmin;
    g_pert_k_values[n] = kval;
    g_pert_radial_weight[n] = std::sqrt(kval * kval * dk_eff);
    for (int i = 0; i < num_coeff; ++i) {
      const std::size_t idx =
          static_cast<std::size_t>(n) * static_cast<std::size_t>(num_coeff) +
          static_cast<std::size_t>(i);
      g_pert_coeff_cos[idx] = RandomSymmetric();
      g_pert_coeff_sin[idx] = RandomSymmetric();
    }
  }
  g_pert_ready = true;
}

void ApplyDensityPerturbations(MeshBlock *pmb) {
  if (!PerturbationsActive()) {
    return;
  }
  InitializePerturbationTables(*pmb->pmy_mesh);
  if (!g_pert_ready) {
    return;
  }

  auto &prim = pmb->phydro->w;
  Coordinates *coord = pmb->pcoord;

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const SphericalPoint sph = CellCenterSpherical(coord, k, j, i);
        if (InOuterBufferHalo(sph.r)) {
          continue;
        }
        const Real delta = g_pert_sigma * EvalSphHarmNoise(sph.r, sph.theta, sph.phi);
        if (!std::isfinite(delta) ||
            std::abs(delta) > kPerturbationWarningThreshold) {
          std::cout << "### Warning in precipitator.cpp: large density perturbation "
                    << "delta=" << delta << " at (r=" << sph.r
                    << ", theta=" << sph.theta << ", phi=" << sph.phi << ")\n";
        }

        const Real rho0 = prim(IDN, k, j, i);
        const Real rho_new = rho0 * (1.0 + delta);
        if (rho_new <= 0.0) {
          std::cout << "### Warning in precipitator.cpp: density perturbation produced "
                    << "rho <= 0 (rho0=" << rho0 << ", delta=" << delta
                    << ", r=" << sph.r << ", theta=" << sph.theta
                    << ", phi=" << sph.phi << ")\n";
        }
        prim(IDN, k, j, i) = rho_new;
      }
    }
  }
}

Real CellCenterRadiusCode(const Coordinates *coord, int k, int j, int i) {
  return CellCenterSpherical(coord, k, j, i).r;
}

//----------------------------------------------------------------------------------------
// Polar mesh generator (θ-spacing)
//----------------------------------------------------------------------------------------
Real PrecipitatorThetaGrid(Real x2, RegionSize rs) {
  // Logical coordinate centered at zero enables symmetric stretching.
  const Real t = 2.0 * x2 - 1.0;
  // Coefficients yield Δθ_pole ≈ 2 Δθ_eq without excessive compression.
  constexpr Real linear = 0.375;
  constexpr Real cubic = 0.125;
  const Real w = 0.5 + linear * t + cubic * t * t * t;
  return rs.x2min + (rs.x2max - rs.x2min) * w;
}

} // namespace

// Forward declaration so we can enroll it before the definition appears.
void PrecipitatorGravity(MeshBlock *pmb, const Real time, const Real dt,
                         const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc,
                         AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Initialize precipitator-specific mesh data and source term
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  g_gamma = pin->GetReal("hydro", "gamma");
  g_gm1 = g_gamma - 1.0;
  Units *units = punit;

  const Real x2rat = pin->GetOrAddReal("mesh", "x2rat", 1.0);
  if (x2rat < 0.0 && UsingSphericalPolarCoordinates()) {
    EnrollUserMeshGenerator(X2DIR, PrecipitatorThetaGrid);
  }

  const std::string profile_filename =
      pin->GetString("precipitator", "hse_profile_filename");
  g_profile = std::unique_ptr<PrecipitatorProfile>(
      new PrecipitatorProfile(profile_filename));

  g_uniform_init = pin->GetOrAddInteger("precipitator", "uniform_init", 0);
  if (g_uniform_init == 1) {
    g_uniform_height = pin->GetReal("precipitator", "uniform_init_height");
  }

  const Real nominal_outer_radius = NominalOuterRadiusCode(mesh_size);
  const Real radial_profile_min = RadialProfileMinCode(mesh_size);
  const Real radial_profile_max = RadialProfileMaxCode(mesh_size);

  g_force_free_param_file =
      pin->GetOrAddString("precipitator", "force_free_param_file",
                          "inputs/force_free_params.txt");
#if defined(COORDINATE_SYSTEM)
  if (!(UsingSphericalPolarCoordinates() || UsingCartesianCoordinates())) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Force-free magnetic configuration requires spherical_polar or cartesian "
        << "coordinates.";
    ATHENA_ERROR(msg);
  }
#endif
  const Real force_free_bfield_gauss =
      pin->GetOrAddReal("precipitator", "force_free_bfield_gauss", -1.0);
  if (force_free_bfield_gauss > 0.0) {
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object must be configured before force_free_bfield_gauss "
          << "can be used.";
      ATHENA_ERROR(msg);
    }
    const Real code_bfield_cgs = units->code_magneticfield_cgs;
    if (!(code_bfield_cgs > 0.0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Invalid code magnetic-field unit when processing force_free_bfield_gauss.";
      ATHENA_ERROR(msg);
    }
    g_force_free_bfield_override = true;
    g_force_free_bfield_override_value = force_free_bfield_gauss / code_bfield_cgs;
  } else {
    g_force_free_bfield_override = false;
  }

  const Real He_mass_fraction = pin->GetOrAddReal("hydro", "He_mass_fraction", 0.25);
  const Real hydrogen_mass_fraction = 1.0 - He_mass_fraction;
  g_magic_mu = 1.0 / (He_mass_fraction * 0.75 + hydrogen_mass_fraction * 2.0);
  g_magic_mmw_cgs = g_magic_mu * Constants::hydrogen_mass_cgs;

  g_enable_powerlaw_cooling =
      (pin->GetOrAddInteger("precipitator", "enable_powerlaw_cooling", 0) != 0);
  const std::string heating_mode =
      pin->GetOrAddString("precipitator", "enable_heating", "none");
  g_enable_magic_heating = (heating_mode == "magic");
  g_inner_sponge_radius =
      pin->GetOrAddReal("precipitator", "inner_sponge_radius", radial_profile_min);
  g_inner_sponge_tau =
      std::max(static_cast<Real>(0.0),
               pin->GetOrAddReal("precipitator", "inner_sponge_tau", 0.0));
  g_outer_sponge_inner_radius =
      pin->GetOrAddReal("precipitator", "outer_sponge_inner_radius",
                        nominal_outer_radius);
  g_outer_sponge_tau =
      std::max(static_cast<Real>(0.0),
               pin->GetOrAddReal("precipitator", "outer_sponge_tau", 0.0));
  g_enable_outer_buffer_halo =
      (pin->GetOrAddInteger("precipitator", "enable_outer_buffer_halo", 0) != 0);
  g_outer_buffer_inner_radius_code =
      pin->GetOrAddReal("precipitator", "outer_buffer_inner_radius",
                        nominal_outer_radius);
  g_outer_buffer_entropy_slope =
      pin->GetOrAddReal("precipitator", "outer_buffer_entropy_slope", 0.0);
  if (g_enable_outer_buffer_halo) {
    if (g_uniform_init == 1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Outer buffer halo is incompatible with uniform_init=1.";
      ATHENA_ERROR(msg);
    }
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object must be configured before enabling the outer buffer halo.";
      ATHENA_ERROR(msg);
    }
    if (!(g_outer_buffer_inner_radius_code > 0.0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "outer_buffer_inner_radius must be positive when "
             "enable_outer_buffer_halo=1.";
      ATHENA_ERROR(msg);
    }
    if (g_outer_buffer_entropy_slope < 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "outer_buffer_entropy_slope must be >= 0.";
      ATHENA_ERROR(msg);
    }
    g_outer_buffer_inner_radius_cgs =
        CodeRadiusToCgs(g_outer_buffer_inner_radius_code, units);
    g_outer_buffer_match_density_cgs =
        g_profile->Density(g_outer_buffer_inner_radius_cgs);
    g_outer_buffer_match_pressure_cgs =
        g_profile->Pressure(g_outer_buffer_inner_radius_cgs);
    if (!(g_outer_buffer_match_density_cgs > 0.0) ||
        !(g_outer_buffer_match_pressure_cgs > 0.0)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Outer buffer halo match state must have positive density and pressure.";
      ATHENA_ERROR(msg);
    }
    g_outer_buffer_match_entropy_cgs =
        g_outer_buffer_match_pressure_cgs /
        std::pow(g_outer_buffer_match_density_cgs, g_gamma);
    g_outer_buffer_match_enthalpy_cgs =
        (g_gamma / g_gm1) * g_outer_buffer_match_pressure_cgs /
        g_outer_buffer_match_density_cgs;
  } else {
    g_outer_buffer_inner_radius_cgs = 0.0;
    g_outer_buffer_match_density_cgs = 0.0;
    g_outer_buffer_match_pressure_cgs = 0.0;
    g_outer_buffer_match_entropy_cgs = 0.0;
    g_outer_buffer_match_enthalpy_cgs = 0.0;
  }

  const bool need_powerlaw_coeff =
      g_enable_powerlaw_cooling || g_enable_magic_heating;
  g_powerlaw_lambda_code = 0.0;
  if (need_powerlaw_coeff) {
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object must be configured before enabling cooling/heating.";
      ATHENA_ERROR(msg);
    }
    // Default Lambda matches the AthenaPK precipitator cooling table (1e-22 erg cm^3/s).
    const Real lambda_cgs =
        pin->GetOrAddReal("precipitator", "powerlaw_lambda_cgs", 1.0e-22);
    if (lambda_cgs <= 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "powerlaw_lambda_cgs must be positive when cooling/heating is enabled.";
      ATHENA_ERROR(msg);
    }
    const Real density_cgs = units->code_density_cgs;
    const Real energy_density_cgs = units->code_energydensity_cgs;
    const Real time_cgs = units->code_time_cgs;
    const Real hydrogen_mass_cgs = Constants::hydrogen_mass_cgs;
    // Convert number-density cooling coefficient (erg cm^3/s) -> mass-density form.
    const Real lambda_mass_cgs =
        lambda_cgs * SQR(hydrogen_mass_fraction / hydrogen_mass_cgs);
    g_powerlaw_lambda_code =
        lambda_mass_cgs * SQR(density_cgs) * time_cgs / energy_density_cgs;
  }

  if (g_enable_magic_heating) {
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object must be configured before enabling magic heating.";
      ATHENA_ERROR(msg);
    }
    if (mesh_size.nx3 <= 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Magic heating requires mesh_size.nx3 > 0 to define a vertical profile.";
      ATHENA_ERROR(msg);
    }
    g_magic_target_temperature =
        pin->GetOrAddReal("precipitator", "thermostat_temperature", 1.0e7);
    g_magic_Kp = pin->GetOrAddReal("precipitator", "thermostat_Kp", 0.0);
    g_magic_h_smooth =
        pin->GetOrAddReal("precipitator", "h_smooth_heatcool", 1.0);
    if (g_magic_h_smooth <= 0.0) {
      g_magic_h_smooth = 1.0;
    }
    g_magic_mmw_code = g_magic_mmw_cgs * units->gram_code;
    g_magic_c_v = (units->k_boltzmann_code / g_magic_mmw_code) / g_gm1;
    g_magic_profile_bins = RadialProfileBinCount(mesh_size);
    g_magic_error_profile.assign(static_cast<std::size_t>(g_magic_profile_bins), 0.0);
    g_magic_profile_ready = false;
    g_magic_profile_time = std::numeric_limits<Real>::quiet_NaN();
    g_magic_profile_dt = std::numeric_limits<Real>::quiet_NaN();
    g_magic_profile_x1min = radial_profile_min;
    g_magic_profile_inv_dr = 0.0;
  } else {
    g_magic_error_profile.clear();
    g_magic_profile_bins = 0;
    g_magic_profile_ready = false;
    g_magic_profile_time = std::numeric_limits<Real>::quiet_NaN();
    g_magic_profile_dt = std::numeric_limits<Real>::quiet_NaN();
  }

  g_pert_radius_min = radial_profile_min;
  g_pert_radius_max = radial_profile_max;
  const int pert_flag =
      pin->GetOrAddInteger("precipitator", "enable_fourier_bessel_perturbations", 0);
  g_pert_sigma = pin->GetOrAddReal("precipitator", "perturbation_sigma", 0.01);
  g_pert_lmax = pin->GetOrAddInteger("precipitator", "perturbation_lmax", 12);
  g_pert_radial_modes =
      pin->GetOrAddInteger("precipitator", "perturbation_radial_modes", 16);
  g_pert_kmin =
      pin->GetOrAddReal("precipitator", "perturbation_kmin", 0.0);
  g_pert_kmax =
      pin->GetOrAddReal("precipitator", "perturbation_kmax", 0.0);
  g_pert_small_kr_threshold =
      pin->GetOrAddReal("precipitator", "perturbation_small_kr_threshold", 1.0);
  const std::string seed_string =
      pin->GetOrAddString("precipitator", "perturbation_seed",
                          "88172645463393265");
  try {
    g_pert_seed = static_cast<std::uint64_t>(std::stoull(seed_string));
  } catch (const std::exception &ex) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Invalid perturbation_seed value: " << seed_string << std::endl
        << ex.what();
    ATHENA_ERROR(msg);
  }
  g_enable_density_perturbations =
      (pert_flag != 0) && (g_pert_sigma != 0.0);
  g_pert_ready = false;
  if (g_enable_density_perturbations) {
    if (g_pert_lmax < 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "perturbation_lmax must be >= 0.";
      ATHENA_ERROR(msg);
    }
    if (g_pert_radial_modes <= 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "perturbation_radial_modes must be > 0.";
      ATHENA_ERROR(msg);
    }
    if (g_pert_small_kr_threshold <= 0.0) {
      g_pert_small_kr_threshold = 1.0;
    }
  } else {
    g_pert_ready = false;
    g_pert_coeff_cos.clear();
    g_pert_coeff_sin.clear();
    g_pert_k_values.clear();
    g_pert_radial_weight.clear();
  }

  if (g_enable_magic_heating) {
    AllocateUserHistoryOutput(1);
    EnrollUserHistoryOutput(0, HistoryMagicHeatingRate, "magic_heat_rate");
  }

  EnrollUserExplicitSourceFunction(PrecipitatorGravity);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Populate hydrostatic background density/pressure (no perturbations or heating)
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (!g_profile) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Precipitator profile not initialized before ProblemGenerator.";
    ATHENA_ERROR(msg);
  }

  Units *units = pmy_mesh->punit;
  if (units == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Units object is not initialized.";
    ATHENA_ERROR(msg);
  }

  const int il = is;
  const int iu = ie;
  const int jl = js;
  const int ju = je;
  const int kl = ks;
  const int ku = ke;

  const bool use_cell_center_background =
      g_enable_outer_buffer_halo || UsingCartesianCoordinates();
  const int nx1 = iu - il + 1;
  std::vector<Real> rho_profile(nx1);
#if NON_BAROTROPIC_EOS
  std::vector<Real> pressure_profile(nx1);
#endif

  const Real length_cgs = units->code_length_cgs;
  const Real density_cgs = units->code_density_cgs;
#if NON_BAROTROPIC_EOS
  const Real pressure_cgs = units->code_pressure_cgs;
#endif

  if (!use_cell_center_background) {
    for (int i = il; i <= iu; ++i) {
      const Real rmin_cgs = pcoord->x1f(i) * length_cgs;
      const Real rmax_cgs = pcoord->x1f(i + 1) * length_cgs;

      Real rho_cgs = 0.0;
      Real prs_cgs = 0.0;
      if (g_uniform_init == 1) {
        rho_cgs = g_profile->Density(g_uniform_height);
        prs_cgs = g_profile->Pressure(g_uniform_height);
      } else {
        auto rho_fn = [&](Real r) { return g_profile->Density(r); };
        auto prs_fn = [&](Real r) { return g_profile->Pressure(r); };
        rho_cgs = AverageProfile(rho_fn, rmin_cgs, rmax_cgs, *g_profile);
        prs_cgs = AverageProfile(prs_fn, rmin_cgs, rmax_cgs, *g_profile);
      }

      const int idx = i - il;
      rho_profile[idx] = rho_cgs / density_cgs;
#if NON_BAROTROPIC_EOS
      pressure_profile[idx] = prs_cgs / pressure_cgs;
#endif
    }
  }

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real rho_code = 0.0;
#if NON_BAROTROPIC_EOS
        Real pressure_code_local = 0.0;
#endif
        if (use_cell_center_background) {
          const Real radius_code = CellCenterRadiusCode(pcoord, k, j, i);
          Real rho_cgs = 0.0;
          Real prs_cgs = 0.0;
          SampleInitialBackgroundStateCgs(radius_code, units, &rho_cgs, &prs_cgs);
          rho_code = rho_cgs / density_cgs;
#if NON_BAROTROPIC_EOS
          pressure_code_local = prs_cgs / pressure_cgs;
#endif
        } else {
          const int idx = i - il;
          rho_code = rho_profile[idx];
#if NON_BAROTROPIC_EOS
          pressure_code_local = pressure_profile[idx];
#endif
        }
        phydro->w(IDN, k, j, i) = rho_code;
#if NON_BAROTROPIC_EOS
        phydro->w(IPR, k, j, i) = pressure_code_local;
#endif
        phydro->w(IVX, k, j, i) = 0.0;
        phydro->w(IVY, k, j, i) = 0.0;
        phydro->w(IVZ, k, j, i) = 0.0;
      }
    }
  }

  ApplyDensityPerturbations(this);

  auto require_finite = [&](const char *label, int k, int j, int i, Real value) {
    if (!std::isfinite(value)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Non-finite value in " << label << " at cell ("
          << k << "," << j << "," << i << ")";
      ATHENA_ERROR(msg);
    }
  };

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        require_finite("density", k, j, i, phydro->w(IDN, k, j, i));
#if NON_BAROTROPIC_EOS
        require_finite("pressure", k, j, i, phydro->w(IPR, k, j, i));
#endif
        require_finite("velocity v1", k, j, i, phydro->w(IVX, k, j, i));
        require_finite("velocity v2", k, j, i, phydro->w(IVY, k, j, i));
        require_finite("velocity v3", k, j, i, phydro->w(IVZ, k, j, i));
      }
    }
  }

#if MAGNETIC_FIELDS_ENABLED
  InitializeForceFreeField(this);
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js,
                                     je, ks, ke);

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie + 1; ++i) {
        require_finite("B^1 face", k, j, i, pfield->b.x1f(k, j, i));
      }
    }
  }
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je + 1; ++j) {
      for (int i = is; i <= ie; ++i) {
        require_finite("B^2 face", k, j, i, pfield->b.x2f(k, j, i));
      }
    }
  }
  for (int k = ks; k <= ke + 1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        require_finite("B^3 face", k, j, i, pfield->b.x3f(k, j, i));
      }
    }
  }
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        require_finite("B^1 cell", k, j, i, pfield->bcc(IB1, k, j, i));
        require_finite("B^2 cell", k, j, i, pfield->bcc(IB2, k, j, i));
        require_finite("B^3 cell", k, j, i, pfield->bcc(IB3, k, j, i));
      }
    }
  }
#endif

#if MAGNETIC_FIELDS_ENABLED
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                             js, je, ks, ke);
#else
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke + 1, je + 1, ie + 1);
  bb.ZeroClear();
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);
#endif

  return;
}

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Allocate auxiliary cooling-time output when power-law cooling is active
//========================================================================================
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  int noutputs = 0;
  if (g_enable_powerlaw_cooling) {
    ++noutputs;
  }
#if MAGNETIC_FIELDS_ENABLED
  ++noutputs;  // reserve space for divB diagnostic
#endif

  ++noutputs;  // density contrast output
#if NON_BAROTROPIC_EOS
  ++noutputs;  // temperature output
#endif

  ++noutputs;  // gravitational potential
  ++noutputs;  // hydrostatic pressure reference
#if NON_BAROTROPIC_EOS
  ++noutputs;  // entropy
  ++noutputs;  // delta pressure
  ++noutputs;  // delta entropy
  ++noutputs;  // delta temperature
#endif

  noutputs += 3;  // dv components
#if NON_BAROTROPIC_EOS
  ++noutputs;  // Mach number
#endif
#if MAGNETIC_FIELDS_ENABLED
  ++noutputs;  // plasma beta
#endif

  if (noutputs == 0) {
    return;
  }

  AllocateUserOutputVariables(noutputs);
  int idx = 0;
  if (g_enable_powerlaw_cooling) {
    SetUserOutputVariableName(idx++, "tcool_myr");
  }
#if MAGNETIC_FIELDS_ENABLED
  SetUserOutputVariableName(idx++, "divB");
#endif
  SetUserOutputVariableName(idx++, kDensityContrastName);
#if NON_BAROTROPIC_EOS
  SetUserOutputVariableName(idx++, kTemperatureOutputName);
#endif
  SetUserOutputVariableName(idx++, "grav_phi");
  SetUserOutputVariableName(idx++, "pressure_hse");
#if NON_BAROTROPIC_EOS
  SetUserOutputVariableName(idx++, "entropy_K");
  SetUserOutputVariableName(idx++, "delta_pressure_over_pressure_bar");
  SetUserOutputVariableName(idx++, "delta_entropy_over_entropy_bar");
  SetUserOutputVariableName(idx++, "delta_temperature_over_temperature_bar");
#endif
  SetUserOutputVariableName(idx++, "dv1_kms");
  SetUserOutputVariableName(idx++, "dv2_kms");
  SetUserOutputVariableName(idx++, "dv3_kms");
#if NON_BAROTROPIC_EOS
  SetUserOutputVariableName(idx++, "mach_sonic");
#endif
#if MAGNETIC_FIELDS_ENABLED
  SetUserOutputVariableName(idx++, "plasma_beta");
#endif
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Fill auxiliary cooling-time output in Myr
//========================================================================================
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  if (nuser_out_var == 0) {
    return;
  }

  auto &prim = phydro->w;
  Units *units = pmy_mesh->punit;
  if (units == nullptr) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Units object is not initialized.";
    ATHENA_ERROR(msg);
  }
  Mesh *mesh = pmy_mesh;
  int next_index = 0;
  const int tcool_index = g_enable_powerlaw_cooling ? next_index++ : -1;
#if MAGNETIC_FIELDS_ENABLED
  const int divb_index = next_index++;
#else
  constexpr int divb_index = -1;
#endif
  const int density_contrast_index = next_index++;
#if NON_BAROTROPIC_EOS
  const int temperature_index = next_index++;
#else
  constexpr int temperature_index = -1;
#endif
  const int grav_phi_index = next_index++;
  const int pressure_hse_index = next_index++;
#if NON_BAROTROPIC_EOS
  const int entropy_index = next_index++;
  const int delta_pressure_index = next_index++;
  const int delta_entropy_index = next_index++;
  const int delta_temperature_index = next_index++;
#else
  constexpr int entropy_index = -1;
  constexpr int delta_pressure_index = -1;
  constexpr int delta_entropy_index = -1;
  constexpr int delta_temperature_index = -1;
#endif
  const int dv1_index = next_index++;
  const int dv2_index = next_index++;
  const int dv3_index = next_index++;
#if NON_BAROTROPIC_EOS
  const int mach_index = next_index++;
#else
  constexpr int mach_index = -1;
#endif
#if MAGNETIC_FIELDS_ENABLED
  const int plasma_beta_index = next_index++;
#else
  constexpr int plasma_beta_index = -1;
#endif

  const std::vector<Real> &rho_bar = GetRadiallyAveragedDensity(mesh);
  const bool has_profile = !rho_bar.empty();

#if NON_BAROTROPIC_EOS
  const std::vector<Real> &pressure_bar = GetRadiallyAveragedPressure(mesh);
  const std::vector<Real> &entropy_bar = GetRadiallyAveragedEntropy(mesh);
  const std::vector<Real> &temperature_bar = GetRadiallyAveragedTemperature(mesh);
#endif
  const std::vector<Real> &v1_bar = GetRadiallyAveragedV1(mesh);
  const std::vector<Real> &v2_bar = GetRadiallyAveragedV2(mesh);
  const std::vector<Real> &v3_bar = GetRadiallyAveragedV3(mesh);

#if NON_BAROTROPIC_EOS
  const Real gm1 = g_gm1;
  const Real gamma = g_gamma;
  const Real million_yr_code = units->million_yr_code;
  const Real lambda = g_powerlaw_lambda_code;
#endif
  const Real velocity_to_kms =
      (units->code_length_cgs / units->code_time_cgs) * 1.0e-5;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real rho = prim(IDN, k, j, i);
        const Real v1 = prim(IVX, k, j, i);
        const Real v2 = prim(IVY, k, j, i);
        const Real v3 = prim(IVZ, k, j, i);
        const Real radius_code = CellCenterRadiusCode(pcoord, k, j, i);
        Real rho_avg = 0.0;
        Real v1_avg = 0.0;
        Real v2_avg = 0.0;
        Real v3_avg = 0.0;
#if NON_BAROTROPIC_EOS
        Real pressure_avg = 0.0;
        Real entropy_avg = 0.0;
        Real temperature_avg = 0.0;
#endif
        bool have_bg = false;
        if (has_profile) {
          rho_avg = SampleRadialProfile(rho_bar, radius_code);
          v1_avg = SampleRadialProfile(v1_bar, radius_code);
          v2_avg = SampleRadialProfile(v2_bar, radius_code);
          v3_avg = SampleRadialProfile(v3_bar, radius_code);
          have_bg = true;
#if NON_BAROTROPIC_EOS
          pressure_avg = SampleRadialProfile(pressure_bar, radius_code);
          entropy_avg = SampleRadialProfile(entropy_bar, radius_code);
          temperature_avg = SampleRadialProfile(temperature_bar, radius_code);
#endif
        }

#if NON_BAROTROPIC_EOS
        const Real pressure = prim(IPR, k, j, i);
        Real entropy = 0.0;
        if (pressure > 0.0 && rho > 0.0) {
          entropy = pressure / std::pow(rho, gamma);
        }
        const Real temperature = ComputeCellTemperature(rho, pressure, *units);

        if (tcool_index >= 0) {
          Real tcool_myr = std::numeric_limits<Real>::infinity();
          if (rho > 0.0 && pressure > 0.0 && lambda > 0.0 && million_yr_code > 0.0) {
            const Real eint = pressure / gm1;
            const Real denom = lambda * rho * rho;
            if (denom > 0.0) {
              const Real tcool_code = eint / denom;
              tcool_myr = tcool_code / million_yr_code;
            }
          }
          user_out_var(tcool_index, k, j, i) = tcool_myr;
        }
#endif

        if (density_contrast_index >= 0) {
          Real contrast = 0.0;
          if (have_bg && rho_avg > 0.0) {
            contrast = (rho - rho_avg) / rho_avg;
          }
          user_out_var(density_contrast_index, k, j, i) = contrast;
        }

#if NON_BAROTROPIC_EOS
        if (temperature_index >= 0) {
          user_out_var(temperature_index, k, j, i) = temperature;
        }
        if (entropy_index >= 0) {
          user_out_var(entropy_index, k, j, i) = entropy;
        }
        if (delta_pressure_index >= 0) {
          Real delta = 0.0;
          if (have_bg && pressure_avg > 0.0) {
            delta = (pressure - pressure_avg) / pressure_avg;
          }
          user_out_var(delta_pressure_index, k, j, i) = delta;
        }
        if (delta_entropy_index >= 0) {
          Real delta = 0.0;
          if (have_bg && entropy_avg > 0.0) {
            delta = (entropy - entropy_avg) / entropy_avg;
          }
          user_out_var(delta_entropy_index, k, j, i) = delta;
        }
        if (delta_temperature_index >= 0) {
          Real delta = 0.0;
          if (have_bg && temperature_avg != 0.0) {
            delta = (temperature - temperature_avg) / temperature_avg;
          }
          user_out_var(delta_temperature_index, k, j, i) = delta;
        }
#endif

        if (grav_phi_index >= 0) {
          user_out_var(grav_phi_index, k, j, i) =
              PotentialInCodeUnits(radius_code, units);
        }
        if (pressure_hse_index >= 0) {
          user_out_var(pressure_hse_index, k, j, i) =
              SampleBackgroundPressureCode(radius_code, units);
        }

        const Real dv1 = have_bg ? (v1 - v1_avg) : 0.0;
        const Real dv2 = have_bg ? (v2 - v2_avg) : 0.0;
        const Real dv3 = have_bg ? (v3 - v3_avg) : 0.0;
        if (dv1_index >= 0) {
          user_out_var(dv1_index, k, j, i) = dv1 * velocity_to_kms;
        }
        if (dv2_index >= 0) {
          user_out_var(dv2_index, k, j, i) = dv2 * velocity_to_kms;
        }
        if (dv3_index >= 0) {
          user_out_var(dv3_index, k, j, i) = dv3 * velocity_to_kms;
        }

#if NON_BAROTROPIC_EOS
        if (mach_index >= 0) {
          Real mach = 0.0;
          if (rho > 0.0 && pressure > 0.0) {
            const Real sound_speed = std::sqrt(gamma * pressure / rho);
            if (sound_speed > 0.0) {
              const Real dv_mag = std::sqrt(SQR(dv1) + SQR(dv2) + SQR(dv3));
              mach = dv_mag / sound_speed;
            }
          }
          user_out_var(mach_index, k, j, i) = mach;
        }
#endif

#if MAGNETIC_FIELDS_ENABLED && NON_BAROTROPIC_EOS
        if (plasma_beta_index >= 0) {
          const Real b1 = pfield->bcc(IB1, k, j, i);
          const Real b2 = pfield->bcc(IB2, k, j, i);
          const Real b3 = pfield->bcc(IB3, k, j, i);
          const Real mag_pressure = 0.5 * (SQR(b1) + SQR(b2) + SQR(b3));
          Real beta = std::numeric_limits<Real>::infinity();
          if (mag_pressure > 0.0) {
            const Real pressure = prim(IPR, k, j, i);
            beta = pressure / mag_pressure;
          }
          user_out_var(plasma_beta_index, k, j, i) = beta;
        }
#endif
      }
    }
  }

#if MAGNETIC_FIELDS_ENABLED
  if (divb_index >= 0) {
    FaceField &bf = pfield->b;
    Coordinates *coord = pcoord;
    const int nx1 = ncells1;
    AthenaArray<Real> face1, face2p, face2m, face3p, face3m, volume;
    face1.NewAthenaArray(nx1 + 1);
    face2p.NewAthenaArray(nx1);
    face2m.NewAthenaArray(nx1);
    face3p.NewAthenaArray(nx1);
    face3m.NewAthenaArray(nx1);
    volume.NewAthenaArray(nx1);

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        coord->Face1Area(k, j, is, ie + 1, face1);
        coord->Face2Area(k, j + 1, is, ie, face2p);
        coord->Face2Area(k, j, is, ie, face2m);
        coord->Face3Area(k + 1, j, is, ie, face3p);
        coord->Face3Area(k, j, is, ie, face3m);
        coord->CellVolume(k, j, is, ie, volume);

        for (int i = is; i <= ie; ++i) {
          const Real flux_x1 =
              face1(i + 1) * bf.x1f(k, j, i + 1) - face1(i) * bf.x1f(k, j, i);
          const Real flux_x2 =
              face2p(i) * bf.x2f(k, j + 1, i) - face2m(i) * bf.x2f(k, j, i);
          const Real flux_x3 =
              face3p(i) * bf.x3f(k + 1, j, i) - face3m(i) * bf.x3f(k, j, i);
          const Real cell_vol = volume(i);
          Real divb = 0.0;
          if (cell_vol > 0.0) {
            divb = (flux_x1 + flux_x2 + flux_x3) / cell_vol;
          }
          user_out_var(divb_index, k, j, i) = divb;
        }
      }
    }
  }
#endif
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Check for non-finite magnetic fields each time step.
//========================================================================================
void Mesh::UserWorkInLoop() {
  if (g_profile && g_inner_sponge_tau > 0.0
      && !UsingCartesianCoordinates()
      && g_inner_sponge_radius > RadialProfileMinCode(mesh_size)) {
    for (int block = 0; block < nblocal; ++block) {
      ApplyInnerSponge(my_blocks(block));
    }
  }
  if (g_profile && g_outer_sponge_tau > 0.0
      && g_outer_sponge_inner_radius < NominalOuterRadiusCode(mesh_size)) {
    for (int block = 0; block < nblocal; ++block) {
      ApplyOuterSponge(my_blocks(block));
    }
  }
#if MAGNETIC_FIELDS_ENABLED
  auto require_finite = [](const char *label, int bid, int k, int j, int i, Real value) {
    if (!std::isfinite(value)) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp (UserWorkInLoop)" << std::endl
          << "Non-finite value in " << label << " on block " << bid
          << " at cell (" << k << "," << j << "," << i << ")";
      ATHENA_ERROR(msg);
    }
  };

  for (int block = 0; block < nblocal; ++block) {
    MeshBlock *pmb = my_blocks(block);
    auto &bf = pmb->pfield->b;
    auto &bcc = pmb->pfield->bcc;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie + 1; ++i) {
          require_finite("B^1 face", block, k, j, i, bf.x1f(k, j, i));
        }
      }
    }
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je + 1; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          require_finite("B^2 face", block, k, j, i, bf.x2f(k, j, i));
        }
      }
    }
    for (int k = pmb->ks; k <= pmb->ke + 1; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          require_finite("B^3 face", block, k, j, i, bf.x3f(k, j, i));
        }
      }
    }
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          require_finite("B^1 cell", block, k, j, i, bcc(IB1, k, j, i));
          require_finite("B^2 cell", block, k, j, i, bcc(IB2, k, j, i));
          require_finite("B^3 cell", block, k, j, i, bcc(IB3, k, j, i));
        }
      }
    }
  }
#endif
}

//========================================================================================
//! \fn void PrecipitatorGravity(...)
//! \brief Gravity source term using tabulated potential differences
//========================================================================================
void PrecipitatorGravity(MeshBlock *pmb, const Real time, const Real dt,
                         const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc,
                         AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar) {
  const bool gravity_enabled =
      (g_uniform_init == 0) && static_cast<bool>(g_profile);
  const bool cooling_enabled =
      g_enable_powerlaw_cooling && (g_powerlaw_lambda_code > 0.0);
  const bool heating_enabled =
      g_enable_magic_heating && (g_powerlaw_lambda_code > 0.0) &&
      (g_magic_profile_bins > 0) && (g_magic_c_v > 0.0) && (g_magic_Kp != 0.0);
  if (!gravity_enabled && !cooling_enabled && !heating_enabled) {
    return;
  }

  Units *units = nullptr;
  if (gravity_enabled || heating_enabled) {
    units = pmb->pmy_mesh->punit;
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object is not initialized.";
      ATHENA_ERROR(msg);
    }
  }
  Coordinates *pcoord = pmb->pcoord;
  const Real outer_radius_code = NominalOuterRadiusCode(pmb->pmy_mesh->mesh_size);
  if (heating_enabled) {
    UpdateMagicHeatingProfile(pmb->pmy_mesh, time, dt);
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real radius_code = CellCenterRadiusCode(pcoord, k, j, i);
        const Real rho = cons(IDN, k, j, i);
        const Real mom1 = cons(IM1, k, j, i);
        const Real mom2 = cons(IM2, k, j, i);
        const Real mom3 = cons(IM3, k, j, i);
        const Real Etot = cons(IEN, k, j, i);
        const Real inv_rho = 1.0 / std::max(rho, TINY_NUMBER);
        const Real KE = 0.5 * (SQR(mom1) + SQR(mom2) + SQR(mom3)) * inv_rho;
        const Real Eint_total = Etot - KE;
        Real thermal_eint = Eint_total;
#if MAGNETIC_FIELDS_ENABLED
        const Real b1 = bcc(IB1, k, j, i);
        const Real b2 = bcc(IB2, k, j, i);
        const Real b3 = bcc(IB3, k, j, i);
        thermal_eint -= 0.5 * (SQR(b1) + SQR(b2) + SQR(b3));
#endif
        const Real pressure = thermal_eint * g_gm1;

        if (gravity_enabled) {
          if (UsingCartesianCoordinates()) {
            const SphericalPoint sph = CellCenterSpherical(pcoord, k, j, i);
            if (rho > 0.0 && sph.r > 0.0) {
              const Real gmag = GravityInCodeUnits(sph.r, units);
              const Real gx = -gmag * sph.rhat_x;
              const Real gy = -gmag * sph.rhat_y;
              const Real gz = -gmag * sph.rhat_z;
              cons(IM1, k, j, i) += dt * rho * gx;
              cons(IM2, k, j, i) += dt * rho * gy;
              cons(IM3, k, j, i) += dt * rho * gz;
              cons(IEN, k, j, i) += dt * (mom1 * gx + mom2 * gy + mom3 * gz);
            }
          } else if (pressure > 0.0) {
            const Real dx1 = pcoord->dx1v(i);
            const Real phi_center = PotentialInCodeUnits(pcoord->x1v(i), units);
            const Real phi_minus = PotentialInCodeUnits(pcoord->x1f(i), units);
            const Real phi_plus = PotentialInCodeUnits(pcoord->x1f(i + 1), units);

            const Real kT_over_mu = pressure * inv_rho;
            if (kT_over_mu > 0.0) {
              const Real p_hse_plus =
                  pressure * std::exp(-(phi_plus - phi_center) / kT_over_mu);
              const Real p_hse_minus =
                  pressure * std::exp(-(phi_minus - phi_center) / kT_over_mu);

              cons(IM1, k, j, i) += dt * (p_hse_plus - p_hse_minus) / dx1;

              const Real vr = mom1 * inv_rho;
              cons(IEN, k, j, i) -= dt * rho * vr * (phi_plus - phi_minus) / dx1;
            }
          }
        }

        if (cooling_enabled) {
          const Real cooling_taper =
              OuterBufferHeatCoolTaper(radius_code, outer_radius_code) *
              MagicHeatingTaper(radius_code);
          if (cooling_taper > 0.0) {
            // Use the same altitude taper as magic heating.
            const Real thermal_energy = std::max(thermal_eint, 0.0);
            if (thermal_energy > 0.0) {
              const Real cooling_strength =
                  cooling_taper * g_powerlaw_lambda_code * rho * rho;
              if (cooling_strength > 0.0) {
                // Exact integration of de/dt = -rho^2 Lambda with constant Lambda
                const Real eint_new =
                    std::max(thermal_energy - dt * cooling_strength, 0.0);
                const Real dE = thermal_energy - eint_new;
                cons(IEN, k, j, i) -= dE;
              }
            }
          }
        }

        if (heating_enabled && g_magic_profile_ready) {
          const Real err = SampleMagicHeatingError(radius_code);
          if (err != 0.0) {
            const Real taper =
                OuterBufferHeatCoolTaper(radius_code, outer_radius_code) *
                MagicHeatingTaper(radius_code);
            if (taper > 0.0) {
              const Real rho_bg = SampleBackgroundDensityCode(radius_code, units);
              const Real pressure_bg = SampleBackgroundPressureCode(radius_code, units);
              Real inv_t_cool = 0.0;
              if (rho_bg > 0.0 && pressure_bg > 0.0 && g_powerlaw_lambda_code > 0.0) {
                const Real thermal_bg = pressure_bg / g_gm1;
                const Real cooling_strength = g_powerlaw_lambda_code * rho_bg * rho_bg;
                if (thermal_bg > 0.0 && cooling_strength > 0.0) {
                  const Real t_cool = thermal_bg / cooling_strength;
                  if (t_cool > 0.0) {
                    inv_t_cool = 1.0 / t_cool;
                  }
                }
              }
              if (inv_t_cool > 0.0) {
                const Real dE_dt =
                    -taper * rho * g_magic_c_v * inv_t_cool * (g_magic_Kp * err);
                cons(IEN, k, j, i) += dt * dE_dt;
              }
            }
          }
        }
      }
    }
  }
  return;
}
