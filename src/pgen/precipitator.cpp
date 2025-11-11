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
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"
#include "../parameter_input.hpp"
#include "../units/units.hpp"

namespace {

// Simple monotone interpolant over tabulated profile data
class PrecipitatorProfile {
 public:
  explicit PrecipitatorProfile(const std::string &filename) { LoadProfile(filename); }

  Real Density(Real r) const { return Interp(rho_table_, r); }
  Real Pressure(Real r) const { return Interp(pressure_table_, r); }
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
Real g_magic_profile_x3min = 0.0;
Real g_magic_profile_inv_dz = 0.0;
Real g_magic_profile_time = std::numeric_limits<Real>::quiet_NaN();
Real g_magic_profile_dt = std::numeric_limits<Real>::quiet_NaN();
bool g_magic_profile_ready = false;
std::vector<Real> g_magic_error_profile;

std::string g_force_free_param_file;
bool g_force_free_loaded = false;
Real g_force_free_alpha = 0.0;
Real g_force_free_amplitude = 1.0;

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

constexpr Real kForceFreeSeriesLimit = 1.0e-6;
constexpr Real kSqrtTwo = 1.41421356237309504880;
constexpr Real kPerturbationWarningThreshold = 1.0e6;

constexpr std::array<Real, 7> kGaussNodes = {
    {-0.94910791234275852453, -0.74153118559939443986, -0.40584515137739716691,
     0.0, 0.40584515137739716691, 0.74153118559939443986, 0.94910791234275852453}};

constexpr std::array<Real, 7> kGaussWeights = {
    {0.12948496616886969327, 0.27970539148927666790, 0.38183005050511894495,
     0.41795918367346938776, 0.38183005050511894495, 0.27970539148927666790,
     0.12948496616886969327}};

template <typename Func>
Real AverageProfile(const Func &func, Real a, Real b, const PrecipitatorProfile &profile) {
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

Real PotentialInCodeUnits(Real coord_value, const Units *units) {
  const Real length_cgs = units->code_length_cgs;
  const Real time_cgs = units->code_time_cgs;
  const Real code_potential_cgs = (length_cgs / (time_cgs * time_cgs)) * length_cgs;
  const Real r_cgs = coord_value * length_cgs;
  return g_profile->Phi(r_cgs) / code_potential_cgs;
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

  const Real x3min = mesh->mesh_size.x3min;
  const Real x3max = mesh->mesh_size.x3max;
  const Real extent = x3max - x3min;
  const bool has_extent = (num_bins > 1) && (extent > 0.0);
  const Real inv_dz = has_extent ? static_cast<Real>(num_bins) / extent : 0.0;

  for (int block = 0; block < mesh->nblocal; ++block) {
    MeshBlock *pmb = mesh->my_blocks(block);
    if (pmb == nullptr || pmb->phydro == nullptr) {
      continue;
    }
    auto &prim = pmb->phydro->w;
    Coordinates *coord = pmb->pcoord;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      const Real z = coord->x3v(k);
      int idx = has_extent ? static_cast<int>((z - x3min) * inv_dz) : 0;
      if (idx < 0) {
        idx = 0;
      }
      if (idx >= num_bins) {
        idx = num_bins - 1;
      }
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          const Real rho = prim(IDN, k, j, i);
          const Real pressure = prim(IPR, k, j, i);
          const Real temperature = ComputeCellTemperature(rho, pressure, *units);
          const Real err = temperature - g_magic_target_temperature;
          const Real cell_volume = coord->GetCellVolume(k, j, i);
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
  g_magic_profile_x3min = x3min;
  g_magic_profile_inv_dz = has_extent ? inv_dz : 0.0;
}

Real SampleMagicHeatingError(Real coord_value) {
  if (!g_magic_profile_ready || g_magic_error_profile.empty()) {
    return 0.0;
  }
  if (g_magic_profile_bins <= 1 || g_magic_profile_inv_dz == 0.0) {
    return g_magic_error_profile.front();
  }
  Real idx_f = (coord_value - g_magic_profile_x3min) * g_magic_profile_inv_dz;
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
    } else if (key == "amplitude") {
      g_force_free_amplitude = value;
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

Real SphericalBesselJ0(Real x) {
  const Real ax = std::abs(x);
  if (ax < kForceFreeSeriesLimit) {
    const Real x2 = x * x;
    const Real x4 = x2 * x2;
    return 1.0 - x2 / 6.0 + x4 / 120.0;
  }
  return std::sin(x) / x;
}

Real SphericalBesselJ1(Real x) {
  const Real ax = std::abs(x);
  if (ax < kForceFreeSeriesLimit) {
    return SeriesJ1(x);
  }
  return std::sin(x) / (x * x) - std::cos(x) / x;
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

void InitializeForceFreeField(MeshBlock *pmb) {
#if MAGNETIC_FIELDS_ENABLED
  LoadForceFreeParameters();
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
        a1(k, j, i) =
            ForceFreeVectorPotentialR(pcoord->x1v(i), pcoord->x2f(j));
        a3(k, j, i) =
            ForceFreeVectorPotentialPhi(pcoord->x1f(i), pcoord->x2f(j));
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

  a1.DeleteAthenaArray();
  a3.DeleteAthenaArray();
#else
  (void)pmb;
#endif
}

int PerturbationCoefficientCount() {
  return (g_pert_lmax + 1) * (g_pert_lmax + 2) / 2;
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
  if (l == 0) {
    return SphericalBesselJ0(x);
  }
  if (l == 1) {
    return SphericalBesselJ1(x);
  }
  if (ax < g_pert_small_kr_threshold) {
    return SmallXSphericalBessel(l, x);
  }
  Real jm1 = SphericalBesselJ0(x);
  Real jcurr = SphericalBesselJ1(x);
  for (int ell = 1; ell < l; ++ell) {
    if (ax < g_pert_small_kr_threshold) {
      return SmallXSphericalBessel(l, x);
    }
    const Real jp1 = ((2.0 * ell + 1.0) / x) * jcurr - jm1;
    if (!std::isfinite(jp1) || std::abs(jp1) > 1.0e12) {
      std::cout << "### Warning in precipitator.cpp: spherical Bessel recurrence overflow "
                << "(l=" << l << ", ell=" << ell << ", x=" << x
                << ", jm1=" << jm1 << ", jcurr=" << jcurr << ", jp1=" << jp1
                << ")\n";
    }
    jm1 = jcurr;
    jcurr = jp1;
  }
  return jcurr;
}

Real AssociatedLegendre(int l, int m, Real x) {
  if (m < 0 || m > l) {
    return 0.0;
  }
  Real pmm = 1.0;
  if (m > 0) {
    Real arg = 1.0 - x * x;
    if (arg < 0.0) {
      arg = 0.0;
    }
    Real somx2 = std::sqrt(arg);
    Real fact = 1.0;
    for (int i = 1; i <= m; ++i) {
      pmm *= -fact * somx2;
      fact += 2.0;
    }
  }
  if (l == m) {
    return pmm;
  }
  Real pmmp1 = x * (2 * m + 1) * pmm;
  if (l == m + 1) {
    return pmmp1;
  }
  Real pll = 0.0;
  for (int i = m + 2; i <= l; ++i) {
    pll = ((2 * i - 1) * x * pmmp1 - (i + m - 1) * pmm) / (i - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

Real HarmonicNorm(int l, int m) {
  const Real ln_ratio = std::lgamma(l - m + 1.0) - std::lgamma(l + m + 1.0);
  const Real norm_sq = ((2.0 * l + 1.0) / (4.0 * PI)) * std::exp(ln_ratio);
  return std::sqrt(norm_sq);
}

Real EvalSphHarmNoise(Real r, Real theta, Real phi) {
  if (!g_pert_ready) {
    return 0.0;
  }
  const Real cos_t = std::cos(theta);
  const Real r_scaled = ScaleRadius(r);
  Real noise = 0.0;
  const int num_coeff = PerturbationCoefficientCount();
  std::size_t idx_base = 0;

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
      const Real norm0 = HarmonicNorm(l, 0);
      const Real plm0 = AssociatedLegendre(l, 0, cos_t);
      const Real coeff0 = g_pert_coeff_cos[idx++];
      noise += coeff0 * level_scale * norm0 * plm0 * radial_val;

      for (int m = 1; m <= l; ++m) {
        const Real norm = HarmonicNorm(l, m);
        const Real plm = AssociatedLegendre(l, m, cos_t);
        const Real base = level_scale * norm * plm * radial_val;
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
  g_pert_radius_min = mesh.mesh_size.x1min;
  g_pert_radius_max = mesh.mesh_size.x1max;
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
        const Real r = coord->x1v(i);
        Real theta = 0.0;
        if (pmb->block_size.nx2 > 1) {
          theta = coord->x2v(j);
        } else {
          theta = 0.5 * (coord->x2f(j) + coord->x2f(j + 1));
        }
        Real phi = 0.0;
        if (pmb->block_size.nx3 > 1) {
          phi = coord->x3v(k);
        } else {
          phi = 0.5 * (coord->x3f(k) + coord->x3f(k + 1));
        }

        const Real delta = g_pert_sigma * EvalSphHarmNoise(r, theta, phi);
        if (!std::isfinite(delta) ||
            std::abs(delta) > kPerturbationWarningThreshold) {
          std::cout << "### Warning in precipitator.cpp: large density perturbation "
                    << "delta=" << delta << " at (r=" << r
                    << ", theta=" << theta << ", phi=" << phi << ")\n";
        }

        const Real rho0 = prim(IDN, k, j, i);
        const Real rho_new = rho0 * (1.0 + delta);
        if (rho_new <= 0.0) {
          std::cout << "### Warning in precipitator.cpp: density perturbation produced "
                    << "rho <= 0 (rho0=" << rho0 << ", delta=" << delta
                    << ", r=" << r << ", theta=" << theta
                    << ", phi=" << phi << ")\n";
        }
        prim(IDN, k, j, i) = rho_new;
      }
    }
  }
}

} // namespace

// Forward declaration so we can enroll it before the definition appears.
void PrecipitatorGravity(MeshBlock *pmb, const Real time, const Real dt,
                         const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Initialize precipitator-specific mesh data and source term
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  g_gamma = pin->GetReal("hydro", "gamma");
  g_gm1 = g_gamma - 1.0;
  Units *units = punit;

  const std::string profile_filename = pin->GetString("precipitator", "hse_profile_filename");
  g_profile = std::unique_ptr<PrecipitatorProfile>(
      new PrecipitatorProfile(profile_filename));

  g_uniform_init = pin->GetOrAddInteger("precipitator", "uniform_init", 0);
  if (g_uniform_init == 1) {
    g_uniform_height = pin->GetReal("precipitator", "uniform_init_height");
  }

  g_force_free_param_file =
      pin->GetOrAddString("precipitator", "force_free_param_file",
                          "inputs/aphi.txt");
#if defined(COORDINATE_SYSTEM)
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in precipitator.cpp" << std::endl
        << "Force-free magnetic configuration requires spherical_polar coordinates.";
    ATHENA_ERROR(msg);
  }
#endif

  const Real He_mass_fraction = pin->GetOrAddReal("hydro", "He_mass_fraction", 0.25);
  const Real hydrogen_mass_fraction = 1.0 - He_mass_fraction;

  g_enable_powerlaw_cooling =
      (pin->GetOrAddInteger("precipitator", "enable_powerlaw_cooling", 0) != 0);
  const std::string heating_mode =
      pin->GetOrAddString("precipitator", "enable_heating", "none");
  g_enable_magic_heating = (heating_mode == "magic");

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
    g_magic_mu = 1.0 / (He_mass_fraction * 0.75 + hydrogen_mass_fraction * 2.0);
    g_magic_mmw_cgs = g_magic_mu * Constants::hydrogen_mass_cgs;
    g_magic_mmw_code = g_magic_mmw_cgs * units->gram_code;
    g_magic_c_v = (units->k_boltzmann_code / g_magic_mmw_code) / g_gm1;
    g_magic_profile_bins = mesh_size.nx3;
    g_magic_error_profile.assign(static_cast<std::size_t>(g_magic_profile_bins), 0.0);
    g_magic_profile_ready = false;
    g_magic_profile_time = std::numeric_limits<Real>::quiet_NaN();
    g_magic_profile_dt = std::numeric_limits<Real>::quiet_NaN();
    g_magic_profile_x3min = mesh_size.x3min;
    g_magic_profile_inv_dz = 0.0;
  } else {
    g_magic_error_profile.clear();
    g_magic_profile_bins = 0;
    g_magic_profile_ready = false;
    g_magic_profile_time = std::numeric_limits<Real>::quiet_NaN();
    g_magic_profile_dt = std::numeric_limits<Real>::quiet_NaN();
  }

  g_pert_radius_min = mesh_size.x1min;
  g_pert_radius_max = mesh_size.x1max;
  const int pert_flag =
      pin->GetOrAddInteger("precipitator", "enable_fourier_bessel_perturbations", 0);
  g_pert_sigma = pin->GetOrAddReal("precipitator", "perturbation_sigma", 0.01);
  g_pert_lmax = pin->GetOrAddInteger("precipitator", "perturbation_lmax", 12);
  g_pert_radial_modes =
      pin->GetOrAddInteger("precipitator", "perturbation_radial_modes", 16);
  g_pert_kmin = pin->GetOrAddReal("precipitator", "perturbation_kmin", 1.0);
  g_pert_kmax = pin->GetOrAddReal("precipitator", "perturbation_kmax", 16.0);
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

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        const int idx = i - il;
        phydro->w(IDN, k, j, i) = rho_profile[idx];
#if NON_BAROTROPIC_EOS
        phydro->w(IPR, k, j, i) = pressure_profile[idx];
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
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js, je, ks, ke);

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
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je, ks, ke);
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
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Fill auxiliary cooling-time output in Myr
//========================================================================================
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  if (nuser_out_var == 0) {
    return;
  }

  int next_index = 0;
  const int tcool_index = g_enable_powerlaw_cooling ? next_index++ : -1;
#if MAGNETIC_FIELDS_ENABLED
  const int divb_index = next_index++;
#else
  constexpr int divb_index = -1;
#endif

  if (tcool_index >= 0) {
    Units *units = pmy_mesh->punit;
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object is not initialized.";
      ATHENA_ERROR(msg);
    }

    const Real gm1 = g_gm1;
    const Real million_yr_code = units->million_yr_code;
    const Real lambda = g_powerlaw_lambda_code;
    auto &prim = phydro->w;

    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          const Real rho = prim(IDN, k, j, i);
          const Real pressure = prim(IPR, k, j, i);
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
                         const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar) {
  const bool gravity_enabled = (g_uniform_init == 0) && static_cast<bool>(g_profile);
  const bool cooling_enabled = g_enable_powerlaw_cooling && (g_powerlaw_lambda_code > 0.0);
  const bool heating_enabled = g_enable_magic_heating && (g_powerlaw_lambda_code > 0.0) &&
                               (g_magic_profile_bins > 0) && (g_magic_c_v > 0.0) &&
                               (g_magic_Kp != 0.0);
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
  if (heating_enabled) {
    UpdateMagicHeatingProfile(pmb->pmy_mesh, time, dt);
  }

  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rho = cons(IDN, k, j, i);
        const Real mom1 = cons(IM1, k, j, i);
        const Real mom2 = cons(IM2, k, j, i);
        const Real mom3 = cons(IM3, k, j, i);
        const Real Etot = cons(IEN, k, j, i);
        const Real inv_rho = 1.0 / std::max(rho, TINY_NUMBER);
        const Real KE = 0.5 * (SQR(mom1) + SQR(mom2) + SQR(mom3)) * inv_rho;
        const Real Eint_total = Etot - KE;
        const Real pressure = Eint_total * g_gm1;

        if (gravity_enabled && pressure > 0.0) {
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

        if (cooling_enabled) {
          const Real radius_code = pcoord->x1v(i);
          const Real cooling_taper = MagicHeatingTaper(radius_code);
          if (cooling_taper > 0.0) {
            // Use the same altitude taper as magic heating.
            Real available_eint = Eint_total;
#if MAGNETIC_FIELDS_ENABLED
            const Real b1 = bcc(IB1, k, j, i);
            const Real b2 = bcc(IB2, k, j, i);
            const Real b3 = bcc(IB3, k, j, i);
            available_eint -= 0.5 * (SQR(b1) + SQR(b2) + SQR(b3));
#endif
            const Real thermal_energy = std::max(available_eint, 0.0);
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
          const Real z = pcoord->x3v(k);
          const Real err = SampleMagicHeatingError(z);
          if (err != 0.0) {
            const Real radius_code = pcoord->x1v(i);
            const Real taper = MagicHeatingTaper(radius_code);
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
