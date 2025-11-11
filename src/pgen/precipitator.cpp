//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator.cpp
//! \brief Idealized galaxy precipitator problem generator (hydrostatic + gravity only)
//========================================================================================

// C headers

// C++ headers
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
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

std::string g_force_free_param_file;
bool g_force_free_loaded = false;
Real g_force_free_alpha = 0.0;
Real g_force_free_amplitude = 1.0;

constexpr Real kForceFreeSeriesLimit = 1.0e-6;

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

  g_enable_powerlaw_cooling =
      (pin->GetOrAddInteger("precipitator", "enable_powerlaw_cooling", 0) != 0);
  g_powerlaw_lambda_code = 0.0;
  if (g_enable_powerlaw_cooling) {
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object must be configured before enabling power-law cooling.";
      ATHENA_ERROR(msg);
    }
    // Default Lambda matches the AthenaPK precipitator cooling table (1e-22 erg cm^3/s).
    const Real lambda_cgs =
        pin->GetOrAddReal("precipitator", "powerlaw_lambda_cgs", 1.0e-22);
    if (lambda_cgs <= 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "powerlaw_lambda_cgs must be positive when enable_powerlaw_cooling is set.";
      ATHENA_ERROR(msg);
    }
    const Real He_mass_fraction = pin->GetOrAddReal("hydro", "He_mass_fraction", 0.25);
    const Real hydrogen_mass_fraction = 1.0 - He_mass_fraction;
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
  if (g_enable_powerlaw_cooling) {
    AllocateUserOutputVariables(1);
    SetUserOutputVariableName(0, "tcool_myr");
  }
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Fill auxiliary cooling-time output in Myr
//========================================================================================
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  if (!g_enable_powerlaw_cooling || nuser_out_var == 0) {
    return;
  }

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
        user_out_var(0, k, j, i) = tcool_myr;
      }
    }
  }
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
  if (!gravity_enabled && !cooling_enabled) {
    return;
  }

  Units *units = nullptr;
  if (gravity_enabled) {
    units = pmb->pmy_mesh->punit;
    if (units == nullptr) {
      std::stringstream msg;
      msg << "### FATAL ERROR in precipitator.cpp" << std::endl
          << "Units object is not initialized.";
      ATHENA_ERROR(msg);
    }
  }
  Coordinates *pcoord = pmb->pcoord;

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
          Real available_eint = Eint_total;
#if MAGNETIC_FIELDS_ENABLED
          const Real b1 = bcc(IB1, k, j, i);
          const Real b2 = bcc(IB2, k, j, i);
          const Real b3 = bcc(IB3, k, j, i);
          available_eint -= 0.5 * (SQR(b1) + SQR(b2) + SQR(b3));
#endif
          const Real thermal_energy = std::max(available_eint, 0.0);
          if (thermal_energy > 0.0) {
            const Real cooling_strength = g_powerlaw_lambda_code * rho * rho;
            if (cooling_strength > 0.0) {
              // Exact integration of de/dt = -rho^2 Lambda with constant Lambda
              const Real eint_new = std::max(thermal_energy - dt * cooling_strength, 0.0);
              const Real dE = thermal_energy - eint_new;
              cons(IEN, k, j, i) -= dE;
            }
          }
        }
      }
    }
  }
  return;
}
