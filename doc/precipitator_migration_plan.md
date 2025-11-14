# Precipitator Migration Plan

## Context & Goal
- AthenaPK currently hosts the full precipitator scenario, spanning the main problem driver, source-term hooks, and supporting utilities (`../athenapk/src/pgen/precipitator.cpp:112-828`, `../athenapk/src/pgen/precipitator_srcterms.cpp:45-320`).
- Athena++ lacks any equivalent problem or infrastructure for the custom diagnostics, turbulence forcing, and power-law cooling plumbing seen in AthenaPK.
- Objective: reproduce the precipitator problem inside Athena++ with feature parity (physics, restartability, diagnostics, and inputs/tests) while respecting Athena++ abstractions.

## High-Level Workstreams
1. **Problem Registration & Build Glue**
   - Add a new `src/pgen/precipitator.cpp` wired into `configure.py`/`Makefile` selection similar to other pgens (e.g., `src/pgen/ssheet.cpp:35-210`).
   - Update the Athena++ launcher so `problem=precipitator` enforces the correct hooks (e.g., source functions and custom mesh generators) the way AthenaPK registers them in `../athenapk/src/main.cpp:186-195`.
   - Ensure data products (`grav_phi`, `pressure_hse`, diagnostics) get declared via Athena++ mechanisms (likely `AllocateRealUserMeshBlockDataField` and custom `AthenaArray`s).

2. **Initialization Logic & State Management**
   - Port the package setup currently in `ProblemInitPackageData` (`../athenapk/src/pgen/precipitator.cpp:146-394`), including:
     - Gravitational potential arrays, HSE reference profiles, derived diagnostics (`entropy`, `turbulent_heating`, `dv`, etc.).
     - Thermostat controller state (target temperature, proportional gain, smoothing parameters).
     - Random density modes (`drho_hat`) and velocity perturbation driver configuration (`sigma_v`, `FewModesFT` phases).
   - Translate the AthenaPK `ProblemGenerator` body (`../athenapk/src/pgen/precipitator.cpp:403-828`) into Athena++ loops over `phydro->w`/`phydro->u`, replicating:
     - Hydrostatic profile sampling via `PrecipitatorProfile`.
     - Inverse FFT density perturbations.
     - Gravity potential fill and background magnetic field initialization.
     - Unit conversions (`Units` helper, code-to-cgs conversions).

3. **Source Terms & Turbulence Driving**
   - Recreate the unsplit/split source handling from `../athenapk/src/pgen/precipitator_srcterms.cpp:45-320`:
     - Gravity update using background potential differences.
     - Ornstein-Uhlenbeck turbulence forcing tied to the `FewModesFT` helper.
     - “Magic heating” proportional controller layered atop the simplified power-law cooling term.
   - Map these onto Athena++’s `Mesh::EnrollUserExplicitSourceFunction` (`src/mesh/mesh.cpp:1468-1482`) and ensure they execute at the appropriate integrator stage.
   - If Athena++ lacks reusable turbulence-driver classes beyond the FFT driver (`src/fft/turbulence.hpp:8-63`), decide whether to adapt that module or continue with the AthenaPK few-modes implementation.

4. **Diagnostics & Output Hooks**
   - Port the radial-profile reductions (`ComputeAvgProfile1D`, `VerticalMeanProfiles`) and derived field fills done before output (`../athenapk/src/pgen/precipitator.cpp:650-828`).
   - Utilize `MeshBlock::UserWorkBeforeOutput` and possibly `Mesh::AllocateUserHistoryOutput` (see `src/mesh/mesh.cpp:1508-1523`) to store derived quantities and history channels.
   - Gravitation potential, hydrostatic pressure, entropy/temperature deltas, Mach, and velocity-dispersion diagnostics now populate via the new radial-profile helpers; only the turbulent-heating rates remain outstanding.
   - Only the turbulent-heating history needs persistent storage between source applications and diagnostics; all other derived fields recompute their radial profiles on demand within `UserWorkBeforeOutput`.

5. **Cooling & Heating Integration**
   - AthenaPK’s precipitator setup couples the “magic heating” controller to a power-law cooling approximation (Λ ∝ n²).
   - Athena++ should retain this power-law model; ensure coefficient conversion, tapers, and smoothing match AthenaPK behaviour.
   - Document the required input parameters (Λ value, smoothing height, thermostat set point/gains) so users can reproduce AthenaPK calibration.

6. **Inputs, Regression Tests, and Documentation**
   - Convert the AthenaPK sample inputs (`../athenapk/inputs/precipitator_lowres.in:5-105`, etc.) into Athena++ `inputs/*.par` files, updating block names and defaults to match Athena++ parser expectations.
   - Add focused regression tests under `tst/regression/scripts/tests/` and document run commands per repository guidelines.
   - Capture any special requirements (cooling parameters, restart metadata) in `doc/` to aid future maintenance.

## Helper Utility Inventory
| Helper / Feature | Location (AthenaPK) | Purpose | Athena++ Equivalent? | Notes / Action |
| --- | --- | --- | --- | --- |
| `PrecipitatorProfile` | `../athenapk/src/utils/precipitator_profile.hpp:8-78` | Loads tabulated background rho/P/phi/B profiles from disk. | **Partial:** linear interpolation via STL containers is sufficient in Athena++. | Implement a lightweight profile loader that linearly interpolates the table without needing Parthenon spline helpers. |
| `ComputeAvgProfile1D` & `VerticalMeanProfiles` | `../athenapk/src/utils/vertical_mean_profiles.hpp:8-134` | Reduce 3D data into radial averages using Parthenon MeshData packs. | **No direct analogue** in Athena++; only ad-hoc reductions in individual pgens. | Likely rewrite using Athena++ `MeshBlock` loops and MPI reductions (see `src/pgen/strat.cpp:320-404` for pattern). |
| `FewModesFT` turbulence helper | `../athenapk/src/utils/few_modes_ft.cpp:1-210` | Maintains explicit Fourier modes, Ornstein-Uhlenbeck updates, restart dumps. | Athena++ has `TurbulenceDriver` in `src/fft/turbulence.hpp:8-63`, but it assumes full FFT driver and different data structures. | Decide whether to port `FewModesFT` (requires Parthenon-like metadata emulation) or refactor to reuse `TurbulenceDriver`. Either way will need significant adaptation. |
| `Units` class | `../athenapk/src/units.hpp:1-118` | Handles code<->cgs conversions, stores constants, exposes to packages. | **Yes:** Athena++ supplies `src/units/units.hpp:1-78`, though API differs (direct members vs. getters). | Map usage to Athena++ `Units`. Might need shim functions or parameter caching. |
| Power-law cooling coefficient | `../athenapk/src/pgen/precipitator_srcterms.cpp:84-160` | Provides Λ n² cooling used by the thermostat. | **Partial:** Athena++ must implement coefficient conversion itself. | Confirm unit conversions and tapers reproduce AthenaPK’s energetics; no tabulated cooling port is required. |
| Random perturbation utilities (`Kokkos::Random_XorShift64_Pool`, Box-Muller loops) | `../athenapk/src/pgen/precipitator.cpp:292-357` | Generate Gaussian spectral amplitudes for density perturbations. | Athena++ lacks Kokkos device RNGs; existing pgens use host RNG (`std::mt19937`) or deterministic patterns. | Need new RNG approach (host-side fill + copy, or integrate standalone random module). |
| Restart state plumbing for turbulence (`FewModesFT::SaveStateBeforeOutput`) | `../athenapk/src/utils/few_modes_ft.cpp:214-309` | Persists Fourier coefficients + RNG state back into input file. | Athena++ restart format is HDF5 and lacks parameter mutation. | Must design an Athena++-compatible restart mechanism (e.g., store in `AthenaArray` and rely on standard restart dumps). |
| `MagicHeatingSrcTerm` proportional controller | `../athenapk/src/pgen/precipitator_srcterms.cpp:205-320` | Applies thermostat-driven heating tied to radial averages. | No built-in equivalent. | Recreate using Athena++ source-term hooks; depends on new diagnostic reductions and power-law cooling coefficient. |

## Risks & Open Questions
- **Parthenon-only constructs:** Many helpers depend on Parthenon’s package metadata and `MeshData` views; replicating these in Athena++ will require bespoke storage or partial Parthenon backports.
- **Restart compatibility:** AthenaPK stores turbulence state via input-parameter mutation, which Athena++ does not support. Need an alternative restart strategy.
- **Cooling coefficient accuracy:** The power-law Λ conversion depends on unit normalization; mismatches will skew cooling times and the thermostat response.
- **Performance considerations:** AthenaPK leverages Kokkos parallelism; Athena++ currently relies on its own OpenMP/MPI loops. Rewriting kernels may affect performance and will need validation.

## Immediate Next Steps
1. Decide whether to port Parthenon helpers wholesale or rewrite them using Athena++ primitives (affects turbulence driver and diagnostics design).
2. Prototype minimal `PrecipitatorProfile` + gravity setup inside Athena++ to validate file I/O and interpolation strategy.
3. Draft interface expectations for turbulence forcing and thermostat heating within Athena++’s source-term framework.
