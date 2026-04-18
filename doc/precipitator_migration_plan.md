# Precipitator Spherical-to-Cartesian Migration Plan

## Scope

This note captures the current plan for migrating the Athena++ `precipitator`
problem generator from `spherical_polar` to `cartesian` coordinates.

The goal is a Cartesian precipitator that preserves the existing spherical
background atmosphere and force-free magnetic topology as closely as practical,
without introducing a special inner boundary or well-balanced reconstruction.

## Agreed Assumptions

- Do not implement well-balanced reconstruction in Cartesian coordinates.
- Do not excise an inner spherical boundary. The Cartesian domain should be a
  full box centered on the halo.
- Do not damp the flow near the center in Cartesian coordinates.
- Keep the existing radius-based source-term tapering behavior.
- Treat preservation of the force-free magnetic topology with small `divB` as
  the main technical risk.

## Current Gaps

The present `src/pgen/precipitator.cpp` already has a few Cartesian-aware
pieces, but several core paths still assume `x1 = r`:

- Gravity updates only the `IM1` direction using `x1` face potentials.
- Force-free field initialization is restricted to `spherical_polar`.
- Density perturbations still read `theta` and `phi` directly from `x2/x3`.
- Radial diagnostics and magic-heating bins still index by global `x1`.
- Inner and outer sponge logic still uses `x1` as the radius coordinate.

## Implementation Plan

### 1. Add a Geometry Helper Layer

Add small helper routines in `src/pgen/precipitator.cpp` for:

- Cell-center Cartesian position `(x, y, z)`.
- Face-center Cartesian positions for each face family.
- Spherical radius `r = sqrt(x^2 + y^2 + z^2)`.
- Spherical angles `(theta, phi)` derived from Cartesian position.
- Radial unit vector `rhat`, including a safe `r -> 0` fallback.

After these helpers exist, route all coordinate-sensitive logic through them
instead of using `x1v()`, `x1f()`, `x2v()`, and `x3v()` as spherical variables.

### 2. Port Hydro Initialization Without Well-Balancing

Use the tabulated HSE profile as a function of spherical radius only:

- Sample `rho` and `P` at the cell-center radius everywhere in the Cartesian
  box.
- Keep the existing outer buffer halo logic, but evaluate its thresholds using
  true radius rather than `x1`.
- Accept the initial transients caused by not doing well-balanced
  reconstruction.

### 3. Rewrite Gravity as a Cartesian Radial Force

Replace the current `x1`-only gravity source with a true Cartesian radial
update:

- Compute `g(r)` or `dPhi/dr` from the tabulated profile.
- Apply momentum source terms to `IM1`, `IM2`, and `IM3` along `rhat`.
- Apply the matching energy source term using `rho v . g`.

This removes the last major assumption that the radial coordinate is aligned
with the `x1` axis.

### 4. Remove Cartesian Central Damping

Do not apply the inner sponge in Cartesian runs:

- Keep the existing inner sponge behavior only for spherical-polar mode, or
  make it a Cartesian no-op.
- Leave the outer sponge alone, but evaluate its trigger radius using true
  spherical radius.

This is distinct from source-term tapering, which should remain radius-based.

### 5. Keep Source-Term Tapers Radius-Based

Do not redesign the heating and cooling tapers:

- Continue to use the existing radius-based taper logic for the outer buffer.
- Continue to use the existing smoothing/tapering behavior for the thermostat.
- In Cartesian runs, feed those routines true spherical radius instead of `x1`.

### 6. Rework Diagnostics and Heating Profiles Around True Radius

Anything described as "radial" should be binned by spherical radius, not by
global `x1` index:

- Shell-averaged density, pressure, entropy, temperature, and velocity
  diagnostics.
- `grav_phi` and `pressure_hse`.
- Density, pressure, entropy, and temperature contrast outputs.
- Magic-heating error profiles and any other radial averages.

This likely requires replacing the current `mesh_size.nx1` and `lx1`
index-based reductions with explicit radius bins.

### 7. Keep the Existing Perturbation Model, but Feed It Cartesian-Derived Angles

The current Fourier-Bessel / spherical-harmonic perturbation model can stay if
it is evaluated consistently:

- Compute `(r, theta, phi)` from Cartesian cell centers.
- Evaluate the existing perturbation basis with those derived coordinates.
- Keep the perturbations disabled while validating the hydro and MHD base state.

### 8. Rewrite the Force-Free MHD Initialization Around a Cartesian Vector Potential

This is the critical path.

The most conservative approach is:

- Start from the existing analytic force-free vector potential used in the
  spherical implementation.
- Evaluate that vector potential at Cartesian edge or vertex locations, not in
  spherical mesh coordinates.
- Transform the vector potential from spherical components to Cartesian
  components.
- Numerically differentiate the Cartesian vector potential to obtain a discrete
  curl on the Cartesian mesh.
- Fill face-centered magnetic fields from that curl so the result remains
  compatible with constrained transport.
- Recompute `bcc` from the face fields and validate `divB`.

Important details:

- The implementation needs a robust `r -> 0` treatment so the origin does not
  introduce non-finite values.
- The discrete curl should be constructed in the same face-centered geometry
  used by Athena++ CT updates, rather than by first building only cell-centered
  fields.
- Validation should focus on both topology preservation and `divB` staying near
  machine error.

### 9. Add a Cartesian Input Deck and Validate in Stages

Add a new input file, likely `inputs/mhd/athinput.precipitator_cartesian`, and
bring features online in this order:

1. Hydro + gravity + cooling/heating only, with `force_free_bfield_gauss = 0`
   and perturbations off.
2. Force-free magnetic initialization, with `divB` and field morphology checks.
3. Density perturbations.

Expected validation steps:

- Startup completes with finite hydro and magnetic variables.
- `divB` remains small after initialization and the first few updates.
- Radial diagnostics recovered from the Cartesian run look sensible.
- The outer buffer and source-term tapers still act at the intended radii.

## Priority Order

1. Geometry helper layer.
2. Cartesian hydro initialization.
3. Cartesian gravity source term.
4. Radius-based diagnostics and heating bins.
5. Cartesian force-free magnetic initialization.
6. Perturbations.
7. New input deck and regression smoke test.

## Primary Risk

The main risk is preserving the existing force-free magnetic topology while
keeping `divB` acceptably small on a Cartesian mesh. The implementation should
optimize for a CT-compatible face-field initialization, even if that requires a
substantial rewrite of the current magnetic setup.
