# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the Athena++ solvers, organized by domain (`mesh/`, `hydro/`, `outputs/`, etc.); place new modules beside peers so `configure.py` can wire them into the generated `Makefile`.
- `tst/regression/` hosts the Python harness (`run_tests.py`) plus suites in `scripts/tests/`; land new physics coverage here alongside reusable data under `tst/regression/data/`.
- `tst/style/` bundles the project copy of `cpplint.py` and helper scripts; run these from the repository root to match CI checks.
- `inputs/` keeps reference problem files consumed by both documentation and regression runs; share reproducers by adding concise `.par` files here.

## Build, Test, and Development Commands
- Run Python scripts with `uv run` instead of invoking `python3` directly.
- `uv run configure.py --prob=orszag_tang --coord=cartesian -b -mpi` prepares a tailored `Makefile`; extend with paths such as `--hdf5_path=$HDF5_ROOT` when enabling optional libraries.
- `make -j$(sysctl -n hw.ncpu)` (or similar) builds `bin/athena`; call `make clean` before switching compile-time options to avoid stale objects.
- `./bin/athena -i inputs/orszag_tang.par` launches a run; store generated output in subdirectory for reproducibility.
- `uv run tst/regression/run_tests.py mhd/orszag_tang` exercises a focused regression, while omitting the argument drives the full matrix; pass `--config="-b -mpi"` to mirror CI flags.

## Coding Style & Naming Conventions
- C++ sources use two-space indentation, 90-character lines, Google include ordering, and `CamelCase` APIs; run `tst/style/check_athena_cpp_style.sh` (which wraps `cpplint`) before review.
- Maintain `snake_case` locals, all-caps IDs (`IDN`, `IEN`), and standard-library qualification (`std::sqrt`, `std::abs`); tabs and trailing whitespace are rejected by the style script.
- Python utilities must satisfy `flake8` per `setup.cfg`; execute `uv run flake8` from the project root before committing helpers or test drivers.

## Testing Guidelines
- Locate suites under `tst/regression/scripts/tests/<domain>` and name new files after the scenario they validate; keep helper logic inside `tst/regression/scripts` so multiple tests can share it.
- Run `uv run tst/regression/run_tests.py` after major changes and attach the summary table to your PR; include targeted invocations when only a subset applies.
- Use `uv run tst/regression/run_tests.py -cov lcov.info` when modifying core infrastructure so coverage shifts are visible to CI reviewers.

## Commit & Pull Request Guidelines
- Write imperative, descriptive commit subjects (e.g., `Fix bit normalization for 12th output`) and cite issues with `Fixes #123` when applicable.
- Gate every PR with the configure, build, regression, and lint commands you used; note them in the PR body together with any required environment variables.
- Keep changes scoped to a single concern, and list follow-up items or validation gaps explicitly so maintainers can plan next steps.
