#!/bin/bash
set -euxo pipefail

OPENPMD_SRC="extern/openPMD-api"
OPENPMD_BUILD="${OPENPMD_SRC}/build"
OPENPMD_PREFIX="${OPENPMD_BUILD}/install"

cmake -S "${OPENPMD_SRC}" -B "${OPENPMD_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${OPENPMD_PREFIX}" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DopenPMD_BUILD_CLI_TOOLS=OFF \
  -DopenPMD_USE_MPI=ON \
  -DopenPMD_USE_PYTHON=OFF
cmake --build "${OPENPMD_BUILD}" --target openPMD -- -j8
cmake --install "${OPENPMD_BUILD}"

python3 configure.py -openpmd --openpmd_path "${OPENPMD_PREFIX}" --prob precipitator --flux hlld --coord spherical_polar -mpi -b
make -j8
