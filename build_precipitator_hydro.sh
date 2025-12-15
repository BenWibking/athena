#!/bin/bash
set -euxo pipefail

OPENPMD_SRC="extern/openPMD-api"
OPENPMD_BUILD="${OPENPMD_SRC}/build"
OPENPMD_PREFIX="${OPENPMD_BUILD}/install"

detect_hdf5_prefix() {
  local candidate prefix
  for var in HDF5_ROOT HDF5_DIR HDF5_HOME; do
    candidate="${!var-}"
    if [[ -n "${candidate}" && ( -d "${candidate}/lib" || -d "${candidate}/lib64" ) ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  local IFS=':'
  for candidate in ${CMAKE_PREFIX_PATH-}; do
    [[ -z "${candidate}" ]] && continue
    case "${candidate}" in
      *[Hh][Dd][Ff]5*)
        if [[ -d "${candidate}/lib" || -d "${candidate}/lib64" ]]; then
          echo "${candidate}"
          return 0
        fi
        ;;
    esac
  done

  local combined_paths="${LD_LIBRARY_PATH-}:${LIBRARY_PATH-}"
  for candidate in ${combined_paths}; do
    if [[ -z "${candidate}" || ! -d "${candidate}" ]]; then
      continue
    fi
    case "${candidate}" in
      *[Hh][Dd][Ff]5*)
        prefix=$(dirname "${candidate%/}")
        if [[ -d "${prefix}/lib" || -d "${prefix}/lib64" ]]; then
          echo "${prefix}"
          return 0
        fi
        ;;
    esac
  done

  return 1
}

detect_mpi_flavor() {
  local output lowered
  if ! command -v mpicxx >/dev/null 2>&1; then
    echo "unknown"
    return 1
  fi
  output=$(mpicxx -show 2>/dev/null || true)
  lowered=$(printf '%s' "${output}" | tr '[:upper:]' '[:lower:]')
  if [[ "${lowered}" == *mpich* ]]; then
    echo "mpich"
  elif [[ "${lowered}" == *openmpi* || "${lowered}" == *ompi* ]]; then
    echo "openmpi"
  else
    echo "unknown"
  fi
}

detect_library_mpi_flavor() {
  local lib="$1" deps lowered
  if [[ ! -e "${lib}" ]]; then
    echo "unknown"
    return 1
  fi
  if command -v otool >/dev/null 2>&1; then
    deps=$(otool -L "${lib}" 2>/dev/null || true)
  elif command -v ldd >/dev/null 2>&1; then
    deps=$(ldd "${lib}" 2>/dev/null || true)
  else
    echo "unknown"
    return 1
  fi
  lowered=$(printf '%s' "${deps}" | tr '[:upper:]' '[:lower:]')
  if [[ "${lowered}" == *openmpi* || "${lowered}" == *ompi* ]]; then
    echo "openmpi"
  elif [[ "${lowered}" == *mpich* ]]; then
    echo "mpich"
  else
    echo "unknown"
  fi
}

find_hdf5_showconfig_tool() {
  local prefix="${1-}" candidate
  for candidate in \
    "${prefix:+${prefix}/bin/h5pcc}" \
    "${prefix:+${prefix}/bin/h5cc}" \
    "$(command -v h5pcc 2>/dev/null || true)" \
    "$(command -v h5cc 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

detect_hdf5_mpi_flavor() {
  local prefix="${1-}" tool config lowered
  tool=$(find_hdf5_showconfig_tool "${prefix}" || true)
  if [[ -z "${tool}" ]]; then
    echo "unknown"
    return 1
  fi

  config=$("${tool}" -showconfig 2>/dev/null || true)
  if [[ -z "${config}" ]]; then
    echo "unknown"
    return 1
  fi

  lowered=$(printf '%s' "${config}" | tr '[:upper:]' '[:lower:]')
  if ! grep -qi "parallel hdf5:[[:space:]]*on" <<<"${config}"; then
    echo "serial"
    return 0
  fi
  if [[ "${lowered}" == *mpich* ]]; then
    echo "mpich"
  elif [[ "${lowered}" == *openmpi* || "${lowered}" == *ompi* ]]; then
    echo "openmpi"
  else
    echo "mpi-unknown"
  fi
}

find_openpmd_lib() {
  local base="${OPENPMD_PREFIX}/lib"
  for candidate in \
    "${base}/libopenPMD.dylib" \
    "${base}/libopenPMD.so" \
    "${base}/libopenPMD.a"; do
    if [[ -e "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

HDF5_PREFIX="$(detect_hdf5_prefix || true)"
CONFIGURE_ARGS=(-openpmd --openpmd_path "${OPENPMD_PREFIX}" -hdf5 --prob precipitator
  --flux hllc --coord spherical_polar -mpi --nghost=3)
if [[ -n "${HDF5_PREFIX}" ]]; then
  echo "### Using HDF5 from ${HDF5_PREFIX}"
  CONFIGURE_ARGS+=(--hdf5_path "${HDF5_PREFIX}")
else
  echo "### HDF5 not detected in environment; relying on default search paths" >&2
fi

MPI_FLAVOR="$(detect_mpi_flavor || true)"
HDF5_MPI_FLAVOR="$(detect_hdf5_mpi_flavor "${HDF5_PREFIX}")"
if [[ "${HDF5_MPI_FLAVOR}" != "serial" && "${HDF5_MPI_FLAVOR}" != "unknown" && "${HDF5_MPI_FLAVOR}" != "mpi-unknown" ]]; then
  if [[ "${MPI_FLAVOR}" != "${HDF5_MPI_FLAVOR}" ]]; then
    echo "### ERROR: HDF5 was built with MPI (${HDF5_MPI_FLAVOR}) while mpicxx appears to use '${MPI_FLAVOR}'. Mixing MPI implementations will fail; rebuild HDF5 or switch to a matching MPI toolchain." >&2
    exit 1
  fi
fi

OPENPMD_LIB="$(find_openpmd_lib || true)"
if [[ -n "${OPENPMD_LIB}" ]]; then
  OPENPMD_MPI_FLAVOR="$(detect_library_mpi_flavor "${OPENPMD_LIB}")"
  if [[ "${OPENPMD_MPI_FLAVOR}" != "unknown" && "${MPI_FLAVOR}" != "unknown" && "${OPENPMD_MPI_FLAVOR}" != "${MPI_FLAVOR}" ]]; then
    echo "### ERROR: openPMD library at ${OPENPMD_LIB} appears to be built with ${OPENPMD_MPI_FLAVOR}, but mpicxx is ${MPI_FLAVOR}. Please remove ${OPENPMD_BUILD} (and ${OPENPMD_PREFIX}) and rebuild with a consistent MPI stack." >&2
    exit 1
  fi
fi

cmake -S "${OPENPMD_SRC}" -B "${OPENPMD_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${OPENPMD_PREFIX}" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DopenPMD_BUILD_CLI_TOOLS=OFF \
  -DopenPMD_USE_MPI=ON \
  -DopenPMD_USE_PYTHON=OFF \
  -DopenPMD_USE_ADIOS2=ON \
  -DopenPMD_USE_HDF5=OFF
cmake --build "${OPENPMD_BUILD}" --target openPMD -- -j8
cmake --install "${OPENPMD_BUILD}"

python3 configure.py "${CONFIGURE_ARGS[@]}"
make -j8
