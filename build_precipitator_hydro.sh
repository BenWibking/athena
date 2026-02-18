#!/bin/bash
set -euxo pipefail

OPENPMD_SRC="extern/openPMD-api"
OPENPMD_BUILD="${OPENPMD_SRC}/build"
OPENPMD_PREFIX="${OPENPMD_BUILD}/install"
ENABLE_OPENPMD="${ENABLE_OPENPMD:-0}"

detect_hdf5_prefix() {
  local candidate prefix tool
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

  for tool in h5pcc h5cc; do
    candidate="$(command -v "${tool}" 2>/dev/null || true)"
    if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
      continue
    fi

    prefix="$("${candidate}" -showconfig 2>/dev/null \
      | awk -F: '/Installation point/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
    if [[ -n "${prefix}" && ( -d "${prefix}/lib" || -d "${prefix}/lib64" ) ]]; then
      echo "${prefix}"
      return 0
    fi
  done

  return 1
}

classify_mpi_flavor_from_text() {
  local lowered="$1"
  # Intel MPI often appears as oneAPI MPI paths or impi module paths.
  if grep -Eqi '(^|[^[:alnum:]_])(intel[[:space:]_-]*mpi|i_mpi|oneapi/mpi|/impi[0-9_/.-]*)([^[:alnum:]_]|$)' <<<"${lowered}"; then
    echo "intelmpi"
  elif grep -Eqi '(^|[^[:alnum:]_])(mpich|hydra|libmpich)([^[:alnum:]_]|$)' <<<"${lowered}"; then
    echo "mpich"
  elif grep -Eqi '(^|[^[:alnum:]_])(open[[:space:]_-]*mpi|ompi|libmpi_ompi|libopen-rte|libopen-pal)([^[:alnum:]_]|$)' <<<"${lowered}"; then
    echo "openmpi"
  else
    echo "unknown"
  fi
}

detect_mpi_flavor() {
  local output lowered
  if ! command -v mpicxx >/dev/null 2>&1; then
    echo "unknown"
    return 1
  fi
  output=$(mpicxx -show 2>/dev/null || true)
  lowered=$(printf '%s' "${output}" | tr '[:upper:]' '[:lower:]')
  classify_mpi_flavor_from_text "${lowered}"
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
  classify_mpi_flavor_from_text "${lowered}"
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
  local prefix="${1-}" tool config lowered parallel_setting compiler_hints flavor
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
  parallel_setting=$(awk -F: '
    /^[[:space:]]*parallel hdf5[[:space:]]*:/ {
      v=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      print tolower(v)
      exit
    }' <<<"${config}")
  if [[ -n "${parallel_setting}" && ! "${parallel_setting}" =~ ^(on|yes|true|1)$ ]]; then
    echo "serial"
    return 0
  fi

  compiler_hints=$(
    awk -F: '
      /^[[:space:]]*(C Compiler|Fortran Compiler|C\+\+ Compiler|Extra libraries|LDFLAGS|AM_LDFLAGS)[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print tolower(v)
      }' <<<"${config}"
  )
  flavor=$(classify_mpi_flavor_from_text "${compiler_hints}")
  if [[ "${flavor}" != "unknown" ]]; then
    echo "${flavor}"
    return 0
  fi

  flavor=$(classify_mpi_flavor_from_text "${lowered}")
  if [[ "${flavor}" != "unknown" ]]; then
    echo "${flavor}"
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
CONFIGURE_ARGS=(-hdf5 --prob precipitator --flux hllc --coord spherical_polar -mpi --nghost=3)
if [[ "${ENABLE_OPENPMD}" == "1" ]]; then
  CONFIGURE_ARGS=(-openpmd --openpmd_path "${OPENPMD_PREFIX}" "${CONFIGURE_ARGS[@]}")
  echo "### openPMD support enabled (ENABLE_OPENPMD=1)"
else
  echo "### openPMD support disabled (set ENABLE_OPENPMD=1 to enable)"
fi
if [[ -n "${HDF5_PREFIX}" ]]; then
  echo "### Using HDF5 from ${HDF5_PREFIX}"
  CONFIGURE_ARGS+=(--hdf5_path "${HDF5_PREFIX}")
else
  echo "### HDF5 not detected in environment; relying on default search paths" >&2
fi

MPI_FLAVOR="$(detect_mpi_flavor || true)"
HDF5_MPI_FLAVOR="$(detect_hdf5_mpi_flavor "${HDF5_PREFIX}")"
if [[ "${HDF5_MPI_FLAVOR}" == "serial" ]]; then
  echo "### ERROR: Detected serial HDF5 at ${HDF5_PREFIX}, but this build uses -mpi and requires parallel HDF5. Set HDF5_ROOT/HDF5_DIR/HDF5_HOME to a parallel HDF5 installation and rerun." >&2
  exit 1
fi
if [[ "${HDF5_MPI_FLAVOR}" != "unknown" && "${HDF5_MPI_FLAVOR}" != "mpi-unknown" ]]; then
  if [[ "${MPI_FLAVOR}" != "${HDF5_MPI_FLAVOR}" ]]; then
    echo "### ERROR: HDF5 was built with MPI (${HDF5_MPI_FLAVOR}) while mpicxx appears to use '${MPI_FLAVOR}'. Mixing MPI implementations will fail; rebuild HDF5 or switch to a matching MPI toolchain." >&2
    exit 1
  fi
fi

if [[ "${ENABLE_OPENPMD}" == "1" ]]; then
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
fi

if [[ -f Makefile ]]; then
  make clean
fi

python3 configure.py "${CONFIGURE_ARGS[@]}"
make -j8
