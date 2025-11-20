#!/bin/bash
set -x
python3 configure.py -openpmd --openpmd_path extern/openPMD-api --prob precipitator --flux hlld --coord spherical_polar -mpi -b
make -j8
