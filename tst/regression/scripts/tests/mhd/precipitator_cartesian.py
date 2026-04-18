"""Cartesian precipitator smoke test."""

import os
import numpy as np
import scripts.utils.athena as athena
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..',
                 'vis', 'python')))
import athena_read  # noqa

athena_read.check_nan_flag = True


def prepare(**kwargs):
    athena.configure('b', prob='precipitator', coord='cartesian', **kwargs)
    athena.make()


def run(**kwargs):
    arguments = [
        'job/problem_id=precipitator_cartesian_smoke',
        'time/nlim=1',
        'time/tlim=1.0e-6',
        'time/ncycle_out=0',
        'precipitator/hse_profile_filename=../../../inputs/hse_3.0.txt',
        'precipitator/force_free_param_file=../../../inputs/force_free_params.txt',
        'mesh/nx1=16',
        'mesh/nx2=16',
        'mesh/nx3=16',
        'meshblock/nx1=8',
        'meshblock/nx2=8',
        'meshblock/nx3=8',
        'output2/variable=prim,divB,grav_phi,pressure_hse',
        'output2/dt=1.0e-6',
    ]
    athena.run('mhd/athinput.precipitator_cartesian', arguments)


def analyze():
    _, _, _, data = athena_read.vtk(
        'bin/precipitator_cartesian_smoke.block0.prim.00001.vtk')

    for values in data.values():
        if not np.all(np.isfinite(values)):
            return False

    divb = np.asarray(data['divB'])
    max_divb = np.max(np.abs(divb))
    return np.isfinite(max_divb) and max_divb < 1.0e-6
