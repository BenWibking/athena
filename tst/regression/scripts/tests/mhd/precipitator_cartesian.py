"""Cartesian precipitator smoke test."""

import glob
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
    athena.configure('b', 'hdf5', prob='precipitator', coord='cartesian',
                     **kwargs)
    athena.make()


def run(**kwargs):
    arguments = [
        'job/problem_id=precipitator_cartesian_smoke',
        'time/nlim=5',
        'time/tlim=0.1',
        'time/ncycle_out=0',
        'precipitator/hse_profile_filename=../../../inputs/hse_3.0.txt',
        'precipitator/force_free_param_file=../../../inputs/force_free_params.txt',
        'precipitator/inner_sponge_tau=0.0',
        'precipitator/outer_sponge_tau=0.0',
        'precipitator/enable_outer_buffer_halo=0',
        'precipitator/enable_powerlaw_cooling=0',
        'precipitator/enable_heating=none',
        'mesh/nx1=16',
        'mesh/nx2=16',
        'mesh/nx3=16',
        'meshblock/nx1=8',
        'meshblock/nx2=8',
        'meshblock/nx3=8',
        'output2/variable=prim,divB',
        'output2/dt=1.0e-3',
    ]
    athena.run('mhd/athinput.precipitator_cartesian', arguments)


def analyze():
    outputs = sorted(glob.glob('bin/precipitator_cartesian_smoke.prim.*.athdf'))
    if not outputs:
        return False
    data = athena_read.athdf(outputs[-1])

    for values in data.values():
        if not np.all(np.isfinite(values)):
            return False

    divb = np.asarray(data['divB'])
    max_divb = np.max(np.abs(divb))
    vel1 = np.asarray(data['vel1'])
    vel2 = np.asarray(data['vel2'])
    vel3 = np.asarray(data['vel3'])
    max_speed = np.max(np.sqrt(vel1*vel1 + vel2*vel2 + vel3*vel3))

    return (np.isfinite(max_divb) and max_divb < 1.0e-6 and
            np.isfinite(max_speed) and max_speed < 5.0e-4)
