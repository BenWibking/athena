#! /usr/bin/env python3

"""
Plot total magnetic and thermal energies from an Athena++ .hst history file.

The Athena++ history output contains magnetic energy partitioned by direction
(1-ME, 2-ME, 3-ME) and kinetic energy partitioned by direction (1-KE, 2-KE,
3-KE). The total thermal/internal energy is derived from the conserved total
energy column (tot-E):

  E_th = tot-E - (1-KE + 2-KE + 3-KE) - (1-ME + 2-ME + 3-ME)
  E_mag = 1-ME + 2-ME + 3-ME
"""

import argparse
import os
import tempfile

import athena_read


def _require_columns(data, names, filename):
    missing = [name for name in names if name not in data]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required columns in {filename!s}: {missing_list}. "
            "Check that NON_BAROTROPIC_EOS and MAGNETIC_FIELDS_ENABLED are on, "
            "and that history output is enabled."
        )


def main(**kwargs):
    input_file = kwargs["input_file"]
    output_file = kwargs["output_file"]
    raw = kwargs["raw"]
    title = kwargs["title"]
    y_log = kwargs["y_log"]

    data = athena_read.hst(input_file, raw=raw)

    _require_columns(
        data,
        ["time", "tot-E", "1-KE", "2-KE", "3-KE", "1-ME", "2-ME", "3-ME"],
        input_file,
    )

    time = data["time"]
    total_ke = data["1-KE"] + data["2-KE"] + data["3-KE"]
    total_me = data["1-ME"] + data["2-ME"] + data["3-ME"]
    total_thermal = data["tot-E"] - total_ke - total_me

    if output_file != "show":
        # Ensure Matplotlib can write its cache in restricted environments.
        tmp_dir = os.path.join(tempfile.gettempdir(), "athena_matplotlib_cache")
        os.makedirs(tmp_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", tmp_dir)
        os.environ.setdefault("XDG_CACHE_HOME", tmp_dir)

        import matplotlib

        matplotlib.use("agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(time, total_me, label="E_mag = 1-ME+2-ME+3-ME")
    ax.plot(time, total_thermal, label="E_th = tot-E - KE - ME")
    ax.set_xlabel("time")
    ax.set_ylabel("energy (history integral)")
    if title is not None:
        ax.set_title(title)
    if y_log:
        ax.set_yscale("log")
    ax.legend(loc="best")

    if output_file == "show":
        plt.show()
    else:
        fig.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot total magnetic and thermal energies vs time from an Athena++ .hst file."
        )
    )
    parser.add_argument("input_file", help="path to input .hst file")
    parser.add_argument(
        "output_file",
        help='output plot filename, or "show" for interactive display',
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="do not prune stale data from previous runs (passed to athena_read.hst)",
    )
    parser.add_argument("--title", help="plot title")
    parser.add_argument(
        "--y_log",
        action="store_true",
        help="use logarithmic y-axis",
    )
    args = parser.parse_args()
    main(**vars(args))
