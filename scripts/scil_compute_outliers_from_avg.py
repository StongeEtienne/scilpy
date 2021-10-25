#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the statistics average for multiples scalar maps in each ROI.
Masks (ROI) can either be a binary mask, or a weighting mask (PVE maps).

IMPORTANT: if the mask contains weights >= 0,
With json output the standard deviation is also computed and weighted.
"""

import argparse
import io
import numpy as np
from scipy.stats import zscore

from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             assert_inputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('averages', nargs='+',
                   help='Average array (.npy, .csv)')

    p.add_argument('--zscore_warning', type=float, default=2.5,
                   help='zscore for warning [%(default)s]')

    p.add_argument('--zscore_error', type=float, default=3.5,
                   help='zscore for error [%(default)s]')

    p.add_argument('--masks_name', nargs='+',
                   help='name for the vals')

    p.add_argument('--metrics_name', nargs='+',
                   help='name for the vals')

    add_overwrite_arg(p)
    add_json_args(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.averages)

    assert(args.zscore_warning <= args.zscore_error)

    all_files = args.averages
    all_files.sort()

    non_empty_files = []
    non_empty_avg = []
    empty_files = []
    for f in all_files:
        f_type = f.split(".")[-1]
        if f_type == "txt":
            s = io.BytesIO(open(f, 'rb').read().replace(b';', b'\n')).readlines()
            if s:
                data = np.loadtxt(s, dtype=float, delimiter=",")
            else:
                data = np.empty(shape=(0, 0))
        elif f_type == "csv":
            data = np.loadtxt(f, dtype=float, delimiter=",")
        elif f_type == "csv":
            data = np.load(f)

        if data.size == 0:
            empty_files.append(f)
        else:
            non_empty_files.append(f)
            non_empty_avg.append(data)

    full_avg = np.stack(non_empty_avg)
    full_zscore = zscore(full_avg, axis=0)
    full_abs_zscore = np.abs(full_zscore)
    err_mask = full_abs_zscore > args.zscore_error
    warn_mask = np.logical_and(full_abs_zscore > args.zscore_warning, np.logical_not(err_mask))
    error_list = np.argwhere(err_mask)
    warn_list = np.argwhere(warn_mask)

    for f in empty_files:
        print("Empty file," + str(f))

    id_err = []
    for (id, row, col) in error_list:
        f_name = non_empty_files[id]
        z = full_zscore[id, row, col]
        if args.masks_name:
            row = args.masks_name[row]
        if args.metrics_name:
            col = args.metrics_name[col]
        print(f"Error, {f_name}, in {row} with {col}, zscore : {z}")
        id_err.append(id)

    id_warn = []
    for (id, row, col) in warn_list:
        f_name = non_empty_files[id]
        z = full_zscore[id, row, col]
        if args.masks_name:
            row = args.masks_name[row]
        if args.metrics_name:
            col = args.metrics_name[col]
        print(f"Warning, {f_name}, in {row} with {col}, zscore : {z}")
        id_warn.append(id)

    nb_sub = len(args.averages)
    nb_with_err = len(np.unique(id_err))
    nb_with_warn = len(np.unique(id_warn))
    nb_with_err_or_warn = len(np.unique(id_err+id_warn))

    print(f"{nb_with_err} subjects with error ( ~{100.0*nb_with_err/nb_sub:.1f}% )")
    print(f"{nb_with_warn} subjects with warning ( ~{100.0*nb_with_warn/nb_sub:.1f}% )")
    print(f"Total of {nb_with_err_or_warn} subjects with error or warning,"
          f" from {len(args.averages)} subjects ( ~{100.0*nb_with_err_or_warn/nb_sub:.1f}% )")


if __name__ == "__main__":
    main()
