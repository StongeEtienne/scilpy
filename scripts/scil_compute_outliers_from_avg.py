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
import nibabel as nib
import numpy as np
from scipy.stats import zscore

from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import weighted_mean_std


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
            s = io.BytesIO(open(f, 'rb').read().replace(b';', b'\n'))
            data = np.loadtxt(s, dtype=float, delimiter=",")
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
    print(full_zscore[1, 2, 3])
    full_abs_zscore = np.abs(full_zscore)
    err_mask = full_abs_zscore > args.zscore_error
    warn_mask = np.logical_and(full_abs_zscore > args.zscore_warning, np.logical_not(err_mask))
    error_list = np.argwhere(err_mask)
    warn_list = np.argwhere(warn_mask)

    for (id, row, col) in error_list:
        f_name = non_empty_files[id]
        z = full_zscore[id, row, col]
        if args.masks_name:
            row = args.masks_name[row]
        if args.metrics_name:
            col = args.metrics_name[col]
        print(f"Error, {f_name}, in {row} with {col}, zscore : {z}")

    for (id, row, col) in warn_list:
        f_name = non_empty_files[id]
        z = full_zscore[id, row, col]
        if args.masks_name:
            row = args.masks_name[row]
        if args.metrics_name:
            col = args.metrics_name[col]
        print(f"Warning, {f_name}, in {row} with {col}, zscore : {z}")
    #print(error_list)
    #print(warn_list)




if __name__ == "__main__":
    main()
