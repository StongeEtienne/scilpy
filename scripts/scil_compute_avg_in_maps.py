#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the statistics (mean, std) of scalar maps, which can represent
diffusion metrics, in a ROI.

The mask can either be a binary mask, or a weighting mask. If the mask is
a weighting mask it should either contain floats between 0 and 1 or should be
normalized with --normalize_weights.

IMPORTANT: if the mask contains weights (and not 0 and 1 exclusively), the
standard deviation will also be weighted.
"""

import argparse
import json
import logging
import os
import sys

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_json_args,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import weighted_mean_std


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_masks', nargs='+',
                   help='Masks volume filename.\nCan be a binary mask or a '
                        'weighted mask.')

    p.add_argument('--metrics', nargs='+', required=True,
                   help='Metrics nifti filename. List of the names of '
                        'the metrics file, in nifti format.')

    p.add_argument('--masks_sum', action='store_true',
                   help='Compute the sum of all values in masks '
                        '(similar to vox count)')

    p.add_argument('--save_avg',
                   help='Save all average to a file (txt, numpy)')

    add_overwrite_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_masks + args.metrics)
    assert_outputs_exist(parser, args, [], optional=args.save_avg)

    # Load mask and validate content depending on flags
    nb_masks = len(args.in_masks)
    nb_metrics = len(args.metrics)
    mask_list = []
    mask_names = []

    for i in range(nb_masks):
        fname = args.in_masks[i]
        mask_img = nib.load(fname)

        if len(mask_img.shape) > 3:
            logging.error('Mask should be a 3D image.')

        # Can be a weighted image
        mask_data = mask_img.get_fdata(dtype=np.float32)
        if np.min(mask_data) < 0:
            logging.error('Mask should not contain negative values.')

        mask_list.append(mask_img.get_fdata(dtype=np.float32))
        mask_names.append(split_name_with_nii(os.path.basename(fname))[0])

    if args.masks_sum:
        sum_mask_arr = np.zeros([nb_masks, 1])
        for i in range(nb_masks):
            sum_mask_arr[i] = np.sum(mask_list[i])

    # Load all metrics files.
    metrics_names = []
    all_avg = np.zeros((nb_masks, nb_metrics))
    all_std = np.zeros((nb_masks, nb_metrics))
    for j in range(nb_metrics):
        fname = args.metrics[j]
        metrics_names.append(split_name_with_nii(os.path.basename(fname))[0])
        metric_data = nib.load(fname).get_fdata(dtype=np.float32)
        for i in range(nb_masks):
            avg, std = weighted_mean_std(mask_list[i], metric_data)
            all_avg[i, j] = avg
            all_std[i, j] = std

    if args.masks_sum:
        avg_data = np.hstack([all_avg, sum_mask_arr])
    else:
        avg_data = all_avg

    # Save / output
    if args.save_avg:
        _, file_ext = os.path.splitext(args.save_avg)

        if file_ext == ".json":
            json_stats = {}
            for i in range(nb_masks):
                mask_dict = {}
                for j in range(nb_metrics):
                    mask_dict[metrics_names[j]] = {
                        'mean': all_avg[i, j],
                        'std': all_std[i, j]
                    }
                if args.masks_sum:
                    mask_dict['sum'] = np.squeeze(sum_mask_arr)[i]
                json_stats[mask_names[i]] = mask_dict
            with open(args.save_avg, 'w') as fp:
                json.dump(json_stats, fp, indent=args.indent,
                          sort_keys=args.sort_keys)

        elif file_ext == ".npy":
            np.save(args.save_avg, avg_data)
        elif file_ext == ".csv":
            np.savetxt(args.save_avg, avg_data, delimiter=",")
        else:
            np.savetxt(args.save_avg, avg_data, delimiter=",", newline=";")
    else:
        np.savetxt(sys.stdout, avg_data, delimiter=",", newline=";")


if __name__ == "__main__":
    main()
