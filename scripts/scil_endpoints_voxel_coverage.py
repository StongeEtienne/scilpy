#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

from dipy.io.utils import get_reference_info
import nibabel as nib
import numpy as np
from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)

from dipy.utils.optpkg import optional_package
vtk_u, _, _ = optional_package('trimeshpy.vtk_util')


REMOVED_INDICES = np.iinfo(np.int32).min


DESCRIPTION = """
Script to filter and cut streamlines based on mesh surfaces and ROI.
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('endpoints',
                   help='Input endpoints in VOX(.npy)')

    p.add_argument('label_volume',
                   help="Volume ref (.nii)")

    p.add_argument('valid_labels', nargs='+', type=int,
                   help="Surface labels to use file")

    p.add_argument('--label_to_count', nargs='+', type=int, default=None,
                   help="Surface labels to use file")


    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.label_to_count is None:
        args.label_to_count = args.valid_labels
    (affine, dims, voxel_sizes, voxel_order) = get_reference_info(args.label_volume)
    label_vox = nib.load(args.label_volume).get_data().flatten()
    nb_vox_total = np.count_nonzero(np.isin(label_vox, args.label_to_count))

    #print(nb_vox_total)

    endpts = np.load(args.endpoints)
    # Compute indices from positions
    endids_a = np.ravel_multi_index(np.round(endpts[:, 0].T).astype(np.int), dims).flatten()
    endids_b = np.ravel_multi_index(np.round(endpts[:, 1].T).astype(np.int), dims).flatten()
    endids = np.stack((endids_a, endids_b), axis=-1)


    endlabel = label_vox[endids]
    valid_stl_mask = np.all(np.isin(endlabel, args.valid_labels), axis=1)

    valid_labels_list = endlabel[valid_stl_mask].flatten()
    valid_ids_list = endids[valid_stl_mask].flatten()
    valid_endpts_mask = np.isin(valid_labels_list, args.label_to_count)
    valid_end_ids = valid_ids_list[valid_endpts_mask]
    nb_unique_endpts_vox = len(np.unique(valid_end_ids))

    #print(nb_unique_endpts_vox)

    print(float(nb_unique_endpts_vox)/float(nb_vox_total))


if __name__ == "__main__":
    main()
