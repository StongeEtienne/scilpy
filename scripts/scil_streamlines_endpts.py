#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length
from dipy.io.utils import get_reference_info
import nibabel as nib
import numpy as np

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.tractanalysis.todi_util import streamlines_to_endpoints


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='get streamlines lengths.')

    p.add_argument('label',
                   help='label (.nii)')
    p.add_argument('in_tractograms', nargs='+', default=[],
                   help='Streamlines file name.')

    p.add_argument('--out_lengths',
                   help='Streamlines lengths file name. (.npy)')
    p.add_argument('--out_vts',
                   help='Streamlines vts position file name. (.npy)')
    p.add_argument('--out_ids',
                   help='Streamlines vts indices file name. (.npy)')

    p.add_argument('--out_labels_id',
                   help='Streamlines endpoints labels file name. (.npy)')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_tractograms + [args.label])
    assert_outputs_exist(parser, args, [],
                         optional=[args.out_lengths, args.out_vts,
                                   args.out_ids, args.out_labels_id])

    (affine, dims, voxel_sizes, voxel_order) = get_reference_info(args.label)
    label_vox = nib.load(args.label).get_data().flatten()
    lengths_list = []
    endpts_list = []
    endids_list = []
    endlabel_list = []

    for tractogram in args.in_tractograms:
        try:
            stateful_tractogram = load_tractogram(tractogram, args.label)
        except ValueError:
            logging.warning("{} Some streamlines had out of "
                            "the image coordinates".format(tractogram))
            stateful_tractogram = load_tractogram(tractogram, args.label,
                                                  bbox_valid_check=False)

        assert(stateful_tractogram.streamlines)

        # Compute lenghts
        if args.out_lengths:
            lengths_list.append(length(stateful_tractogram.streamlines))

        # Change to voxel space
        stateful_tractogram.to_vox()
        stl = list(stateful_tractogram.streamlines)

        # Compute positions
        endpts = np.swapaxes(streamlines_to_endpoints(stl), 0, 1)
        # Force bbox
        endpts[:, :, 0] = np.clip(endpts[:, :, 0], 0, dims[0] - 1)
        endpts[:, :, 1] = np.clip(endpts[:, :, 1], 0, dims[1] - 1)
        endpts[:, :, 2] = np.clip(endpts[:, :, 2], 0, dims[2] - 1)

        if args.out_vts:
            endpts_list.append(endpts)

        # save some memory
        del stl, stateful_tractogram

        # Compute indices from positions
        endids_a = np.ravel_multi_index(np.round(endpts[:, 0].T).astype(np.int), dims).flatten()
        endids_b = np.ravel_multi_index(np.round(endpts[:, 1].T).astype(np.int), dims).flatten()

        if args.out_ids:
            endids = np.stack((endids_a, endids_b), axis=-1)
            endids_list.append(endids)

        # Compute labels from indices
        endlabel = np.stack((label_vox[endids_a], label_vox[endids_b]), axis=-1)
        if args.out_labels_id:
            endlabel_list.append(endlabel)

    if args.out_lengths:
        stacked_lenghts = np.concatenate(lengths_list, axis=0)
        np.save(args.out_lengths, stacked_lenghts)

    if args.out_vts:
        stacked_endpts = np.concatenate(endpts_list, axis=0)
        np.save(args.out_vts, stacked_endpts)

    if args.out_ids:
        stacked_endids = np.concatenate(endids_list, axis=0)
        np.save(args.out_ids, stacked_endids)

    if args.out_labels_id:
        stacked_endlab = np.concatenate(endlabel_list, axis=0)
        np.save(args.out_labels_id, stacked_endlab)


if __name__ == "__main__":
    main()
