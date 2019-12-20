#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse
import logging

import numpy as np

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)

DESCRIPTION = """
Script to compute connectivity matrix from surface intersections
generated from 'scil_surface_tractogram_filtering.py'.

The resulting connectivity matrix SUM to the number of given
intersections (streamlines), if no intersected indices are removed.
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('labels_per_stl',
                   help='Input label per stl (.npy)')

    p.add_argument('output_connectivity_matrix',
                   help="Surface intersections file (.npy)")

    p.add_argument('labels_to_use', nargs='+', type=int,
                   help="Surface labels to use file")

    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, required=args.labels_per_stl)
    assert_outputs_exist(parser, args, args.output_connectivity_matrix)

    # Get number of label with and without the filler

    input_labels = np.load(args.labels_per_stl)
    coo = labels_to_coo(input_labels, args.labels_to_use)

    np.save(args.output_connectivity_matrix, coo)
    return


def labels_to_coo(labeled_stl, labels_list):
    labels_list = np.asarray(labels_list)
    nb_labels = len(labels_list)
    coo_shape = (nb_labels, nb_labels)

    valid_labels_id = np.argwhere(np.all(np.isin(labeled_stl, labels_list), axis=1)).squeeze()
    valid_labels = labeled_stl[valid_labels_id]
    print(labeled_stl.shape)
    print(valid_labels.shape)

    ordered_labels = -np.ones_like(valid_labels, dtype=np.int)
    for i in range(nb_labels):
        ordered_labels[valid_labels == labels_list[i]] = i

    assert np.all(ordered_labels >= 0)

    # construct the connectivity matrix
    indices_1d = np.ravel_multi_index(ordered_labels.T, dims=coo_shape)
    bins = np.bincount(indices_1d, minlength=np.prod(coo_shape))
    coo = bins.reshape(coo_shape)

    return coo


if __name__ == "__main__":
    main()
