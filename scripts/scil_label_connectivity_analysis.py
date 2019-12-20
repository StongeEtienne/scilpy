#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import argparse
import logging
import os

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

    p.add_argument('labels_per_stl', nargs='+',
                   help='Input label per stl (.npy)')

    p.add_argument('--labels', nargs='+', type=int,
                   help="Surface intersections file")

    p.add_argument('--test_test', action='store_true')

    p.add_argument('--test_retest', action='store_true',
                   help=("test-retest computation based on the metric given,"
                         + " use the prefix before '_' to match subject"))

    p.add_argument('--distance_metric', type=str, default='chisqr',
                   help="connectivity_matrix distances")

    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, required=args.labels_per_stl)
    #assert_outputs_exist(parser, args, args.output_connectivity_matrix)

    # Get number of label with and without the filler
    nb_co_m = len(args.labels_per_stl)
    nb_labels = len(args.labels)
    co_shape = (nb_labels, nb_labels)

    args.labels_per_stl.sort()
    co_matrices = np.zeros([nb_co_m, co_shape[0], co_shape[0]])
    for i in range(nb_co_m):
        labels_per_stl = np.load(args.labels_per_stl[i])
        co_matrices[i] = labels_to_coo(labels_per_stl, args.labels)

    dist_matrix = np.zeros([nb_co_m, nb_co_m])
    for i in range(nb_co_m):
        for j in range(i):
            dist_matrix[i, j] = m_dist(co_matrices[i], co_matrices[j],
                                       norm=True, symetrize=True,
                                       diag_to_zero=True,
                                       dist=args.distance_metric)
            dist_matrix[j, i] = dist_matrix[i, j]

    # if args.out_dist:
    #     np.save(args.out_dist, dist_matrix)
    #
    # if args.out_csv:
    #     np.savetxt(args.out_csv, dist_matrix, delimiter=",")

    if args.test_retest:
        # Generate a list of distance (inter + intra) per subject, for each
        intra_dists_per_sub = []
        inter_dists_per_sub = []
        for i in range(nb_co_m):
            intra_dists_i = []
            inter_dists_i = []
            prefix_i = os.path.basename(args.labels_per_stl[i]).split("_")[0]
            for j in range(nb_co_m):
                prefix_j = os.path.basename(args.labels_per_stl[j]).split("_")[0]

                if i != j:
                    if prefix_i == prefix_j:
                        intra_dists_i.append(dist_matrix[i, j])
                    else:
                        inter_dists_i.append(dist_matrix[i, j])
            intra_dists_per_sub.append(intra_dists_i)
            inter_dists_per_sub.append(inter_dists_i)

        # Compute the average per subject
        intra_dists_avgs = []
        inter_dists_avgs = []
        for intra_dists in intra_dists_per_sub:
            if intra_dists:
                intra_dists_avgs.append(np.mean(intra_dists))
        for inter_dists in inter_dists_per_sub:
            if inter_dists:
                inter_dists_avgs.append(np.mean(inter_dists))

        # Compute the final average
        print("Intra subject distance (avg, std)")
        print((np.mean(intra_dists_avgs), np.std(intra_dists_avgs)))
        print("Inter subject distance (avg, std)")
        print((np.mean(inter_dists_avgs), np.std(inter_dists_avgs)))
        print("Inter/Intra ratio")
        print((np.mean(inter_dists_avgs) / np.mean(intra_dists_avgs)))


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


def m_dist(a, b, norm=True, symetrize=True, diag_to_zero=True, dist='chisqr'):
    k_shift = 0
    if diag_to_zero:
        np.fill_diagonal(a, 0.0)
        np.fill_diagonal(b, 0.0)
        k_shift = 1

    if symetrize:
        a = a + a.T
        b = b + b.T

        a = a[np.triu_indices(len(a), k=k_shift)]
        b = b[np.triu_indices(len(b), k=k_shift)]

    if norm:
        a = a / np.sum(a)
        b = b / np.sum(b)
    dist = dist.lower()
    if dist == 'chisqr':  # PDFs Chi dquare
        mask = ((a + b) > 0.0)
        return np.sum(np.square(a[mask] - b[mask]) / (a[mask] + b[mask])) / 2.0
    elif dist == 'l1':
        return np.sum(np.abs(a - b))  # L1
    elif dist == 'l2':
        return np.sqrt(np.sum(np.square(a - b)))  # L2
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
