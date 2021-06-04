#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to extract streamlines from a specific point and radius,
"""

import argparse
import os
import logging

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from fury import window, actor
import nibabel as nib
import numpy as np
from scipy.spatial.ckdtree import cKDTree
import trimeshpy.vtk_util as vtk_u

import scilpy.tractanalysis.todi_util as todi_u
from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)



EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file (trk or tck).')

    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file (trk or tck).')

    p.add_argument('reference',
                   help='Reference image.')

    p.add_argument('radius', type=float, default=10.0,
                   help='Radius for the search in mm [%(default)s]')

    p.add_argument('coord', nargs='+',
                   help='coordinates (.txt, .csv, .npy, or 3 float)')

    p.add_argument('--only_endpoints', action='store_true',
                   help='Search streamlines only from endpoints [%(default)s].')

    p.add_argument('--visualize', action='store_true',
                   help='Visualize streamlines with sphere.')
    add_overwrite_arg(p)
    add_reference_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram)

    if len(args.coord) == 3:
        coords = np.array(args.coord, dtype=float)
    if len(args.coord) == 1:
        coord_file = args.coord[0]
        in_ext = os.path.splitext(coord_file)[1].lower()
        if in_ext == ".txt":
            coords = np.loadtxt(coord_file)
        elif in_ext == ".csv":
            coords = np.loadtxt(coord_file, delimiter=',')
        elif in_ext == ".npy":
            coords = np.load(coord_file)
    else:
        logging.error("coordinates should be 3 floats or a single files with 3 values")

    if len(coords) != 3:
        logging.error("coordinates file should contain only 3 values")

    ref_img = nib.load(args.reference)
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram, bbox_check=False)
    sft.to_voxmm()
    sft.to_corner()
    slines = sft.streamlines

    coords = coords.reshape((1, 3))
    voxmm_pos = np.squeeze(vtk_u.vtk_to_voxmm(coords, ref_img))
    # Search streamlines near coordinates
    if args.only_endpoints:
        endpoints = np.zeros([len(slines), 2, 3], dtype=np.float)
        for i in range(len(slines)):
            # TODO optimise
            endpoints[i, 0] = slines[i][0]
            endpoints[i, 1] = slines[i][-1]
        endpts_tree = cKDTree(endpoints.reshape((-1, 3)))
        indices = np.asarray(endpts_tree.query_ball_point(voxmm_pos, r=args.radius))
        indices = np.unique(indices//2)

    else:
        indices = []
        for i in range(len(slines)):
            # TODO optimise
            sline_tree = cKDTree(slines[i])
            res = np.asarray(sline_tree.query_ball_point(voxmm_pos, r=args.radius))
            if len(res) > 0:
                indices.append(i)

    if len(indices) > 0:
        new_sft = StatefulTractogram.from_sft(slines[indices], sft)
    else:
        new_sft = StatefulTractogram.from_sft([], sft)
        logging.warning('The file {} contains 0 streamline'.format(
            args.out_tractogram))

    save_tractogram(new_sft, args.out_tractogram, bbox_valid_check=False)

    if args.visualize:
        vtk_scene = window.Scene()
        dot_actor = actor.sphere(np.array(voxmm_pos.reshape((-1, 3))), [0.8, 0, 0, 0.3], radii=args.radius)
        vtk_scene.add(dot_actor)
        line_actor = actor.line(new_sft.streamlines, colors=[0.0, 1.0, 0.0, 1.0])
        vtk_scene.add(line_actor)
        line_actor = actor.line(sft.streamlines, colors=[0.0, 0.0, 1.0, 0.3])
        vtk_scene.add(line_actor)
        window.show(vtk_scene, title='scil_extract_streamlines_in_sphere', order_transparent=True)


if __name__ == "__main__":
    main()
