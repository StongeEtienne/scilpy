#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamlinespeed import length
import numpy as np
from dipy.io.utils import get_reference_info
from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.tractanalysis.todi_util import streamlines_to_endpoints


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='TODO')

    p.add_argument('tractogram',
                   help='Streamlines file name.')

    p.add_argument('metric',
                   help='metric (.nii)')

    p.add_argument('surface',
                   help='Input surfaces (.vtk)')

    p.add_argument('oui_metric_per_vts',
                   help='Output metric alors surfaces (.npy)')

    p.add_argument('--method', default="mean",
                   help='Method used, ')


    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, args.in_tractograms + [args.ref])
    assert_outputs_exist(parser, args, [args.out_lengths, args.out_vts, args.out_ids])

    (affine, dims, voxel_sizes, voxel_order) = get_reference_info(args.ref)

    stateful_tractogram = load_tractogram(args.tractogram, args.ref)
    mesh = load_mesh_from_file(args.surface)

    # 1) Estimates metric for each streamlines points

    # 2) regroup streamlines metric into a single value per streamlines

    # 3) Find nearest surfaces points

    # 4) Project streamlines metrics to surfaces vertices

    # 5) regroup vertices metric into a single value per vertex


if __name__ == "__main__":
    main()
