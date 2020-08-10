#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""
Script to compute the mass-stiffness flow [1].
"""

import argparse
import logging
import os

import numpy as np
from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_args_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('surface',
                   help='Input surface (Freesurfer or supported by VTK)')

    p.add_argument('out_surface',
                   help='Output surface (formats supported by VTK)')

    p.add_argument('-m', '--vts_mask',
                   help='Vertices mask, where to apply the flow (.npy)')

    p.add_argument('-n', '--nb_steps', type=int, default=10,
                   help='Number of steps for the flow [%(default)s]')

    p.add_argument('-s', '--step_size', type=float, default=10.0,
                   help='Flow step size [%(default)s]')

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, required=args.surface,
                        optional=args.vts_mask)

    assert_outputs_exist(parser, args, args.out_surface)

    out_path = os.path.dirname(args.out_surface)
    if not os.path.isdir(out_path):
        parser.error('Output directory {} doesn\'t exist.'.format(out_path))

    # Check smoothing parameters
    if args.nb_steps < 2:
        parser.error("Number of steps should be 2 or higher")

    if args.step_size <= 0.0:
        parser.error("Step size should be strictly positive")

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load mesh
    mesh = load_mesh_from_file(args.surface)

    # Step size (zero for masked vertices)
    step_size_per_vts = args.step_size
    if args.vts_mask:
        extension = args.vts_mask.split(".")[-1].lower()
        if extension == ".npy":
            mask = np.load(args.vts_mask)
        else:
            mask = np.loadtxt(args.vts_mask)

        step_size_per_vts *= mask.astype(np.float)

    # Compute mass stiffness Flow
    vts = mesh.mass_stiffness_smooth(
        nb_iter=args.nb_steps,
        diffusion_step=step_size_per_vts,
        flow_file=None)

    # Update vertices and save mesh
    mesh.set_vertices(vts)
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
