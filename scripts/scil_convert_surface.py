#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert cortical surface (vtk or freesurfer).

> scil_convert_surface.py surf.vtk converted_surf.ply
"""

import argparse

from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (add_overwrite_arg,
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
                   help='Input surface (FreeSurfer or supported by VTK).')

    p.add_argument('out_surface',
                   help='Output flipped surface (formats supported by VTK).')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.surface)
    assert_outputs_exist(parser, args, args.out_surface)

    mesh = load_mesh_from_file(args.surface)
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()