#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import numpy as np
from dipy.utils.optpkg import optional_package

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exists)

trimeshpy, have_trimeshpy, _ = optional_package('trimeshpy')


DESCRIPTION = """
Script to de-concatenate (split) surfaces from a single file (vtk or freesurfer).
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('concatenated_surface',
                   help='Input basic surfaces; i.e. white matter'
                        '(supported by VTK)')

    p.add_argument('surfaces_indices_map',
                   help='Surface id map, given by concatenate,'
                        ' one per vertex (.npy)')

    p.add_argument('surface_id', type=int,
                   help='Surface id to output decatenate'
                        ' (white surfaces 0 and 1 by default)')

    p.add_argument('out_surface',
                   help='Output decatenated surface (supported by VTK)')

    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    assert_inputs_exist(parser, required=[args.concatenated_surface,
                                          args.surfaces_indices_map])
    assert_outputs_exists(parser, args, [args.out_surface])

    conc_mesh = trimeshpy.TriMesh_Vtk(args.concatenated_surface, None)
    dec_mask = np.load(args.surfaces_indices_map) == args.surface_id

    if not np.any(dec_mask):
        logging.error("Given surface_id does not exist in the indices_map")
    else:
        dec_vts = conc_mesh.get_vertices()[dec_mask]
        dec_tri_mask = np.any(dec_mask[conc_mesh.get_triangles()], axis=1)
        dec_tri = conc_mesh.get_triangles()[dec_tri_mask]
        dec_tri = dec_tri - np.min(dec_tri)

        dec_mesh = trimeshpy.TriMesh_Vtk(dec_tri, dec_vts)
        dec_mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
