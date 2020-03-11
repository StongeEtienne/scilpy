#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import numpy as np
from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)

from trimeshpy.trimesh_vtk import TriMesh_Vtk


REMOVED_INDICES = np.iinfo(np.int32).min


DESCRIPTION = """
Script to filter and cut streamlines based on mesh surfaces and ROI.
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""

EPS = 0.000001


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('coverage_map',
                   help='Input coverage map (triangle by defailt (.npy)')

    p.add_argument('surface')

    p.add_argument('surface_type')

    p.add_argument('surface_mask')

    p.add_argument('curvature_vts_out')
    p.add_argument('curvature_weight')

    p.add_argument('--surface_type_to_use', nargs='+', type=int, default=[1])

    p.add_argument('--coverage_is_vts', action='store_true')
    # p.add_argument('--histo_min', type=float, default=-0.0001)
    # p.add_argument('--histo_max', type=float, default=0.0001)
    # p.add_argument('--histo_step', type=float, default=0.000001)

    add_overwrite_arg(p)
    return p

def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    mesh = TriMesh_Vtk(args.surface, None)
    if args.coverage_is_vts:
        init_scalar = np.load(args.coverage_map).astype(np.float)
    else:
        tri_scalar = np.load(args.coverage_map).astype(np.float)
        tv_map = mesh.triangle_vertex_map()
        init_scalar = np.squeeze(np.asarray(tv_map.T.dot(tri_scalar.T)))  # /3.0

    surface_type = np.load(args.surface_type)
    surface_mask_type = np.zeros_like(surface_type, dtype=np.bool)

    for i in args.surface_type_to_use:
        surface_mask_type = np.logical_or(surface_mask_type, surface_type == i)

    surface_mask = np.load(args.surface_mask)
    vts_mask = np.logical_and(surface_mask_type, surface_mask)

    curv_scalar = init_scalar[vts_mask]
    cotan_curv = mesh.vertices_cotan_curvature(area_weighted=True)
    cotan_curv_in_mask = cotan_curv[vts_mask]

    np.save(args.curvature_vts_out, cotan_curv_in_mask)
    np.save(args.curvature_weight, curv_scalar)


if __name__ == "__main__":
    main()
