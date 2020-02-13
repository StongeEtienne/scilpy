#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

from dipy.io.utils import get_reference_info
import nibabel as nib
import numpy as np
from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)

import trimeshpy.vtk_util as vtk_u
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.math.util import dot, allclose_to, is_logging_in_debug


REMOVED_INDICES = np.iinfo(np.int32).min


DESCRIPTION = """
Script to filter and cut streamlines based on mesh surfaces and ROI.
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""

EPS = 0.0


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('coverage_map',
                   help='Input coverage map (triangle by defailt (.npy)')

    p.add_argument('surface')

    p.add_argument('surface_type')

    p.add_argument('surface_mask')

    p.add_argument('output')

    p.add_argument('--surface_type_to_use', nargs='+', type=int, default=[1])

    p.add_argument('--coverage_is_vts', action='store_true')

    p.add_argument('--norm_to', nargs='+', type=float,  default=[1000000])

    p.add_argument('--print', action='store_true')

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

    init_scalar_sum = float(init_scalar.sum())

    # for each normalization term (Streamlines count)
    if args.print:
        print(("norm", " :", "coverage", "pos_coverage", "neg_coverage", "global pos_ratio"))


    results = np.zeros([len(args.norm_to), 5])

    index = 0
    for norm_v in args.norm_to:
        vts_scalar = init_scalar*norm_v/init_scalar_sum

        surface_type = np.load(args.surface_type)
        surface_mask_type = np.zeros_like(surface_type, dtype=np.bool)
        for i in args.surface_type_to_use:
            surface_mask_type = np.logical_or(surface_mask_type, surface_type == i)

        surface_mask = np.load(args.surface_mask)
        vts_mask = np.logical_and(surface_mask_type, surface_mask)

        nb_vts_in_wm = np.count_nonzero(vts_mask)
        nb_vts_in_wm_reached = np.count_nonzero(vts_scalar[vts_mask] >= 1.0)

        # coverage
        coverage = float(nb_vts_in_wm_reached) / float(nb_vts_in_wm)

        curv_normal_mtx = mesh.mean_curvature_normal_matrix(area_weighted=True)
        direction = curv_normal_mtx.dot(mesh.get_vertices())
        normal_dir = mesh.vertices_normal(normalize=False)
        pos_curv = dot(direction, normal_dir, axis=1) < EPS
        neg_curv = dot(direction, normal_dir, axis=1) > EPS

        nb_vts_in_pos_curv = np.count_nonzero(np.logical_and(vts_mask, pos_curv))
        nb_vts_in_pos_curv_reached = np.count_nonzero(vts_scalar[np.logical_and(vts_mask, pos_curv)] >= 1.0)

        nb_vts_in_neg_curv = np.count_nonzero(np.logical_and(vts_mask, neg_curv))
        nb_vts_in_neg_curv_reached = np.count_nonzero(vts_scalar[np.logical_and(vts_mask, neg_curv)] >= 1.0)

        pos_coverage= float(nb_vts_in_pos_curv_reached)/float(nb_vts_in_pos_curv)
        neg_coverage = float(nb_vts_in_neg_curv_reached)/float(nb_vts_in_neg_curv)

        nb_pos_curv = vts_scalar[np.logical_and(vts_mask, pos_curv)].sum()
        nb_neg_curv = vts_scalar[np.logical_and(vts_mask, neg_curv)].sum()
        pos_ratio = float(nb_pos_curv)/float(nb_pos_curv+nb_neg_curv)

        if args.print:
            print((norm_v, " :", coverage, pos_coverage, neg_coverage, pos_ratio))

        results[index] = [norm_v, coverage, pos_coverage, neg_coverage, pos_ratio]
        index += 1

    np.save(args.output, results)


if __name__ == "__main__":
    main()
