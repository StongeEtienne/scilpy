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

EPS = 1.0e-8
REACHED = 0.5


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

    p.add_argument('--nb_resampling', type=int, default=10)

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

    index = 0
    results = np.zeros([len(args.norm_to), 5])
    if args.nb_resampling is None:
        for norm_v in args.norm_to:

            surface_type = np.load(args.surface_type)
            surface_mask_type = np.zeros_like(surface_type, dtype=np.bool)
            for i in args.surface_type_to_use:
                surface_mask_type = np.logical_or(surface_mask_type, surface_type == i)

            surface_mask = np.load(args.surface_mask)
            vts_mask = np.logical_and(surface_mask_type, surface_mask)

            cotan_curv = mesh.vertices_cotan_curvature(area_weighted=True)
            pos_curv = cotan_curv < EPS
            neg_curv = cotan_curv > EPS

            nb_vts_in_wm = np.count_nonzero(vts_mask)
            nb_vts_in_pos_curv = np.count_nonzero(np.logical_and(vts_mask, pos_curv))
            nb_vts_in_neg_curv = np.count_nonzero(np.logical_and(vts_mask, neg_curv))

            cov_list = []
            pos_list = []
            neg_list = []
            ratio_list = []
            for i in range(args.nb_resampling):
                randint = np.random.choice(len(init_scalar), int(norm_v), p=(init_scalar/init_scalar_sum))

                vts_scalar = np.bincount(randint, minlength=len(init_scalar))
                nb_vts_in_wm_reached = np.count_nonzero(vts_scalar[vts_mask])

                cov_list.append(float(nb_vts_in_wm_reached) / float(nb_vts_in_wm))

                nb_vts_in_pos_curv_reached = np.count_nonzero(vts_scalar[np.logical_and(vts_mask, pos_curv)])
                nb_vts_in_neg_curv_reached = np.count_nonzero(vts_scalar[np.logical_and(vts_mask, neg_curv)])

                pos_list.append(float(nb_vts_in_pos_curv_reached)/float(nb_vts_in_pos_curv))
                neg_list.append(float(nb_vts_in_neg_curv_reached)/float(nb_vts_in_neg_curv))

                nb_pos_curv = vts_scalar[np.logical_and(vts_mask, pos_curv)].sum()
                nb_neg_curv = vts_scalar[np.logical_and(vts_mask, neg_curv)].sum()
                ratio_list.append(float(nb_pos_curv)/float(nb_pos_curv+nb_neg_curv))

            coverage = np.mean(cov_list)
            pos_coverage = np.mean(pos_list)
            neg_coverage = np.mean(neg_list)
            pos_ratio = np.mean(ratio_list)

            results[index] = [norm_v, coverage, pos_coverage, neg_coverage, pos_ratio]
            index += 1

    np.save(args.output, results)


if __name__ == "__main__":
    main()
