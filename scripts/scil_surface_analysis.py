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
SET Analysis (coverage, density, bias)
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""

EPS = 0.000001
REACHED = 0.5


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('set_density',
                   help='Input density map, triangle count (.npy)')
    p.add_argument('surface',
                   help='Input surfaces (.vtk)')
    p.add_argument('surface_type',
                   help='Vertices type, GM-WM-SC (.npy)')
    p.add_argument('surface_mask',
                   help='Vertices mask, intersection mask (.npy)')

    p.add_argument('coverage_out',
                   help="Vertices reached, average of 'coverage_nb_resample'"
                        " trials with 'coverage_with' endpoints (.npy)")
    p.add_argument('vts_curvature_out',
                   help="Vertices curvatures (.npy)")
    p.add_argument('vts_weight_out',
                   help="Vertices weight output (.npy)")

    p.add_argument('--density_is_vts', action='store_true',
                   help='Input density map is in vertices count')

    p.add_argument('--surface_type_to_use', nargs='+', type=int, default=[1],
                   help='Surface type to use, normally 0=GM, 1=WM, 2=SC')
    p.add_argument('--coverage_with', nargs='+', type=int,  default=[1000000],
                   help='Number of endpoints used for the coverage estimate')
    p.add_argument('--coverage_nb_resample', type=int, default=10,
                   help='Number of resampling trial for coverage estimate')

    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load mesh
    mesh = TriMesh_Vtk(args.surface, None)
    if args.density_is_vts:
        init_scalar = np.load(args.set_density).astype(np.float)
    else:
        tri_scalar = np.load(args.set_density).astype(np.float)
        tv_map = mesh.triangle_vertex_map()
        init_scalar = np.squeeze(np.asarray(tv_map.T.dot(tri_scalar.T)))  # /3.0

    # Load surface type
    surface_type = np.load(args.surface_type)
    surface_mask_type = np.zeros_like(surface_type, dtype=np.bool)

    # Merge surface type to use
    for i in args.surface_type_to_use:
        surface_mask_type = np.logical_or(surface_mask_type, surface_type == i)

    # Load mask and combine it with surface types
    surface_mask = np.load(args.surface_mask)
    vts_mask = np.logical_and(surface_mask_type, surface_mask)

    # Compute curvatures
    curv_scalar = init_scalar[vts_mask]
    cotan_curv = mesh.vertices_cotan_curvature(area_weighted=True)
    cotan_curv_in_mask = cotan_curv[vts_mask]

    # Save local curvatures and vts weight
    np.save(args.vts_curvature_out, cotan_curv_in_mask)
    np.save(args.vts_weight_out, curv_scalar)

    # initial density sum
    init_scalar_sum = float(init_scalar.sum())

    # Curvature vts mask (positive and negative)
    pos_curv_mask = np.logical_and(vts_mask, cotan_curv < EPS)
    neg_curv_mask = np.logical_and(vts_mask, cotan_curv > EPS)

    # Count vts (also positive and negative curvature)
    nb_vts_in_wm = np.count_nonzero(vts_mask)
    nb_vts_in_pos_curv = np.count_nonzero(pos_curv_mask)
    nb_vts_in_neg_curv = np.count_nonzero(neg_curv_mask)

    index = 0
    results = np.zeros([len(args.coverage_with), 5])
    for norm_v in args.coverage_with:
        out_list = np.zeros([args.coverage_nb_resample, 4], dtype=np.float)
        for i in range(args.coverage_nb_resample):
            # Randomly choose endpoints
            randint = np.random.choice(len(init_scalar), int(norm_v), p=(init_scalar / init_scalar_sum))

            # Density and count based on the sampling
            vts_scalar = np.bincount(randint, minlength=len(init_scalar))
            nb_vts_in_wm_reached = np.count_nonzero(vts_scalar[vts_mask])

            # Coverage percentage
            out_list[i, 0] = float(nb_vts_in_wm_reached) / float(nb_vts_in_wm)

            # Curvature count
            pos_scalar = vts_scalar[pos_curv_mask]
            neg_scalar = vts_scalar[pos_curv_mask]
            nb_vts_in_pos_curv_reached = np.count_nonzero(pos_scalar)
            nb_vts_in_neg_curv_reached = np.count_nonzero(neg_scalar)

            # Curvature percentage
            out_list[i, 1] = float(nb_vts_in_pos_curv_reached) / float(nb_vts_in_pos_curv)
            out_list[i, 2] = float(nb_vts_in_neg_curv_reached) / float(nb_vts_in_neg_curv)

            # Curvature ratio
            nb_pos_curv = pos_scalar.sum()
            nb_neg_curv = neg_scalar.sum()
            out_list[i, 3] = float(nb_pos_curv) / float(nb_pos_curv + nb_neg_curv)

        result_i = np.mean(out_list, axis=0)

        results[index, 0] = norm_v
        results[index, 1:] = result_i
        index += 1

    np.save(args.coverage_out, results)


if __name__ == "__main__":
    main()
