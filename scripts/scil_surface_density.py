#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import nibabel as nib
import numpy as np
from scipy.spatial.ckdtree import cKDTree
from scilpy.io.utils import add_overwrite_arg

from dipy.utils.optpkg import optional_package
from trimeshpy import vtk_util as vtk_u
from fury import window, actor


REMOVED_INDICES = np.iinfo(np.int32).min


DESCRIPTION = """
Script to filter and cut streamlines based on mesh surfaces and ROI.
"""

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('denity',
                   help='Input density map(.npy)')

    p.add_argument('surfaces',
                   help='Input surfaces (.vtk)')

    p.add_argument('--labels',
                   help='Vertices labels (.npy)')

    p.add_argument('--surfaces_id',
                   help='Input surfaces id / type (.npy)')

    p.add_argument('--indices', type=int, nargs='+', default=[1],
                   help='Indices surfaces_id')

    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load meshes
    vertices = vtk_u.get_polydata_vertices(vtk_u.load_polydata(args.surfaces))
    surfaces_labels = np.load(args.surfaces_labels)
    #wm_mask = surfaces_labels != REMOVED_INDICES

    wm_vertices_vox = vtk_u.vtk_to_vox(vertices, nib.load(args.ref))
    kdtree = cKDTree(wm_vertices_vox)

    # Closest points searching
    endpoints = np.load(args.endpoints)
    nb_labels = surfaces_labels.max()+1
    coo_shape = (nb_labels, nb_labels)

    #wm_surface_label = surfaces_labels[wm_mask]
    assert len(surfaces_labels) == len(wm_vertices_vox)

    # compute distance with current surface (kd-tree query)
    dist, indices = kdtree.query(
        endpoints, k=1, distance_upper_bound=args.max_distance,
        n_jobs=args.processes)

    # distance compared to other surfaces ****
    valid_nearest = np.all(np.isfinite(dist), axis=1)
    nearest_id = indices[valid_nearest]
    label_per_endpts = surfaces_labels[nearest_id]

    valid_labels_id = np.argwhere(~np.any(label_per_endpts == REMOVED_INDICES, axis=1)).squeeze()
    valid_labels = label_per_endpts[valid_labels_id]

    #assert np.all(valid_labels >= 0)

    # construct the connectivity matrix
    indices_1d = np.ravel_multi_index(valid_labels.T, dims=coo_shape)
    bins = np.bincount(indices_1d, minlength=np.prod(coo_shape))
    coo = bins.reshape(coo_shape)
    np.save(args.output_connectivity_matrix, coo)

    # count endpts on vertices
    #valid_vts_id = nearest_id[valid_labels_id].flatten()
    valid_vts_id = nearest_id.flatten()
    wm_vts_count = np.bincount(valid_vts_id, minlength=len(wm_vertices_vox))
    #full_vts_count = np.zeros_like(surfaces_labels)
    #full_vts_count[wm_mask] = wm_vts_count

    #scene = window.Scene()
    #scene.add(actor.dots(endpoints.reshape(-1, 3).astype(np.float32), color=(1., 0., 0.), dot_size=1))
    #scene.add(actor.dots(wm_vertices_vox.astype(np.float32), color=(0.8, 0.8, 0.8), dot_size=1))
    #window.show(scene)

    np.save(args.output_vts_count, wm_vts_count)


if __name__ == "__main__":
    main()
