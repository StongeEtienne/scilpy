#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to load and transform a surface (FreeSurfer or VTK supported),
This script is using ANTs transform (affine.txt, warp.nii.gz).

Best usage with vtk and ANTs from T1 to b0:
> ConvertTransformFile 3 output0GenericAffine.mat vtk_transfo.txt --hm
> scil_transform_surface.py lh_white_lps.vtk vtk_transfo.txt lh_white_b0.vtk\\
    --ants_warp output1InverseWarp.nii.gz

The input surface needs to be in *T1 world LPS* coordinates
(aligned over the T1 in MI-Brain).
The resulting surface should be aligned *b0 world LPS* coordinates
(aligned over the b0 in MI-Brain).

Best usage to reverse the transformation (ex. b0 -> T1)
scil_transform_surface.py lh_white_b0.vtk vtk_transfo.txt lh_white_t1_lps.vtk\\
    --ants_warp output1Warp.nii.gz --inverse
"""

import argparse
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from trimeshpy.io import load_mesh_from_file
import trimeshpy.vtk_util as vtk_u

from scilpy.io.utils import (add_overwrite_arg,
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

    p.add_argument('in_vertices',
                   help='Input surface or vertices (.txt, .npy).')

    p.add_argument('ants_affine',
                   help='Affine transform from ANTs (.txt).')

    p.add_argument('out_vertices',
                   help='Output surface(.txt, .npy).')

    p.add_argument('--inverse', action='store_true',
                   help='Inverse the transformation (and apply the warp first).')

    p.add_argument('--ants_warp',
                   help='Warp image from ANTs (NIfTI format).')

    p.add_argument('--out_fmt', default="%1.8f",
                   help='Out float format [%(default)s]')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_vertices, args.ants_affine],
                        args.ants_warp)
    assert_outputs_exist(parser, args, args.out_vertices)

    # Load vertices
    in_ext = os.path.splitext(args.in_vertices)[1].lower()
    if in_ext == ".txt":
        vts = np.loadtxt(args.in_vertices)
    elif in_ext == ".csv":
        vts = np.loadtxt(args.in_vertices, delimiter=',')
    elif in_ext == ".npy":
        vts = np.load(args.in_vertices)

    if vts.ndim == 1:
        vts = vts.reshape((1, -1))
    affine = np.loadtxt(args.ants_affine)

    print(vts[0])

    # apply transform
    if args.inverse:
        vts = apply_warp(vts, args.ants_warp)
        vts = apply_affine(vts, affine)
    else:
        vts = apply_affine(vts, np.linalg.inv(affine))
        print(vts[0])
        vts = apply_warp(vts, args.ants_warp)
        print(vts[0])

    # Save vertices
    out_ext = os.path.splitext(args.out_vertices)[1].lower()
    if out_ext == ".txt":
        np.savetxt(args.out_vertices, vts, fmt=args.out_fmt)
    elif out_ext == ".csv":
        np.savetxt(args.out_vertices, vts, delimiter=',', fmt=args.out_fmt)
    elif out_ext == ".npy":
        np.save(args.out_vertices, vts)


def apply_affine(vts, affine):
    # Transform mesh vertices
    return (np.dot(affine[:3, :3], vts.T) + affine[:3, 3:4]).T


def apply_warp(vts, warp_file):
    if warp_file:
        warp_img = nib.load(warp_file)
        warp = np.squeeze(warp_img.get_fdata(dtype=np.float32))

        # Get vertices translation in voxel space, from the warp image
        vts_vox = vtk_u.vtk_to_vox(vts, warp_img)
        tx = map_coordinates(warp[..., 0], vts_vox.T, order=1)
        ty = map_coordinates(warp[..., 1], vts_vox.T, order=1)
        tz = map_coordinates(warp[..., 2], vts_vox.T, order=1)

        # Apply vertices translation in world coordinates
        vts += np.array([tx, ty, tz]).T
    return vts


if __name__ == "__main__":
    main()
