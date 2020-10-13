#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reshape / reslice / resample *.nii or *.nii.gz using a reference.
This script can be used to align freesurfer/civet output, as .mgz,
to the original input image.


>>> scil_reshape_to_reference.py wmparc.mgz t1.nii.gz wmparc_t1.nii.gz \\
    --interpolation nearest
"""

import argparse

import nibabel as nib

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_file',
                   help='Path of the image (.nii or .mgz) to be reshaped.')
    p.add_argument('transform_to_use',
                   choices=['sform', 'qform'],
                   help='Interpolation: "sform" or "qform".')
    p.add_argument('out_file',
                   help='Output filename of the reshaped image (.nii).')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_file)
    assert_outputs_exist(parser, args, args.out_file)

    nib_file = nib.load(args.in_file)
    if args.transform_to_use == "sform":
        transfo = nib_file.get_sform()
    else:
        transfo = nib_file.get_qform()

    new_nib = nib.Nifti1Image(nib_file.get_data(), transfo)
    nib.save(new_nib, args.out_file)


if __name__ == "__main__":
    main()
