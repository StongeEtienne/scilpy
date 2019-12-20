# -*- coding: utf-8 -*-

from os.path import splitext

from dipy.utils.optpkg import optional_package
from nibabel.freesurfer.io import read_geometry

trimeshpy, have_trimeshpy, _ = optional_package('trimeshpy')


SUPPORTED_VTK_EXTENSIONS = [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]


def load_mesh_from_file(mesh_file, mesh_assert=False):
    file_extension = splitext(mesh_file)[1].lower()
    if file_extension in SUPPORTED_VTK_EXTENSIONS:
        return load_mesh_with_vtk(mesh_file, mesh_assert=mesh_assert)
    else:
        return load_mesh_with_nibabel(mesh_file, mesh_assert=mesh_assert)


def load_mesh_with_vtk(mesh_file, mesh_assert=False):
    # Load surface with TriMeshPy, VTK supported formats
    if have_trimeshpy:
        return trimeshpy.TriMesh_Vtk(mesh_file, None, assert_args=mesh_assert)
    else:
        raise ImportError("TriMeshPy python module is not detected,"
                          " cannot load VTK supported formats")


def load_mesh_with_nibabel(mesh_file, mesh_assert=False):
    # Load surface with Nibabel (mainly Freesurfer surface)
    [vts, tris] = read_geometry(mesh_file)
    return trimeshpy.TriMesh_Vtk(tris, vts, assert_args=mesh_assert)
