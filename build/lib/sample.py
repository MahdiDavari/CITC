# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Pymatgen (Python Materials Genomics) is a robust, open-source Python library
for materials analysis. This is the root package.
"""


# Useful aliases for commonly used objects and modules.
# Allows from pymatgen import <class> for quick usage.
import os
import warnings
import ruamel.yaml as yaml
from fnmatch import fnmatch

__author__ = "Pymatgen Development Team"
__email__ = "pymatgen@googlegroups.com"
__maintainer__ = "Shyue Ping Ong"
__maintainer_email__ = "shyuep@gmail.com"
__version__ = "2019.10.16"



from .core.periodic_table import Element, Specie, DummySpecie



def loadfn(fname):
    """
    Convenience method to perform quick loading of data from a filename. The
    type of object returned depends the file type.
    Args:
        fname (string): A filename.
    Returns:
        Note that fname is matched using unix-style, i.e., fnmatch.
        (Structure) if *POSCAR*/*CONTCAR*/*.cif
        (Vasprun) *vasprun*
        (obj) if *json* (passthrough to monty.serialization.loadfn)
    """
    if (fnmatch(fname, "*POSCAR*") or fnmatch(fname, "*CONTCAR*") or
            ".cif" in fname.lower()) or fnmatch(fname, "*.vasp"):
        return Structure.from_file(fname)
    elif fnmatch(fname, "*vasprun*"):
        from pymatgen.io.vasp import Vasprun
        return Vasprun(fname)
    elif fnmatch(fname, "*.json*"):
        from monty.serialization import loadfn
        return loadfn(fname)
