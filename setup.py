"""
Script for building the example.

Usage:
    python setup.py build --build-lib=./
"""
import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
include_dirs = [numpy.get_include()]
# Determine the extra link arguments based on the platform
if sys.platform == 'win32':
    include_dirs.append(os.path.join(os.environ['ProgramFiles(x86)'], 'Windows Kits', '10', 'Include', '10.0.19041.0', 'um'))
    include_dirs.append(os.path.join(os.environ['ProgramFiles(x86)'], 'Windows Kits', '10', 'Include', '10.0.19041.0', 'shared'))
    libraries = ["opengl32", "glu32", "glew32"]
    extra_link_args = []
elif sys.platform == 'darwin':
    extra_link_args = ["-framework", "OpenGL"]
    libraries = []
elif sys.platform.startswith('linux'):
    extra_link_args = []
    libraries = ["GL", "GLU"]
else:
    extra_link_args = []
    libraries = ["GL", "GLU"]


extensions = [
    Extension("molGL", 
              # ["src/molGL.c"],
              ["src/molGL.pyx"],
              libraries = libraries,
              include_dirs = include_dirs,
              extra_link_args=extra_link_args
              )
    ]

RELEASE = "0.0.1"

setup(
    name="pyQuteMol",
    author='Naveen Michaud-Agrawal',
    license='GPL 2',
    url='https://github.com/MDAnalysis/pyQuteMol',
    packages=['Qutemol', 'Qutemol.presets'],
    package_dir= {'Qutemol': 'python'},
    ext_package='Qutemol',
    ext_modules=cythonize(extensions),  # Use cythonize to handle the .pyx file
    #cmdclass = {'build_ext': build_ext}
)
