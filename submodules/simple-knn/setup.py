#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

if sys.platform == "win32":
    nvcc_compiler_flags += [
        "-DWIN32_LEAN_AND_MEAN",
        "-allow-unsupported-compiler",
    ]

setup(
    name="simple_knn",
    packages=['simple_knn'],
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
                "spatial.cu",
                "simple_knn.cu",
                "ext.cpp"
            ],
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags})
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)
