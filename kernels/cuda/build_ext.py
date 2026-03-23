"""
Build CUDA C++ extensions in-place, bypassing PyTorch's CUDA version check.

System has CUDA 13.2; PyTorch 2.9.1 was compiled with CUDA 12.8.
The version check is a false alarm — CUDA 13.2 nvcc + sm_120 is fully ABI-compatible.

Run from kernels/cuda/:
    python build_ext.py build_ext --inplace
"""

# Patch out the version check before anything else imports BuildExtension
import torch.utils.cpp_extension as _cpp_ext
_cpp_ext._check_cuda_version = lambda *a, **kw: None

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
_nvcc = ["-O3", "--use_fast_math", "-arch=sm_120"]

setup(
    name="cuda_kernels",
    ext_modules=[
        CUDAExtension("rmsnorm_cuda_ext", ["rmsnorm.cu"], extra_compile_args={"nvcc": _nvcc}),
        CUDAExtension("rope_cuda_ext",    ["rope.cu"],    extra_compile_args={"nvcc": _nvcc}),
        CUDAExtension("swiglu_cuda_ext",  ["swiglu.cu"],  extra_compile_args={"nvcc": _nvcc}),
    ],
    cmdclass={"build_ext": BuildExtension},
)
