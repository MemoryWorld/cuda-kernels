import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

_nvcc_flags = ["-O3", "--use_fast_math", "-arch=sm_120"]

setup(
    name="cuda_kernels",
    ext_modules=[
        CUDAExtension(
            "rmsnorm_cuda_ext",
            ["rmsnorm.cu"],
            extra_compile_args={"nvcc": _nvcc_flags},
        ),
        CUDAExtension(
            "rope_cuda_ext",
            ["rope.cu"],
            extra_compile_args={"nvcc": _nvcc_flags},
        ),
        CUDAExtension(
            "swiglu_cuda_ext",
            ["swiglu.cu"],
            extra_compile_args={"nvcc": _nvcc_flags},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
