from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='raft-alt-cuda-corr',
    ext_modules=[
        CUDAExtension('raft_alt_cuda_corr',
            sources=['correlation.cpp', 'correlation_kernel.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

