from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='correlation',
    ext_modules=[
        CUDAExtension('alt_cuda_corr',
            sources=['correlation.cpp', 'correlation_kernel.cu'],
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

