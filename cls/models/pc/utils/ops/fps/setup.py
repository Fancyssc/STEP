# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='farthest_point_sampling_cuda',
    ext_modules=[
        CUDAExtension('farthest_point_sampling_cuda', [
            'farthest_point_sampling.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

