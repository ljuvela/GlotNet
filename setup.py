import os
from sys import platform
from setuptools import setup, Extension, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from glob import glob

sources = ['ext/bindings.cpp']
extra_compiler_args = ['-std=c++17', '-O0', '-w']
extra_compiler_args += ['-march=native'] # clang vectorization (according to Eigen docs)
include_dirs = []

####################### BOF Source files #######################

# sources += glob('ext/src/*.cpp')
sources += ['ext/src/Activations.cpp',
            'ext/src/Convolution.cpp',
            'ext/src/ConvolutionLayer.cpp',
            'ext/src/ConvolutionStack.cpp',
            'ext/src/WaveNet.cpp',
            ]
sources += glob('ext/bind/*.cpp')

eigen_headers = os.path.join(os.environ['CONDA_PREFIX'], 'include', 'eigen3')
include_dirs += [eigen_headers]

####################### EOF Source files #######################

# Prune out duplicate source files
sources = list(set(sources))

if platform == "linux" or platform == "linux2":
    compiler = 'g++'  # Linux
elif platform == "darwin":
    compiler = 'clang++'  # Mac
elif platform == "win32":
    raise NotImplementedError("Compiling GlotNet extensions for Windows is not currently supported")

os.environ['CXX'] = compiler

setup(name='glotnet',
    packages=['glotnet'],
    package_dir={'glotnet': 'python'},
      ext_modules=[
          CppExtension(name='glotnet.cpp_extensions',
                       sources=sources,
                       extra_compile_args=extra_compiler_args,
                       include_dirs=include_dirs)
      ],
      cmdclass={'build_ext': BuildExtension}
      )
