import os
from sys import platform
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from glob import glob

sources = ['ext/bindings.cpp']
extra_compiler_args = ['-std=c++17', '-O0', '-w']
extra_compiler_args += ['-march=native'] # clang vectorization (according to Eigen docs)
include_dirs = []

sources += glob('ext/src/*.cpp')
sources += glob('ext/bind/*.cpp')

prefix = os.environ.get('CONDA_PREFIX', None)
if prefix is None:
    prefix = os.environ.get('CONDA', None) # GitHub Actions Conda build
if prefix is None:
    prefix = '/usr/local' # best generic guess for unix


def locate_eigen_headers():
    eigen_headers = os.path.join(prefix, 'include', 'eigen3')
    if os.path.isdir(eigen_headers):
        return eigen_headers
    dir_path = os.path.dirname(os.path.realpath(__file__))
    eigen_headers = os.path.join(dir_path, 'third_party', 'eigen')
    if os.path.isdir(eigen_headers):
        return eigen_headers
    # raise error if not found
    raise ImportError("Eigen headers not found, try either \n"+\
        "1) If using conda, install by 'conda install -c conda-forge eigen'"+\
        "2) If using pip, clone the Eigen submodule by running 'git submodule update --recursive --init'")
    
eigen_headers = locate_eigen_headers()
include_dirs += [eigen_headers]

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
      ext_modules=[
          CppExtension(name='glotnet.cpp_extensions',
                       sources=sources,
                       extra_compile_args=extra_compiler_args,
                       include_dirs=include_dirs)
      ],
      cmdclass={'build_ext': BuildExtension}
      )
