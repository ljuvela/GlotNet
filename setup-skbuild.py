from setuptools import find_packages
from skbuild import setup
packages = find_packages('.', exclude='third_party')

setup(
    name="glotnet",
    version="1.2.3",
    description="",
    author='Lauri Juvela',
    license="",
    packages=packages,
    cmake_install_dir='ext',
    python_requires='>=3.7',
)
