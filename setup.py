from setuptools import find_packages
from skbuild import setup

setup(
    name="glotnet",
    version="0.1.0",
    description="",
    author='Lauri Juvela',
    license="",
    packages=['glotnet'],
    package_dir={'glotnet': 'src/glotnet'},
    cmake_install_dir='glotnet',
    python_requires='>=3.7',
)
