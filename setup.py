from setuptools import find_packages
from skbuild import setup

packages = find_packages('glotnet')
packages = [f'glotnet.{p}' for p in packages]
packages.append('glotnet')

setup(
    name="glotnet",
    version="0.1.2",
    description="",
    author='Lauri Juvela',
    license="",
    packages=packages,
    cmake_install_dir='glotnet',
    python_requires='>=3.7',
)
