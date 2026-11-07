# GlotNet

This repository implements 
  - Autoregressive WaveNet in PyTorch
  - C++ extensions with PyTorch bindings for fast inference
  - Various signal processing functions for linear autoregressive modeling in GlotNet


## Installation

Pre-built wheels (Linux x86_64/aarch64, macOS arm64, Python 3.11+) are published to PyPI:
```bash
pip install glotnet
```
The C++ extension is compiled against a specific PyTorch version, so each GlotNet release requires a matching `torch` minor version (currently `torch==2.15.*`).

### Building from source

Installing from source requires building the C++ extensions. The extensions are built with CMake, driven by [scikit-build-core](https://scikit-build-core.readthedocs.io) when running `pip install`. A C++17 compiler is required (on macOS, the Xcode Command Line Tools: `xcode-select --install`).

The following commands should get you into working state on most systems (tested on Linux and Mac)
```bash
# Create a conda environment (Python and build tools from conda-forge, PyTorch from PyPI)
conda env create -n glotnet -f environment.yml   # CPU-only Linux: environment-cpu.yml
conda activate glotnet

# Clone git submodules (Eigen)
git submodule update --init --recursive

# Build extensions and install (editable)
pip install -v --no-build-isolation -e .

# Run pytest unit tests to check everthing works correctly
pytest test
```

`--no-build-isolation` is required so that the extension is compiled against the torch version installed in the environment. Without it, pip builds against the latest torch from PyPI, which may not match the installed version.

With the editable install (`-e`), Python changes take effect immediately. After changing C++ code, rerun the `pip install` command; the CMake build directory `build/` is reused, so only changed files are recompiled.

Note: avoid installing conda packages that depend on `pytorch` into the environment, as they would install a second copy of torch next to the PyPI one.

### Using the C++ library from CMake

The inference code (`include/glotnet/`, `ext/src/`) is plain C++17 with Eigen and has no torch or Python dependency, so it can be used in e.g. audio plugins:
```cmake
include(FetchContent)
FetchContent_Declare(glotnet
    GIT_REPOSITORY https://github.com/ljuvela/GlotNet.git
    GIT_TAG master)
FetchContent_MakeAvailable(glotnet)
target_link_libraries(my_plugin PRIVATE GlotNet::core)
```
As a subproject only the core library is built (`GLOTNET_BUILD_PYTHON` defaults to `OFF`). Weights are passed as `glotnet::ConstArrayView` (pointer + shape, PyTorch layout). See `test/cpp/consumer/` for a complete example. To build and run the C++ tests standalone:
```bash
cmake -S . -B build-core -DGLOTNET_BUILD_PYTHON=OFF -DGLOTNET_BUILD_TESTS=ON
cmake --build build-core && ctest --test-dir build-core
```

### Docker

Docker can also be used to build an image with the requried packages.

Build docker container
```bash
docker build -t glotnet_docker .
```
The docker container can be run using the command in `run_docker.sh`. This creates an interactive container using the selected number of GPU devices and mounts the directory containing the repository and a dataset.

When the container is initialised with `run_docker.sh`, we call

```bash
git submodule update --recursive --init
pip install -v --no-build-isolation -e .
pytest test
```

A bash terminal is then opened in the container to interact with the repository.

