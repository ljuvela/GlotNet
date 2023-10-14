# GlotNet

This repository implements 
  - Autoregressive WaveNet in PyTorch
  - C++ extensions with PyTorch bindings for fast inference
  - Various signal processing functions for linear autoregressive modeling in GlotNet


## Installation

Installing and testing the package requires building the C++ extensions. The build is triggered by running `pip install .`. This will take a few minutes, please be patient.

Pre-built packages for pip are work in progress, but not active right now. If you would like to contribute, please see the Issues page.

The following commands should get you into working state on most systems (tested on Linux and Mac)
```
# Create a conda environment
conda create -n glotnet python=3.10
conda activate glotnet

# Install requirements in with conda
conda install -c pytorch -c conda-forge pytorch torchaudio tensorboard scikit-build pysoundfile cmake eigen ninja pytest

# Clone git submodules
git submodule update --init --recursive

# Build extensions and install
pip install -v .

# Run pytest unit tests to check everthing works correctly
pytest test
```

### Building for development

Add the `-e` flag in the pip build command for editable installation
```bash
pip install -v -e .
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
pip install -v -e .
pytest test
```

A bash terminal is then opened in the container to interact with the repository.

