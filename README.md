# GlotNet

## Build status

![pip_package](https://github.com/ljuvela/GlotNet/actions/workflows/python-package.yml/badge.svg)
![conda_package](https://github.com/ljuvela/GlotNet/actions/workflows/python-package-conda.yml/badge.svg)

## Dependencies

### Conda environment

Create environment based on `environment.yml`
```bash
conda env create -n glotnet -f environment.yml
conda activate glotnet
```

### Pip package manager

Pip can not handle C++ dependencies, but in this case the only external dependency is the lightweight header-only Eigen library
```bash
git submodule update --recursive --init
pip install -r requirements.txt
```

### Build

```
python setup.py develop -- -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native -O3 -std=c++17"
```

```
python setup.py develop -- -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-march=native -O0 -g -std=c++17"
```

Build and install
```bash
pip install .
```

Run tests

```bash
pytest test
```

### Development

Build in edit mode 
```bash
pip install -v -e .
```

Flake8 linter tests must pass. Install by
```bash
conda install flake8
```
or if using pip
```bash
pip install flake8
```

Run linter by 
```bash
flake8
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

### Acknowledgements


### Tests