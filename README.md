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
pip install -v e .
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

### Acknowledgements


### Tests