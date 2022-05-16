# GlotNet

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

Build
```bash
python setup.py install
```

Run tests

```bash
pytest test
```

### Development

```bash
python setup.py develop
```

Flake8 linter tests must pass. Install by
```bash
conda install flake8
```

```bash
flake8
```

### Acknowledgements


### Tests