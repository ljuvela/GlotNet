# TODO

## Build system (done)

- [x] Replace torchaudio file I/O with soundfile (`glotnet/data/audio_io.py`)
- [x] Conda env: conda-forge tooling + PyPI torch (`environment.yml`, `environment-cpu.yml`)
- [x] Pin pthreads OpenBLAS to avoid duplicate libomp with torch wheels
- [x] Switch scikit-build -> scikit-build-core; editable installs build the extension
- [x] Use pybind11 headers bundled with torch; drop pybind11 submodule
- [x] Use Eigen submodule headers directly (no Eigen CMake project)
- [ ] Remove pybind11 submodule from git (`git submodule deinit` / `git rm`)

## C++ core / Python bindings split

Goal: C++ code usable from CMake-based audio plugins without linking libtorch.

- [x] Remove torch dependency from `ext/src/`: setters take
      `glotnet::ConstArrayView` (`ext/src/ArrayView.h`, pointer + shape in
      torch layout); shapes validated in core (`std::invalid_argument`)
- [x] Tensor -> view conversion in `ext/bind/TensorView.h` (`glotnet::view`)
- [x] CMake: torch-free `glotnet_core` static library (alias `GlotNet::core`)
- [x] CMake: Python module links `glotnet_core` + torch
- [x] CMake option `GLOTNET_BUILD_PYTHON` (ON by default)
- [ ] Namespaces: `ConvolutionLayer`, `ConvolutionStack`, `WaveNet` are in the
      global namespace, and `ConvolutionLayer.h` has `using namespace glotnet;`
- [ ] Separate public headers (e.g. `include/glotnet/`) from sources
- [ ] Only set `CMAKE_CXX_STANDARD` when top-level (`PROJECT_IS_TOP_LEVEL`)
- [ ] Make the project consumable via `add_subdirectory` / `FetchContent`
- [ ] C++ unit tests for the core library (independent of pytest)

## Dependencies

- [ ] torchaudio is in maintenance mode; still used for `functional.lfilter`
      (tests) and mel spectrogram transforms (`glotnet/sigproc/melspec.py`).
      Consider replacing to drop the dependency.

## Docker

- [ ] Update base image (`nvcr.io/nvidia/pytorch:22.11-py3`, torch 1.13)
- [ ] Check CMake >= 3.18 in image; verify `pip install --no-build-isolation -e .`
- [ ] Revisit `DOCKER_LIB_PATH` hack in `CMakeLists.txt`

## Packaging / CI

- [x] cibuildwheel config in `pyproject.toml` (Linux x86_64/aarch64 manylinux_2_28,
      macOS arm64, CPython 3.11-3.15); libtorch excluded from wheel repair
- [x] Wheels workflow with trusted publishing (`pypi-cibuildwheel.yml`)
- [x] CI: build + pytest on Linux and macOS (`python-package.yml`); conda workflow removed
- [ ] Configure trusted publishers on test.pypi.org and pypi.org, plus GitHub
      environments `testpypi` and `pypi`
- [ ] First run: Actions > Wheels > Run workflow (publish to TestPyPI), then
      `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple glotnet`
- [ ] Torch is pinned to one minor version (`torch==2.15.*` in pyproject.toml,
      environment*.yml, requirements.txt); bump all together per release.
      Longer term: the libtorch stable ABI, or the torch-free core above,
      could remove the pin
- [ ] Windows wheels (not attempted)
