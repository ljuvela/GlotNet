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
      `glotnet::ConstArrayView` (`include/glotnet/ArrayView.h`, pointer + shape in
      torch layout); shapes validated in core (`std::invalid_argument`)
- [x] Tensor -> view conversion in `ext/bind/TensorView.h` (`glotnet::view`)
- [x] CMake: torch-free `glotnet_core` static library (alias `GlotNet::core`)
- [x] CMake: Python module links `glotnet_core` + torch
- [x] CMake option `GLOTNET_BUILD_PYTHON` (ON by default)
- [x] Namespaces: all core classes (and `Activations`) in `namespace glotnet`;
      no using-directives in headers
- [x] Only set `CMAKE_CXX_STANDARD` when built standalone
- [x] C++ tests (`test/cpp/test_core.cpp`, `-DGLOTNET_BUILD_TESTS=ON`, ctest);
      CI job builds the core without Python/torch
- [x] Public headers in `include/glotnet/` (`#include "glotnet/WaveNet.h"`)
- [x] Consumable via `add_subdirectory` / `FetchContent`: Python module off by
      default as a subproject; parent's `Eigen3::Eigen` used if present;
      example/CI check in `test/cpp/consumer/`
- [ ] `ext/src/LFilter.h` declares `AllPoleFilter` with no implementation
      and no users: implement or delete
- [ ] `install()` + exported CMake package config (`find_package(GlotNet)`),
      if needed beyond FetchContent
- [ ] Model weight loading for plugins (currently only via setters; e.g. a
      file format exported from the Python side)
- [ ] Clean up core API: mixed `int`/`size_t`/`int64_t` sizes, activation
      names as strings (enum), `std::unique_ptr<Distribution>(new ...)`

## Categorical distribution with mu-law companding

Goal: classic 8-bit mu-law categorical output (WaveNet / LPCNet style) as an
alternative to the Gaussian, for the pretrained models shipped with the release.

Current state:
- Inference always goes through C++; `WaveNetAR`/`GlotNetAR` hardcode
  `GaussianDensity`, and `WaveNetAR::setDistribution(name)` ignores its argument - Lauri: yes this need to be parametric
- C++ `Distribution::sample` has no notion of parameter count (Gaussian assumes 2)
- Trainer constructs every distribution with `entropy_floor=` (`trainer.py:70`,
  registry at `:426`); `Identity()` calls `super().__init__()` without args - Lauri: entropy floor is not needed for categorical distribution, only for location-scale family distributions to avoid div-by-zero when the model gets confident
- `GaussianDensity::setVarianceShape` has unused mu-law-like code
- C++ RNG is seeded with `time(0)`: no reproducible sampler tests - Lauri: find a way to set seed during testing

Design decisions (to review):
- [ ] Output head: `2^num_bits` logits (256 for 8 bits, mu = 255); - Lauri: ok
      `params_dim = 256`, `output_channels` already follows from `params_dim`
- [ ] AR feedback: dequantized float value, so `input_channels = 1` and the AR
      loops stay unchanged (one-hot / embedding input as a later option) - Lauri: yes, float is ok, embedding is overkill
- [ ] Temperature: logits / T; reuses voiced/unvoiced temperature logic
      (LPCNet-style low-probability thresholding as a later option)
- [ ] GlotNet: categorical over the excitation `e`; only
      `sample_after_filtering=false` supported (shifting a categorical by the
      LPC prediction is not meaningful). Check excitation range vs [-1, 1] on
      real data: fixed input gain, or clip and count
- [ ] Quantization noise: 8-bit mu-law noise is audible on clean speech; use
      pre-emphasis (`config.pre_emphasis`, `sigproc/emphasis.py`) as in LPCNet

Implementation steps (one commit each):
1. [x] `glotnet/sigproc/mulaw.py`: `mu_law_encode`, `mu_law_decode`,
       `quantize`, `dequantize` in plain torch (no torchaudio); header-only
       C++ counterpart `include/glotnet/MuLaw.h`
2. [x] Python `CategoricalDensity(Distribution)`: `nll` = cross entropy on
       quantized mu-law targets; `sample` = softmax(logits / T) -> categorical
       -> dequantize -> decode; T = 0 is argmax; fixed `Identity()`.
       `use_extension=True` raises until step 3. Trainer kwargs/registry
       moved to step 4 (C++ inference would misread logits as Gaussian)
3. [ ] C++: `paramsDim()` and `seed(uint64_t)` on the base class;
       `CategoricalDensity(bits)` with stable log-softmax and inverse-CDF
       sampling (no per-sample allocation); real `setDistribution(name, bits)`
       in `WaveNetAR` and `GlotNetAR`; reject GlotNetAR + sample_after_filtering;
       bindings `sample_categorical`, `set_distribution`
4. [ ] Wiring: Python AR models pass the distribution to `_impl`; `Config`
       gets `distribution: 'categorical'`, `num_bits: 8`; trainer registry with
       per-class constructor kwargs (no entropy floor for categorical);
       trainer logging of `batch_penalty`/`batch_log_scale` only for Gaussian
5. [ ] Tests: mu-law round trip and agreement with torchaudio (reference only);
       `nll` == cross entropy; T -> 0 gives argmax; seeded C++ sampler histogram
       vs softmax (chi-square, N ~ 1e5); C++ vs Python sampler statistics;
       end-to-end `WaveNetAR(categorical).inference` shape/range;
       torch-free categorical check in `test/cpp/test_core.cpp`
6. [ ] Small training run first: categorical 8-bit vs Gaussian baseline on the
       same data; compare NLL in bits/sample and listen
7. [ ] Checkpoints: store distribution name and `num_bits` in the saved
       config (relates to "Model weight loading for plugins" above)

References (verify venues/years before citing):
- ITU-T Rec. G.711 (1972/1988), Pulse code modulation (PCM) of voice
  frequencies (mu-law companding)
- van den Oord et al. (2016), WaveNet: A Generative Model for Raw Audio,
  arXiv:1609.03499 (256-way softmax over mu-law samples)
- Tamamori et al. (2017), Speaker-Dependent WaveNet Vocoder, Interspeech
- Valin & Skoglund (2019), LPCNet: Improving Neural Speech Synthesis Through
  Linear Prediction, ICASSP (categorical excitation + LPC, pre-emphasis,
  modified sampling)
- Kalchbrenner et al. (2018), Efficient Neural Audio Synthesis (WaveRNN), ICML
- Jin et al. (2018), FFTNet: A Real-Time Speaker-Dependent Neural Vocoder, ICASSP
- Salimans et al. (2017), PixelCNN++, ICLR (discretized mixture of logistics)
- Ping et al. (2019), ClariNet, ICLR (single Gaussian AR WaveNet)
- Juvela et al. (2019), GlotNet - A Raw Waveform Model for the Glottal
  Excitation in Statistical Parametric Speech Synthesis, IEEE/ACM TASLP

## Pretrained models

- [ ] Choose datasets and check licences before shipping weights
      (LJSpeech: public domain; VCTK, LibriTTS: CC BY 4.0, attribution required)
- [ ] Decide where weights live (release assets, HF Hub, ...) and how they are
      loaded from Python and from C++



## Mixture distributions

I want support for mixture distributions from the location-scale family

- Mixture of Gaussians
- Discretized Logistic Mixture Likelihood
- Misture of Laplacians

Especially interesting is is a mixture of one Gaussian and one Laplacian, which combines L2 and L1 residual minimisation. This is novel in the WaveNet/GlotNet context, and also matches well with speech science: unvoiced residual is close to Gaussian, voiced residual is close to Laplacian. Both are continous, so reshaping the sampling with all-pole filters is also possible.

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
