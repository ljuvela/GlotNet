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
3. [x] C++: `paramsDim()` and `seed(uint64_t)` on the base class;
       `CategoricalDensity(bits)` with stable log-softmax and inverse-CDF
       sampling (no per-sample allocation); real `setDistribution(name, bits)`
       in `WaveNetAR` and `GlotNetAR`; reject GlotNetAR + sample_after_filtering;
       bindings `sample_categorical`, `set_distribution`
       Notes: bindings no longer reset the distribution to Gaussian in
       `set_parameters`; Python `use_extension=True` seeds the C++ sampler from
       torch's RNG (`torch.manual_seed` makes it reproducible); default C++
       seed is `std::random_device` (was `time(0)`)
4. [x] Wiring: Python AR models select the C++ sampler from the distribution
       (`cpp_name`, `num_bits`); `inference()` checks params_dim == output
       channels and seeds the C++ sampler from torch's RNG (Gaussian too);
       `Config.num_bits`; trainer registry with per-class kwargs; Gaussian-only
       logging in `Trainer._log_training_step`; GlotNet trainer rejects
       categorical + `glotnet_sample_after_filtering`.
       Fixed on the way: trainers' `model_ar` did not pass the distribution
       (always default Gaussian); `TrainerGlotNet.model_ar` was rebuilt on
       every access
5. [x] Tests: mu-law vs torchaudio, NLL vs log-softmax, greedy/argmax,
       seeded C++ and Python sampler histograms vs softmax (mutation-checked),
       C++ == Python AR reference at T = 0 (WaveNet, conditional, GlotNet),
       C++ vs Python AR sampling statistics at T = 0.7 (class-index mean/std;
       a wrong C++ temperature gives std ratio ~1.2), trainers fit/generate,
       torch-free sampler checks in `test/cpp/test_core.cpp`
6. [x] Small training run first: categorical 8-bit vs Gaussian baseline on the
       same data; compare NLL in bits/sample and listen.
       Smoke test (one LibriTTS-R utterance, 5000 iters, laptop CPU):
       teacher-forced 4.72 (categorical) vs 5.81 (Gaussian, mass per mu-law bin)
       bits/sample; generation via C++ OK after reload from checkpoint.
       Results: /Users/lauri/DATA/glotnet_smoke/RESULTS.md. Still to do: listen;
       a real multi-speaker run before shipping pretrained models
7. [x] Checkpoints: `Trainer.save` writes {format, version, config, model};
       `Trainer.load` checks model_type/distribution/num_bits against the
       trainer config (clear error instead of a state_dict size mismatch) and
       still accepts plain state dicts; `Trainer.config_from_checkpoint(path)`
       rebuilds the config. C++-side loading is still the separate
       "Model weight loading for plugins" item

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

### Implementation plan (draft, to review)

Current state:
- `GaussianDensity` is the only location-scale distribution (2 params: m, log_s)
- Python GlotNet AR reference (`glotnet.py:235`) and C++ `GlotNetAR.cpp:82`
  shift only the mean by the LPC prediction for `sample_after_filtering`, and
  `setDistribution` hardcodes "only gaussian can be shifted"
- **Existing bug:** `trainer_glotnet.py:163` and `:236` do
  `params = params + p_curr`. `p_curr` has 1 channel, so it broadcasts and
  shifts log_s too. Training and inference disagree for Gaussian - Lauri: good, this needs to be fixed, only the means should shift
  `sample_after_filtering`. Fix this first, in its own commit
- C++ `distributions::create(name, num_bits)` and the binding
  `set_distribution(name, num_bits)` cannot describe a component list
- `evaluate.py` (smoke test) computes the mass per mu-law bin for the Gaussian
  by hand. A `cdf` method on every location-scale distribution would make the
  bits/sample comparison generic

Design decisions (to review):
- [ ] One class `LocationScaleMixture(components)`. `components` is a tuple of
      family names from {'gaussian', 'laplace', 'logistic'}, e.g.
      `('gaussian', 'laplace')`, `('gaussian',) * 4`, `('logistic',) * 10`.
      Config: `distribution='mixture'`, `mixture_components=(...)`.
      Mixed families cost nothing extra: the NLL is a logsumexp over the
      per-component log-probabilities either way - Lauri: good
- [ ] Parameter layout in channel blocks: `[logits(K), locs(K), log_scales(K)]`,
      so `params_dim = 3K` (PixelCNN++ layout). K = 1 needs no logit, but keep
      3K for uniformity. `GaussianDensity` stays as is for old checkpoints.
      Test that a K = 1 Gaussian mixture gives the same NLL - Lauri: good
- [ ] Shared location option, `mixture_shared_loc=True`: one location and K
      scales, so `params_dim = 2K + 1`. For Gaussian + Laplace this is literally
      "L2 + L1 around the same prediction", which is the interpretation in the
      motivation above. Independent locations are the general case.
      Proposal: implement both, since only the slicing differs - Lauri: it's a special case, but connected to my hypothesis, let's keep this
- [ ] Discretization: optional `discretize_bits` (default None = continuous
      density). If set, the likelihood is the CDF mass of a linear bin of width
      2/(2^bits - 1), and the edge bins absorb the tails at ±1. This works for
      every family with a closed-form CDF: Gaussian (erf), Laplace, logistic
      (sigmoid). "Discretized logistic mixture" = `('logistic',)*10` + 
      `discretize_bits=16`, as in Parallel WaveNet. Discretization bounds the
      likelihood, so the entropy floor matters less. Keep it for continuous use
- [ ] Entropy floor: the same hinge on every component's log_scale. An unused
      component can collapse otherwise (its weight goes to 0 and its scale
      shrinks freely). The log scale floor stays in C++ sampling - Lauri: we could connect this to the number of bits in the original signal, then the entropy floor has a clear interpretation and units: floor is the entropy of uniform categorical distribution at 16 bit bins (for 16bit signed integer)
- [ ] Temperature: scales every component's scale (as the Gaussian does) and
      leaves the mixture weights alone. T = 0 returns the location of the most
      probable component. That is the mode only approximately, which is fine
      as a greedy path. Optional later: sharpen the weights too
      (logits / T_w), or set the Laplace weight from voicing at inference - Lauri: I haven't tried weight sharpening before, keep in todo for later
- [ ] GlotNet `sample_after_filtering`: shifting all locations by the LPC
      prediction is exact for location-scale mixtures (unlike categorical), so
      the mixture supports it. Replace the name check with a
      `shiftLocation(params, offset)` virtual in C++. It returns false or throws
      for categorical. Python gets a matching `shift_location(params, p)` used
      by the trainer and the AR reference. That also fixes the bug above.
      With discretization the bins follow the signal x, not e - Lauri: good
- [ ] Output layer init: identical-family components start symmetric. Spread
      the initial location/scale biases (e.g. log scales over [-5, -2]) so
      they do not stay identical. Gaussian + Laplace differ by family, but
      spread them too: - Lauri: any components in the signal are likely zero-mean, but spreading the scales is good. Group initialisations by family (no reason to have Gaussian or Laplacian components start with higher scales than the other)

Implementation steps (one commit each):
0. [x] Fix the `params + p_curr` bug in `TrainerGlotNet` (shift channel 0
       only). Done with `GaussianDensity.shift_location(params, offset)`, used
       in `fit` and `validate`. The mixture gets the same method in step 2.
       Regression test (`test_trainer_glotnet.py`, fit and validate): hooks on
       the model and the criterion check that the mean moves by exactly
       `p_curr` and `log_s` is unchanged. It fails on the old code.
       Checkpoints trained with Gaussian `sample_after_filtering` before this
       fix learned a log_s offset of -p and should be retrained
1. [x] `glotnet/losses/location_scale.py`: per-family `log_prob`, `cdf`,
       `sample` (inverse CDF: Laplace `m - b*sign(u)*log1p(-2|u|)`, logistic
       `m + s*(log u - log1p(-u))`, Gaussian `randn`). Tests against
       `torch.distributions.{Normal,Laplace}` and a logistic `TransformedDistribution`
       Also in this step:
       - Stable `log_cdf` (Gaussian `log_ndtr`)
       - `log_mass(lo, hi)` for discretized likelihoods: open edge bins,
         reflection into the left tail, and the midpoint rule below 0.01
         scale units. Gradients stay finite at ±inf edges
       - `entropy` / `log_scale_from_entropy`, and
         `uniform_bin_entropy(num_bits)` = log(2 / 2^bits), for the
         bit-depth entropy floor (step 2)
       Tests are in `test/test_distribution/test_location_scale.py` and were
       mutation-checked. Note: `torch.special.ndtr` / `Normal.cdf` lose
       precision in the left tail even in float64, so the reference uses `erfc`
2. [x] Python `LocationScaleMixture`: `nll` (log_softmax + logsumexp, with the
       optional discretized branch using log-CDF differences and stable tails
       as in PixelCNN++), entropy-floor hinge, `sample`, `cdf`,
       `shift_location`, `log_stats()` (mean weight and min log-scale per
       component, for TensorBoard). `use_extension=True` raises until step 3.
       Tests: against `MixtureSameFamily` for same-family mixtures and by hand
       for mixed families, K=1 == GaussianDensity, discretized mass sums to 1
       over all bins, sample histograms vs the density, gradcheck
       As implemented (`glotnet/losses/distributions.py`):
       - Components are evaluated per family group (one call per family),
         then reordered
       - `log_prob`, `log_mass(lo, hi)`, `cdf`, `bin_edges`, `split`
       - Discretization: 2^bits levels on [-1, 1] (step 2/(2^bits - 1)), open
         edge bins, targets beyond ±1 go to the edge bins. Samples are
         clipped and snapped to the levels
       - Entropy floor: `entropy_floor_bits=16` (None disables it). Each
         family's log-scale floor has the entropy of one bin; the hinge is
         relu(floor - log_s)^2
       - Sampling: the weights are not tempered. T scales the chosen
         component's scale, and T = 0 gives the location of the argmax
         component
       - `initial_bias()`: zero logits and locations. The log scales are
         spread within each family over the same entropies (Gaussian
         log-scale range [-5, -2]); a single component takes the middle.
         Not applied to the model yet (step 4)
       - Not done yet: `log_stats()` for Gaussian/categorical, and the Python
         GlotNet AR reference still shifts channel 0 in place (step 4)
       Tests are in `test/test_distribution/test_mixture.py` and were
       mutation-checked. They include a small maximum-likelihood fit: a
       shared-location Gaussian + Laplace recovers its weight and both scales
3. [x] C++ `LocationScaleMixture` in `Distributions.h/.cpp`. Component enum,
       preallocated weight buffer, inverse-CDF component choice (as in
       Categorical), then a per-family draw. `paramsDim()` = 3K or 2K+1.
       Factory: replace `create(name, num_bits)` with
       `create(const DistributionSpec&)`, where
       `struct DistributionSpec { name; num_bits; components; shared_loc; }`.
       Keep a `(name, num_bits)` overload. Add the `shiftLocation` virtual and
       use it in `GlotNetAR` instead of `x_dist[0] += p_curr`. Bindings:
       `sample_mixture`, and `set_distribution(name, num_bits=8,
       components=[], shared_loc=False)`. Torch-free checks in
       `test/cpp/test_core.cpp`
       As implemented:
       - `Family` enum + `familyFromName`
       - `LocationScaleMixture(components, shared_loc, discretize_bits = 0,
         log_scale_floor = -14)`, matching the Python class: untempered
         component choice; T scales the scale; T <= 0 gives the location of
         the argmax component; discretized samples are clipped and snapped
         (`nearbyint`, half to even like `torch.round`)
       - `DistributionSpec { name, num_bits, components, shared_loc,
         discretize_bits }`, with the `(name, num_bits)` overloads kept in
         `create` and both `setDistribution`s
       - `canShiftLocation()` / `shiftLocation()` on the base class
         (Gaussian, mixture). `GlotNetAR` checks it instead of the name
       - **Fixed on the way:** `GlotNetAR::processConditional` ignored
         `sample_after_filtering` (always sampled e). That is invisible for
         continuous location-scale distributions (x = p + e either way) but
         wrong for discretized ones. Now it is the same as `process`
       - Bindings: `set_distribution(name, num_bits=8, components=[],
         shared_loc=False, discretize_bits=0)` and `sample_mixture(params,
         temperature, components, shared_loc, discretize_bits, seed)`. The
         Python `LocationScaleMixture.sample(use_extension=True)` uses it,
         seeded from torch's RNG
       - Tests: C++ (component frequencies, KS vs the mixture CDF at T = 1
         and 0.5 with interleaved families, greedy, shared location, seeding,
         grid and clipping, shiftLocation, factory errors, AR models;
         discretized GlotNet with sample_after_filtering gives x on the grid
         in `process` and `processConditional`) and Python (C++ sampler KS
         tests, greedy == Python, discretized grid, seeding, per-timestep T).
         Mutation-checked
4. [ ] Wiring:
       - Python distributions expose `cpp_spec()` (a dict) instead of
         `cpp_name`/`num_bits` special-casing in `set_cpp_distribution`
       - Config: `mixture_components`, `mixture_shared_loc`, `discretize_bits`
       - Trainer registry entry
       - `CHECKPOINT_MODEL_FIELDS` += the new fields
       - `_log_training_step` uses `criterion.log_stats()`. Gaussian and
         categorical implement it too, which removes the isinstance check
       - GlotNet trainer allows `sample_after_filtering` for location-scale
         distributions
5. [ ] Tests:
       - C++ == Python AR reference at T = 0 (WaveNet; GlotNet with and
         without sample_after_filtering)
       - Sampling statistics at T = 0.7 (as for categorical)
       - Trainer fit/generate
       - Checkpoint round trip with components
6. [ ] Smoke test on the same utterance as `glotnet_smoke`, scoring everything
       in bits per mu-law sample through `cdf`. Generalise `evaluate.py` to
       any distribution with a `cdf`. Candidates:
       - Gaussian (baseline)
       - Gaussian + Laplace, independent locations
       - Gaussian + Laplace, shared location
       - 4 Gaussians
       - 10 discretized logistics (16 bit)
       Science check for the motivation: on GlotNet (excitation), plot the
       Laplace component's mean responsibility per frame against the voicing
       decision. The hypothesis predicts a high responsibility in voiced
       frames and a low one in unvoiced frames

References (verify venues/years before citing):
- Bishop (1994), Mixture Density Networks, Aston University tech. report NCRG/94/004
- Graves (2013), Generating Sequences With Recurrent Neural Networks, arXiv:1308.0850
- Salimans et al. (2017), PixelCNN++, ICLR (discretized logistic mixture)
- van den Oord et al. (2018), Parallel WaveNet: Fast High-Fidelity Speech
  Synthesis, ICML (10-component discretized logistic mixture, 16-bit audio)
- Ping et al. (2019), ClariNet, ICLR (single Gaussian; notes on mixture
  instability and the log-scale floor)
- Gazor & Zhang (2003), Speech probability distribution, IEEE Signal
  Processing Letters (Laplacian / Gamma fit of speech samples)
- Tukey (1960), A survey of sampling from contaminated distributions
  (Gaussian mixtures as robust error models); Huber (1964), Robust
  estimation of a location parameter, Ann. Math. Stat. (L2/L1 hybrid loss)
- Novelty claim: do a literature search before writing it up. Look at
  Laplace/Gaussian-Laplace output distributions in neural vocoders and
  LPCNet-family excitation models

## Performance bottlenecks

- Identify performance bottlenecks in C++ inference code and proposed improvements. 
- Create benchmark for measuring performance; these are not part of the unit test suite
- Compare pairwise and keep the best implementation; this can be iterated in stages

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
