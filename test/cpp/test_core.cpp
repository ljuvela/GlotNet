// Minimal tests for the torch-free core library (no test framework needed).
// Build with -DGLOTNET_BUILD_TESTS=ON and run with ctest.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "glotnet/Convolution.h"
#include "glotnet/MuLaw.h"
#include "glotnet/Distributions.h"
#include "glotnet/WaveNetAR.h"
#include "glotnet/GlotNetAR.h"
#include "glotnet/WaveNet.h"

namespace
{

int failures = 0;

#define CHECK(cond)                                                    \
    do                                                                 \
    {                                                                  \
        if (!(cond))                                                   \
        {                                                              \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n",          \
                         __FILE__, __LINE__, #cond);                   \
            ++failures;                                                \
        }                                                              \
    } while (0)

std::vector<float> randn(size_t n, std::mt19937 &rng)
{
    std::normal_distribution<float> dist;
    std::vector<float> v(n);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

// Causal dilated Conv1d, torch layout W (O, I, K), data time-major (T, C).
// Equivalent to torch Conv1d with (K - 1) * dilation zeros padded on the left.
std::vector<float> referenceConv(const std::vector<float> &x, const std::vector<float> &W,
                                 const std::vector<float> &b, int64_t T, int64_t I,
                                 int64_t O, int64_t K, int64_t dilation)
{
    std::vector<float> y(T * O);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t o = 0; o < O; ++o)
        {
            float acc = b[o];
            for (int64_t k = 0; k < K; ++k)
            {
                const int64_t ts = t - (K - 1 - k) * dilation;
                if (ts < 0)
                    continue;
                for (int64_t i = 0; i < I; ++i)
                    acc += W[(o * I + i) * K + k] * x[ts * I + i];
            }
            y[t * O + o] = acc;
        }
    return y;
}

void testConvolutionMatchesReference()
{
    std::mt19937 rng(0);
    const int64_t T = 50, I = 3, O = 4, K = 3, dilation = 2;
    auto x = randn(T * I, rng);
    auto W = randn(O * I * K, rng);
    auto b = randn(O, rng);

    glotnet::Convolution conv(I, O, K, dilation);
    conv.setWeight({W.data(), {O, I, K}});
    conv.setBias({b.data(), {O}});

    std::vector<float> y(T * O);
    conv.process(x.data(), y.data(), T);

    const auto ref = referenceConv(x, W, b, T, I, O, K, dilation);
    float max_err = 0.0f;
    for (size_t n = 0; n < y.size(); ++n)
        max_err = std::fmax(max_err, std::fabs(y[n] - ref[n]));
    CHECK(max_err < 1e-5f);
}

void testShapeValidation()
{
    std::vector<float> W(4 * 3 * 4);
    glotnet::Convolution conv(3, 4, 3);

    bool threw = false;
    try
    {
        conv.setWeight({W.data(), {4, 3, 4}}); // wrong filter width
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        conv.setBias({W.data(), {3}}); // wrong number of channels
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);
}

void testWaveNetRuns()
{
    // Smoke test: default-initialised WaveNet processes a block without
    // crashing and produces finite output.
    const int T = 32;
    glotnet::WaveNet wavenet(1, 2, 4, 4, 0, 2, "gated", {1, 2, 4});
    wavenet.prepare(T);
    std::vector<float> x(T, 0.5f), y(T * 2, NAN);
    wavenet.process(x.data(), y.data(), T);
    for (float v : y)
        CHECK(std::isfinite(v));
}

void testMuLaw()
{
    namespace mulaw = glotnet::mulaw;
    const int mu = mulaw::muFromBits(8);
    CHECK(mu == 255);

    // reference codes from glotnet/sigproc/mulaw.py (== torchaudio)
    const float x[] = {-0.9f, -0.1f, -0.001f, 0.0f, 0.002f, 0.3f, 0.75f};
    const int q_ref[] = {2, 52, 122, 128, 137, 228, 248};
    for (int i = 0; i < 7; i++)
        CHECK(mulaw::encode(x[i], mu) == q_ref[i]);

    // every class decodes to a value that encodes back to the same class
    for (int q = 0; q <= mu; q++)
        CHECK(mulaw::encode(mulaw::decode(q, mu), mu) == q);
    CHECK(mulaw::decode(0, mu) == -1.0f);
    CHECK(mulaw::decode(mu, mu) == 1.0f);
    // clipping
    CHECK(mulaw::encode(-3.0f, mu) == 0);
    CHECK(mulaw::encode(3.0f, mu) == mu);
}

void testCategoricalSampler()
{
    namespace dist = glotnet::distributions;
    const int bits = 4;
    dist::CategoricalDensity cat(bits);
    CHECK(cat.paramsDim() == 16);

    std::mt19937 rng(7);
    std::vector<float> logits = randn(16, rng);

    // greedy at zero temperature
    const size_t argmax = std::max_element(logits.begin(), logits.end()) - logits.begin();
    CHECK(cat.sampleIndex(logits.data(), 0.0f) == static_cast<int>(argmax));

    // empirical frequencies match softmax(logits / T)
    const int N = 100000;
    for (float temp : {1.0f, 0.5f})
    {
        cat.seed(123);
        std::vector<int> counts(16, 0);
        for (int n = 0; n < N; n++)
            counts[cat.sampleIndex(logits.data(), temp)]++;

        const float m = *std::max_element(logits.begin(), logits.end());
        double z = 0.0;
        for (float l : logits)
            z += std::exp((l - m) / temp);
        for (int k = 0; k < 16; k++)
        {
            const double p = std::exp((logits[k] - m) / temp) / z;
            const double tol = 5.0 * std::sqrt(p * (1.0 - p) / N) + 1e-4;
            CHECK(std::abs(counts[k] / double(N) - p) < tol);
        }
    }

    // same seed, same sequence; samples lie on the mu-law grid
    const int T = 64;
    std::vector<float> params(T * 16), temp(T, 1.0f), a(T), b(T);
    for (int t = 0; t < T; t++)
        std::copy(logits.begin(), logits.end(), params.begin() + t * 16);
    cat.seed(5);
    cat.sample(params.data(), a.data(), T, temp.data());
    cat.seed(5);
    cat.sample(params.data(), b.data(), T, temp.data());
    CHECK(a == b);
    const int mu = glotnet::mulaw::muFromBits(bits);
    for (float v : a)
        CHECK(glotnet::mulaw::decode(glotnet::mulaw::encode(v, mu), mu) == v);

    bool threw = false;
    try { dist::CategoricalDensity bad(0); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);
}

void testSetDistribution()
{
    const int T = 64;
    std::vector<float> x(T, 0.0f), temp(T, 1.0f), y1(T), y2(T);

    glotnet::WaveNetAR model(1, 256, 4, 4, 0, 2, "gated", {1, 2});
    model.setDistribution("categorical", 8);
    model.setSeed(42);
    model.process(x.data(), temp.data(), y1.data(), T);
    model.setSeed(42);
    model.process(x.data(), temp.data(), y2.data(), T);
    CHECK(y1 == y2);
    for (float v : y1)
        CHECK(std::isfinite(v) && std::abs(v) <= 1.0f);

    auto throws = [](auto f) {
        try { f(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    // parameter count must match the output channels
    CHECK(throws([&] { model.setDistribution("gaussian"); }));
    glotnet::WaveNetAR gauss(1, 2, 4, 4, 0, 2, "gated", {1, 2});
    CHECK(throws([&] { gauss.setDistribution("categorical", 8); }));
    CHECK(!throws([&] { gauss.setDistribution("gaussian"); }));
    CHECK(throws([&] { gauss.setDistribution("laplace"); }));

    // GlotNet: categorical only without sample_after_filtering
    glotnet::GlotNetAR glot_after(3, 256, 4, 4, 0, 2, "gated", {1, 2}, 10, true);
    CHECK(throws([&] { glot_after.setDistribution("categorical", 8); }));
    glotnet::GlotNetAR glot(3, 256, 4, 4, 0, 2, "gated", {1, 2}, 10, false);
    CHECK(!throws([&] { glot.setDistribution("categorical", 8); }));
}

// CDF of a standard family member, for the sampler tests
double familyCdf(glotnet::distributions::Family f, double z)
{
    using glotnet::distributions::Family;
    switch (f)
    {
    case Family::Gaussian: return 0.5 * std::erfc(-z / std::sqrt(2.0));
    case Family::Laplace: return z < 0 ? 0.5 * std::exp(z) : 1.0 - 0.5 * std::exp(-z);
    case Family::Logistic: return 1.0 / (1.0 + std::exp(-z));
    }
    return 0.0;
}

void testMixtureSampler()
{
    namespace dist = glotnet::distributions;
    using F = dist::Family;
    auto throws = [](auto f) {
        try { f(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };

    // interleaved families: each component must draw from its own family
    const std::vector<F> families = {F::Laplace, F::Gaussian, F::Logistic, F::Gaussian};
    const size_t K = families.size();
    dist::LocationScaleMixture mix(families);
    CHECK(mix.paramsDim() == 3 * K);
    CHECK(dist::LocationScaleMixture(families, true).paramsDim() == 2 * K + 1);

    //                        logits                 locs                    log scales
    const std::vector<float> p = {0.5f, -0.3f, 0.1f, 0.2f,  -0.2f, 0.0f, 0.4f, 0.1f,  -1.0f, -2.5f, -1.5f, -0.5f};

    // component frequencies match softmax(logits), never tempered
    {
        const int N = 100000;
        std::vector<int> counts(K, 0);
        mix.seed(11);
        for (int n = 0; n < N; n++)
            counts[mix.sampleComponent(p.data())]++;
        double z = 0.0;
        for (size_t k = 0; k < K; k++)
            z += std::exp(p[k]);
        for (size_t k = 0; k < K; k++)
        {
            const double w = std::exp(p[k]) / z;
            CHECK(std::abs(counts[k] / double(N) - w) < 5.0 * std::sqrt(w * (1 - w) / N));
        }
    }

    // Kolmogorov-Smirnov distance to the mixture CDF, scales multiplied by T
    for (float temp : {1.0f, 0.5f})
    {
        const int N = 100000;
        std::vector<double> x(N);
        mix.seed(3);
        for (auto &v : x)
            v = mix.sampleOne(p.data(), temp);
        std::sort(x.begin(), x.end());
        double z = 0.0;
        for (size_t k = 0; k < K; k++)
            z += std::exp(p[k]);
        double ks = 0.0;
        for (int n = 0; n < N; n++)
        {
            double cdf = 0.0;
            for (size_t k = 0; k < K; k++)
            {
                const double scale = temp * std::exp(p[2 * K + k]);
                cdf += std::exp(p[k]) / z * familyCdf(families[k], (x[n] - p[K + k]) / scale);
            }
            ks = std::max(ks, std::abs(cdf - (n + 1.0) / N));
        }
        CHECK(ks < 2.0 / std::sqrt(double(N)));
    }

    // zero temperature: location of the most probable component (index 0)
    CHECK(mix.sampleOne(p.data(), 0.0f) == p[K + 0]);
    // shared location: params [logits, loc, log_scales]
    dist::LocationScaleMixture shared(families, true);
    std::vector<float> ps = {0.5f, -0.3f, 0.1f, 0.2f,  0.25f,  -1.0f, -2.5f, -1.5f, -0.5f};
    CHECK(shared.sampleOne(ps.data(), 0.0f) == 0.25f);

    // same seed, same sequence (also after a cached normal deviate)
    {
        const int T = 33;
        std::vector<float> params, temp(T, 1.0f), a(T), b(T);
        for (int t = 0; t < T; t++)
            params.insert(params.end(), p.begin(), p.end());
        mix.seed(9);
        mix.sample(params.data(), a.data(), T, temp.data());
        mix.seed(9);
        mix.sample(params.data(), b.data(), T, temp.data());
        CHECK(a == b);
    }

    // discretized: clipped to [-1, 1] and on the 2^bits grid
    {
        const int bits = 4;
        const double n = (1 << bits) - 1;
        dist::LocationScaleMixture disc({F::Gaussian, F::Logistic}, false, bits);
        const std::vector<float> pd = {0.0f, 0.0f,  1.5f, -0.3f,  -1.0f, -2.0f};
        disc.seed(1);
        bool clipped = false;
        for (int i = 0; i < 2000; i++)
        {
            const float v = disc.sampleOne(pd.data(), 1.0f);
            CHECK(std::abs(v) <= 1.0f);
            const double q = (v + 1.0) * n / 2.0;
            CHECK(std::abs(q - std::round(q)) < 1e-4);
            clipped |= (v == 1.0f);
        }
        CHECK(clipped);
    }

    // shiftLocation moves the locations only
    {
        std::vector<float> q = p;
        mix.shiftLocation(q.data(), 0.5f);
        for (size_t i = 0; i < q.size(); i++)
            CHECK(q[i] == ((i >= K && i < 2 * K) ? p[i] + 0.5f : p[i]));
        std::vector<float> qs = ps;
        shared.shiftLocation(qs.data(), -0.25f);
        for (size_t i = 0; i < qs.size(); i++)
            CHECK(qs[i] == (i == K ? 0.0f : ps[i]));
        dist::GaussianDensity gauss;
        float g[2] = {0.1f, -3.0f};
        gauss.shiftLocation(g, 0.2f);
        CHECK(g[0] == 0.1f + 0.2f && g[1] == -3.0f);
        dist::CategoricalDensity cat(4);
        CHECK(!cat.canShiftLocation() && mix.canShiftLocation() && gauss.canShiftLocation());
        bool threw = false;
        try { cat.shiftLocation(g, 1.0f); } catch (const std::logic_error &) { threw = true; }
        CHECK(threw);
    }

    // factory
    dist::DistributionSpec spec;
    spec.name = "mixture";
    spec.components = {"gaussian", "laplace"};
    spec.shared_loc = true;
    CHECK(dist::create(spec)->paramsDim() == 5);
    spec.components = {"gaussian", "cauchy"};
    CHECK(throws([&] { dist::create(spec); }));
    spec.components = {};
    CHECK(throws([&] { dist::create(spec); }));
    spec.components = {"gaussian"};
    spec.discretize_bits = 25;
    CHECK(throws([&] { dist::create(spec); }));
}

void testArMixture()
{
    namespace dist = glotnet::distributions;
    auto throws = [](auto f) {
        try { f(); } catch (const std::invalid_argument &) { return true; }
        return false;
    };
    dist::DistributionSpec spec;
    spec.name = "mixture";
    spec.components = {"gaussian", "laplace"};

    const int T = 64;
    std::vector<float> x(T, 0.0f), temp(T, 1.0f), y1(T), y2(T);
    glotnet::WaveNetAR model(1, 6, 4, 4, 0, 2, "gated", {1, 2});
    CHECK(throws([&] { model.setDistribution("gaussian"); }));
    model.setDistribution(spec);
    model.setSeed(4);
    model.process(x.data(), temp.data(), y1.data(), T);
    model.setSeed(4);
    model.process(x.data(), temp.data(), y2.data(), T);
    CHECK(y1 == y2);
    for (float v : y1)
        CHECK(std::isfinite(v));
    spec.shared_loc = true;
    CHECK(throws([&] { model.setDistribution(spec); })); // 5 != 6 channels
    spec.shared_loc = false;

    // GlotNet: a mixture can be shifted by the LPC prediction
    glotnet::GlotNetAR glot_after(3, 6, 4, 4, 0, 2, "gated", {1, 2}, 2, true);
    CHECK(!throws([&] { glot_after.setDistribution(spec); }));

    // sample_after_filtering with a discretized mixture: the signal x (not
    // the excitation) lies on the grid, also in processConditional
    const int bits = 4;
    const double n = (1 << bits) - 1;
    spec.components = {"gaussian"};
    spec.discretize_bits = bits;
    const int order = 2, cond_channels = 1;
    std::vector<float> a(T * (order + 1)), cond(T * cond_channels, 0.1f), out(T);
    for (int t = 0; t < T; t++)
    {
        a[t * (order + 1) + 0] = 1.0f;
        a[t * (order + 1) + 1] = -0.5f;
        a[t * (order + 1) + 2] = 0.2f;
    }
    auto on_grid = [&](const std::vector<float> &v) {
        int off = 0;
        for (float s : v)
        {
            const double q = (s + 1.0) * n / 2.0;
            off += std::abs(q - std::round(q)) > 1e-4;
        }
        return off;
    };
    for (bool after : {true, false})
    {
        glotnet::GlotNetAR g(3, 3, 4, 4, cond_channels, 2, "gated", {1, 2}, order, after);
        g.setDistribution(spec);
        g.setSeed(2);
        g.process(x.data(), a.data(), temp.data(), out.data(), T);
        const int off_plain = on_grid(out);
        g.setSeed(2);
        g.processConditional(x.data(), a.data(), cond.data(), temp.data(), out.data(), T);
        const int off_cond = on_grid(out);
        if (after)
            CHECK(off_plain == 0 && off_cond == 0);
        else
            CHECK(off_plain > 0 && off_cond > 0); // x = e + p is off the grid
    }
}

} // namespace

int main()
{
    testConvolutionMatchesReference();
    testShapeValidation();
    testWaveNetRuns();
    testMuLaw();
    testCategoricalSampler();
    testSetDistribution();
    testMixtureSampler();
    testArMixture();
    if (failures)
        std::fprintf(stderr, "%d check(s) failed\n", failures);
    else
        std::printf("all tests passed\n");
    return failures ? 1 : 0;
}
