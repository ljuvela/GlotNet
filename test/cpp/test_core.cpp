// Minimal tests for the torch-free core library (no test framework needed).
// Build with -DGLOTNET_BUILD_TESTS=ON and run with ctest.

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "glotnet/Convolution.h"
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

} // namespace

int main()
{
    testConvolutionMatchesReference();
    testShapeValidation();
    testWaveNetRuns();
    if (failures)
        std::fprintf(stderr, "%d check(s) failed\n", failures);
    else
        std::printf("all tests passed\n");
    return failures ? 1 : 0;
}
