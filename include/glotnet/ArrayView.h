#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace glotnet
{

/// Non-owning view of a contiguous, row-major float array.
///
/// Used to pass parameters into the core library without depending on a
/// tensor library. The memory layout follows PyTorch conventions, e.g.
/// Conv1d weights are (out_channels, in_channels, filter_width).
/// The viewed data must stay alive for the duration of the call it is
/// passed to; the core copies what it needs.
struct ConstArrayView
{
    const float *data = nullptr;
    std::vector<int64_t> shape;

    ConstArrayView() = default;
    ConstArrayView(const float *data, std::vector<int64_t> shape)
        : data(data), shape(std::move(shape)) {}

    size_t ndim() const { return shape.size(); }

    int64_t numel() const
    {
        int64_t n = 1;
        for (auto s : shape)
            n *= s;
        return n;
    }

    /// Throw std::invalid_argument unless the shape equals `expected`.
    void checkShape(std::initializer_list<int64_t> expected, const char *name) const
    {
        if (data == nullptr || shape != std::vector<int64_t>(expected))
            throw std::invalid_argument(
                std::string(name) + ": expected shape " + shapeString(expected) +
                ", got " + shapeString(shape) + (data ? "" : " (null data)"));
    }

private:
    template <typename T>
    static std::string shapeString(const T &s)
    {
        std::string out = "(";
        bool first = true;
        for (auto v : s)
        {
            if (!first)
                out += ", ";
            out += std::to_string(v);
            first = false;
        }
        return out + ")";
    }
};

} // namespace glotnet
