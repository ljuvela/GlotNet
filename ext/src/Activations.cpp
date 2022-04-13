#include <cassert>
#include "Activations.h"
#include <iostream>
namespace {
    typedef float (* activationFunction)(float x);

    template <activationFunction activation>
    void applyActivation(float **data, size_t rows, size_t cols)
    {
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
                data[i][j] = activation(data[i][j]);
        }
    }
    
    template <activationFunction activation>
    void applyActivation(float *data, size_t rows, size_t cols)
    {
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = activation(data[i]);
        }
    }
    
    inline size_t idx(size_t row, size_t col, size_t cols)
    {
        return row * cols + col;
    }
    
    typedef float (* gatedActivationFunction)(float x1, float x2);
    template<gatedActivationFunction activation>
    void applyGatedActivation(float *data, size_t rows, size_t cols)
    { // TODO rename to time, channels
        // const size_t rowsHalf = rows / 2;
        // for (size_t row = 0; row < rowsHalf; ++row)
        // {
        //     const size_t startIdx1 = idx(row, 0, cols);
        //     const size_t startIdx2 = idx(row + rowsHalf, 0, cols);
        //     for (size_t col = 0; col < cols; ++col)
        //         data[startIdx1+col] = activation(data[startIdx1 + col], data[startIdx2 + col]);
        // }

        // std::cout << "Gated activation: rows " << rows << ", cols " << cols << std::endl;
        const size_t channels = cols;
        const size_t timesteps = rows; 
        const size_t channels_half = channels / 2;
        for (size_t t = 0; t < timesteps; t++)
        {
            for (size_t c = 0; c < channels_half; c++)
            {
                const float f_in = data[t * channels + c];
                const float g_in = data[t * channels + channels_half + c];
                // std::cout << "f_in " << f_in << ", g_in " << g_in << std::endl;
                data[t * channels + c] = activation(f_in, g_in);
            }
        } 
    }
}

namespace Activations {
    
    inline float tanh(float x)
    {
        return tanhf(x);
    }
    
    inline float sigmoid(float x)
    {
        return 1.0f / (1.0f + expf(-x));
    }
    
    inline float relu(float x)
    {
        if (x < 0.0f)
            return 0.0f;
        else
            return x;
    }
    
    inline float softsign(float x)
    {
        return x / (1.0f + fabsf(x));
    }
    
    inline float linear(float x)
    {
        return x;
    }
    
    inline float gated(float x1, float x2)
    {
        return tanh(x1)*sigmoid(x2);
    }
    
    inline float softgated(float x1, float x2)
    {
        return softsign(x1) * softsign(x2);
    }
    
    void tanh(float** data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)tanh>(data, rows, cols);
    }
    void sigmoid(float** data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)sigmoid>(data, rows, cols);
    }
    void relu(float** data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)relu>(data, rows, cols);
    }
    void softsign(float** data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)softsign>(data, rows, cols);
    }
    void linear(float** data, size_t rows, size_t cols)
    {
        return;
    }
    
    void tanh(float* data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)tanh>(data, rows, cols);
    }
    void sigmoid(float* data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)sigmoid>(data, rows, cols);
    }
    void relu(float* data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)relu>(data, rows, cols);
    }
    void softsign(float* data, size_t rows, size_t cols)
    {
        applyActivation<(activationFunction)softsign>(data, rows, cols);
    }
    void linear(float* data, size_t rows, size_t cols)
    {
        return;
    }
    void gated(float* data, size_t rows, size_t cols)
    {
        assert(rows % 2 == 0);
        applyGatedActivation<(gatedActivationFunction)gated>(data, rows, cols);
    }
    void softgated(float* data, size_t rows, size_t cols)
    {
        assert(rows % 2 == 0);
        applyGatedActivation<(gatedActivationFunction)softgated>(data, rows, cols);
    }
    
    bool isGated(std::string name)
    {
        if ((name == "gated") || (name == "softgated"))
            return true;
        return false;
    }
    
    activationFunction getActivationFunction(std::string name)
    {
        if (name == "tanh")
            return tanh;
        else if (name == "sigmoid")
            return sigmoid;
        else if (name == "relu")
            return relu;
        else if (name == "softsign")
            return softsign;
        else if (name == "linear")
            return linear;
        else
            throw std::invalid_argument("Received unkown activation name.");
    }
    
    activationFuncArray getActivationFuncArray(std::string name)
    {
        if (name == "tanh")
            return tanh;
        else if (name == "sigmoid")
            return sigmoid;
        else if (name == "relu")
            return relu;
        else if (name == "softsign")
            return softsign;
        else if (name == "linear")
            return linear;
        else if (name == "gated")
            return gated;
        else if (name == "softgated")
            return softgated;
        else
            throw std::invalid_argument("Received unkown activation name.");
    }
    
    activationFunc2DArray getActivationFunc2DArray(std::string name)
    {
        if (name == "tanh")
            return tanh;
        else if (name == "sigmoid")
            return sigmoid;
        else if (name == "relu")
            return relu;
        else if (name == "softsign")
            return softsign;
        else if (name == "linear")
            return linear;
        else
            throw std::invalid_argument("Received unknown activation name.");
    }
}
