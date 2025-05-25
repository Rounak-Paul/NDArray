#include "ndarray.hpp"
#include <cmath> // for std::exp

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Matrix-vector multiplication
nd::NDArray<float> matvec_dot(const nd::NDArray<float>& mat, const nd::NDArray<float>& vec) {
    if (mat.ndim() != 2 || vec.ndim() != 1 || mat.shape()[1] != vec.shape()[0]) {
        throw std::runtime_error("Shape mismatch in matvec_dot");
    }

    std::vector<float> result(mat.shape()[0], 0.0f);
    for (size_t i = 0; i < mat.shape()[0]; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < mat.shape()[1]; ++j) {
            sum += mat.at({i, j}) * vec.at({j});
        }
        result[i] = sum;
    }
    return nd::NDArray<float>({mat.shape()[0]}, result);
}

// Elementwise addition
nd::NDArray<float> vec_add(const nd::NDArray<float>& a, const nd::NDArray<float>& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("Shape mismatch in vec_add");
    std::vector<float> result(a.shape()[0]);
    for (size_t i = 0; i < a.shape()[0]; ++i) {
        result[i] = a.at({i}) + b.at({i});
    }
    return nd::NDArray<float>({a.shape()[0]}, result);
}

// Elementwise sigmoid activation
nd::NDArray<float> sigmoid_vec(const nd::NDArray<float>& a) {
    std::vector<float> result(a.shape()[0]);
    for (size_t i = 0; i < a.shape()[0]; ++i) {
        result[i] = sigmoid(a.at({i}));
    }
    return nd::NDArray<float>({a.shape()[0]}, result);
}

int main() {
    // Input: 2 features
    nd::NDArray<float> input({2}, {0.5f, -0.3f});

    // Weights for input -> hidden (2x2)
    nd::NDArray<float> w1({2, 2}, {
        0.1f, 0.4f,
        -0.2f, 0.3f
    });

    // Biases for hidden layer
    nd::NDArray<float> b1({2}, {0.01f, -0.02f});

    // Weights for hidden -> output (1x2)
    nd::NDArray<float> w2({1, 2}, {0.7f, -0.5f});

    // Bias for output layer
    nd::NDArray<float> b2({1}, {0.05f});

    // Forward pass
    auto hidden_input = vec_add(matvec_dot(w1, input), b1);
    auto hidden_output = sigmoid_vec(hidden_input);

    auto output_input = vec_add(matvec_dot(w2, hidden_output), b2);
    auto output = sigmoid_vec(output_input);

    std::cout << "Output of ANN: " << output << std::endl;

    return 0;
}