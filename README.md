# NDArray - High-Performance N-Dimensional Array Library for C++

A comprehensive, NumPy-inspired N-dimensional array library for C++ with built-in neural network operations, designed for scientific computing, machine learning, and deep learning applications.

## Features

- **N-Dimensional Arrays**: Support for arrays of any dimensionality
- **Memory Efficient**: Shared pointer-based memory management with copy-on-write semantics
- **Broadcasting**: NumPy-style broadcasting for arithmetic operations
- **Neural Network Operations**: Built-in convolution, pooling, activation functions, and more
- **Mathematical Functions**: Comprehensive set of mathematical operations
- **Type Safe**: Template-based design supporting any numeric type
- **Performance Optimized**: Efficient memory layout and vectorized operations
- **NumPy-Like API**: Familiar interface for Python developers

## Quick Start

### Installation

Simply include the header file in your project:

```cpp
#include "ndarray.hpp"
```

### Basic Usage

```cpp
#include "ndarray.hpp"
#include <iostream>

int main() {
    // Create arrays
    NDArray<float> arr1({2, 3}, 5.0f);  // 2x3 array filled with 5.0
    NDArray<float> arr2 = {{1, 2, 3}, {4, 5, 6}};  // Initialize with data
    
    // Basic operations
    auto result = arr1 + arr2;
    std::cout << "Result:\n" << result << std::endl;
    
    // Matrix multiplication
    NDArray<float> a({2, 3});  // 2x3 matrix
    NDArray<float> b({3, 2});  // 3x2 matrix
    auto product = a.dot(b);   // Results in 2x2 matrix
    
    return 0;
}
```

## Detailed Usage

### Array Creation

```cpp
// Different ways to create arrays
NDArray<double> zeros_arr = zeros<double>({3, 4});          // 3x4 zeros
NDArray<double> ones_arr = ones<double>({2, 2});            // 2x2 ones
NDArray<double> range_arr = arange<double>(0, 10, 0.5);     // Range array
NDArray<double> lin_arr = linspace<double>(0, 1, 100);      // Linear space
NDArray<double> eye_arr = eye<double>(3);                   // 3x3 identity matrix
NDArray<double> rand_arr = random<double>({2, 3}, 0, 1);    // Random values [0,1)

// From initializer lists
NDArray<int> arr1d = {1, 2, 3, 4, 5};
NDArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};

// Type aliases for convenience
Array arr_double({3, 3});      // NDArray<double>
ArrayF arr_float({3, 3});      // NDArray<float>
ArrayI arr_int({3, 3});        // NDArray<int>
```

### Array Properties and Access

```cpp
NDArray<float> arr({2, 3, 4});

// Properties
std::cout << "Shape: ";
for (auto dim : arr.shape()) std::cout << dim << " ";
std::cout << "\nDimensions: " << arr.ndim() << std::endl;
std::cout << "Size: " << arr.size() << std::endl;
std::cout << "Empty: " << arr.empty() << std::endl;

// Element access
arr(0, 1, 2) = 5.0f;                    // Multi-dimensional indexing
arr[{0, 1, 2}] = 5.0f;                  // Vector indexing
float val = arr.at({0, 1, 2});          // Bounds-checked access
float flat_val = arr.flat(5);           // Flat indexing
```

### Reshaping and Transposition

```cpp
NDArray<float> arr({2, 6});
auto reshaped = arr.reshape({3, 4});    // Reshape to 3x4
auto transposed = arr.T();              // Transpose (reverse all dimensions)

// For 2D matrices, T() swaps rows and columns
NDArray<float> matrix({3, 4});
auto matrix_T = matrix.T();  // Now 4x3
```

### Arithmetic Operations

```cpp
NDArray<float> a({2, 3});
NDArray<float> b({2, 3});

// Element-wise operations
auto sum = a + b;          // Addition
auto diff = a - b;         // Subtraction  
auto product = a * b;      // Element-wise multiplication
auto quotient = a / b;     // Element-wise division

// Scalar operations
auto scaled = a * 2.0f;    // Multiply by scalar
auto shifted = a + 1.0f;   // Add scalar

// In-place operations
a += b;                    // In-place addition
a *= 2.0f;                 // In-place scalar multiplication
```

### Mathematical Functions

```cpp
NDArray<float> arr = random<float>({3, 3}, -2, 2);

// Element-wise mathematical functions
auto abs_arr = abs(arr);       // Absolute values
auto sqrt_arr = sqrt(arr);     // Square root
auto exp_arr = exp(arr);       // Exponential
auto log_arr = log(arr);       // Natural logarithm
auto sin_arr = sin(arr);       // Sine
auto cos_arr = cos(arr);       // Cosine
auto pow_arr = pow(arr, 2.0f); // Power

// Or using member functions
auto abs_arr2 = arr.abs();
auto exp_arr2 = arr.exp();
```

### Reduction Operations

```cpp
NDArray<float> arr = random<float>({3, 4, 5});

// Global reductions
float total = arr.sum();          // Sum all elements
float average = arr.mean();       // Mean of all elements
float minimum = arr.min();        // Minimum value
float maximum = arr.max();        // Maximum value

// Axis-wise reductions
auto row_sums = arr.sum(0);       // Sum along axis 0
auto col_means = arr.mean(1);     // Mean along axis 1
```

### Linear Algebra

```cpp
// Matrix multiplication
NDArray<float> A({3, 4});
NDArray<float> B({4, 2});
auto C = A.dot(B);                // 3x2 result

// Vector operations
NDArray<float> v1({5});
NDArray<float> v2({5});
auto dot_product = v1.dot(v2);    // Scalar result

// Matrix-vector multiplication
NDArray<float> matrix({3, 4});
NDArray<float> vector({4});
auto result = matrix.dot(vector); // 3-element result
```

### Slicing

```cpp
NDArray<float> arr({4, 5, 6});

// Define slice ranges: {start, end} for each dimension
std::vector<std::pair<size_t, size_t>> ranges = {
    {1, 3},  // Rows 1-2 (end exclusive)
    {0, 5},  // All columns
    {2, 5}   // Depth 2-4
};

auto sliced = arr.slice(ranges);  // Results in 2x5x3 array
```

## Neural Network Operations

### Activation Functions

```cpp
NDArray<float> x = random<float>({100, 10}, -2, 2);

// Activation functions
auto relu_out = x.relu();         // ReLU: max(0, x)
auto sigmoid_out = x.sigmoid();   // Sigmoid: 1/(1+exp(-x))
auto tanh_out = x.tanh();         // Hyperbolic tangent
auto softmax_out = x.softmax(1);  // Softmax along axis 1

// Using free functions
auto relu_out2 = relu(x);
auto sigmoid_out2 = sigmoid(x);
```

### Convolution and Pooling

```cpp
// 2D Convolution
NDArray<float> input({1, 3, 32, 32});     // 1 batch, 3 channels, 32x32 image
NDArray<float> kernel({16, 3, 5, 5});     // 16 filters, 3 input channels, 5x5

auto conv_out = input.conv2d(kernel, 1, 2);  // stride=1, padding=2

// Max Pooling
auto pooled = conv_out.max_pool2d(2, 2);  // 2x2 pool, stride=2
```

### Weight Initialization

```cpp
std::mt19937 rng(42);  // Random number generator

// Xavier/Glorot initialization
auto weights1 = xavier_uniform<float>({784, 128}, rng);

// He initialization (better for ReLU)
auto weights2 = he_uniform<float>({128, 64}, rng);
```

### Loss Functions

```cpp
NDArray<float> predictions({32, 10});  // 32 samples, 10 classes
NDArray<float> targets({32, 10});      // One-hot encoded

// Mean Squared Error
float mse = mean_squared_error(predictions, targets);

// Cross-entropy loss
float ce_loss = cross_entropy_loss(predictions, targets);
```

### Batch Normalization and Dropout

```cpp
NDArray<float> input({32, 64, 28, 28});  // Batch of feature maps
NDArray<float> mean({64});               // Per-channel statistics
NDArray<float> var({64});
NDArray<float> gamma({64});              // Learned parameters
NDArray<float> beta({64});

// Batch normalization (inference mode)
auto normalized = input.batch_norm(mean, var, gamma, beta);

// Dropout (training mode)
std::mt19937 rng(42);
auto dropped = input.dropout(0.5f, rng);  // 50% dropout rate
```

## Complete Neural Network Example

```cpp
#include "ndarray.hpp"
#include <iostream>
#include <random>

class SimpleNN {
private:
    ArrayF W1, b1, W2, b2;
    std::mt19937 rng;
    
public:
    SimpleNN(size_t input_size, size_t hidden_size, size_t output_size) 
        : rng(std::random_device{}()) {
        
        // Initialize weights
        W1 = he_uniform<float>({input_size, hidden_size}, rng);
        b1 = zeros<float>({1, hidden_size});
        W2 = he_uniform<float>({hidden_size, output_size}, rng);
        b2 = zeros<float>({1, output_size});
    }
    
    ArrayF forward(const ArrayF& X) {
        // First layer
        auto z1 = X.dot(W1) + b1;
        auto a1 = relu(z1);
        
        // Output layer
        auto z2 = a1.dot(W2) + b2;
        return softmax(z2, 1);
    }
    
    float compute_loss(const ArrayF& X, const ArrayF& y) {
        auto predictions = forward(X);
        return cross_entropy_loss(predictions, y);
    }
};

int main() {
    // Create model
    SimpleNN model(784, 128, 10);  // MNIST-like: 784->128->10
    
    // Generate dummy data
    auto X = random<float>({32, 784});  // 32 samples, 784 features
    auto y = zeros<float>({32, 10});    // One-hot labels
    
    // Set random labels
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9);
    
    for (size_t i = 0; i < 32; ++i) {
        y[{i, static_cast<size_t>(dis(gen))}] = 1.0f;
    }
    
    // Forward pass
    auto predictions = model.forward(X);
    float loss = model.compute_loss(X, y);
    
    std::cout << "Predictions shape: ";
    for (auto dim : predictions.shape()) std::cout << dim << " ";
    std::cout << "\nLoss: " << loss << std::endl;
    
    return 0;
}
```

## Advanced Features

### Broadcasting

```cpp
NDArray<float> a({3, 1});     // 3x1 array
NDArray<float> b({1, 4});     // 1x4 array
auto result = a + b;          // Broadcasting creates 3x4 result
```

### Memory Sharing

Arrays use shared pointers for efficient memory management:

```cpp
NDArray<float> original({1000, 1000});
NDArray<float> copy = original;           // Shares memory
NDArray<float> reshaped = original.reshape({500, 2000}); // Also shares memory

// Memory is only copied when modified (copy-on-write)
```

### Type Support

The library supports any numeric type:

```cpp
NDArray<double> double_arr({3, 3});
NDArray<float> float_arr({3, 3});
NDArray<int> int_arr({3, 3});
NDArray<std::complex<float>> complex_arr({3, 3});
```

## Performance Tips

1. **Prefer in-place operations** when possible: `arr += other` vs `arr = arr + other`
2. **Use appropriate data types**: `float` for neural networks, `double` for scientific computing
3. **Minimize reshaping**: Plan your array shapes to avoid unnecessary reshaping
4. **Batch operations**: Process multiple samples together for better cache efficiency

## Building and Requirements

### Requirements
- C++11 or later
- Standard library support for `<vector>`, `<memory>`, `<random>`, etc.

### Building
```bash
# Header-only library - just include the header
g++ -std=c++11 -O3 your_program.cpp -o your_program

# For debug builds
g++ -std=c++11 -g -DDEBUG your_program.cpp -o your_program
```

### CMake Integration
```cmake
# In your CMakeLists.txt
target_include_directories(your_target PRIVATE path/to/ndarray)
```

## API Reference Summary

### Core Classes
- `NDArray<T>`: Main N-dimensional array class
- Type aliases: `Array` (double), `ArrayF` (float), `ArrayI` (int), `ArrayL` (long)

### Creation Functions
- `zeros<T>()`, `ones<T>()`, `full<T>()`, `eye<T>()`
- `arange<T>()`, `linspace<T>()`, `random<T>()`
- `zeros_like()`, `ones_like()`, `full_like()`

### Mathematical Operations
- Element-wise: `+`, `-`, `*`, `/`, `abs()`, `sqrt()`, `exp()`, `log()`, `sin()`, `cos()`, `pow()`
- Reductions: `sum()`, `mean()`, `min()`, `max()`
- Linear algebra: `dot()`, matrix operations

### Neural Network Functions
- Activations: `relu()`, `sigmoid()`, `tanh()`, `softmax()`
- Layers: `conv2d()`, `max_pool2d()`, `batch_norm()`, `dropout()`
- Initialization: `xavier_uniform()`, `he_uniform()`
- Losses: `mean_squared_error()`, `cross_entropy_loss()`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{ndarray_cpp,
  title={NDArray: High-Performance N-Dimensional Array Library for C++},
  author={Rounak Paul},
  year={2025},
  url={https://github.com/Rounak-Paul/NDArray.git}
}
```

## Acknowledgments

- Inspired by NumPy and other scientific computing libraries
- Built with performance and usability in mind
- Designed for modern C++ applications