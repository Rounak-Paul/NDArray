#include "../Include/ndarray.hpp"

#include <iostream>
#include <random>

class SimpleNN {
private:
    ArrayF W1, b1, W2, b2;
    std::mt19937 rng;
    
public:
    SimpleNN(size_t input_size, size_t hidden_size, size_t output_size) 
        : rng(std::random_device{}()) {
        
        // Initialize weights with He initialization
        W1 = he_uniform<float>({input_size, hidden_size}, rng);
        b1 = zeros<float>({1, hidden_size});
        W2 = he_uniform<float>({hidden_size, output_size}, rng);
        b2 = zeros<float>({1, output_size});
    }
    
    ArrayF forward(const ArrayF& X) {
        // First layer: X @ W1 + b1
        auto z1 = X.dot(W1) + b1;  // Broadcasting handles bias addition
        auto a1 = relu(z1);    // ReLU activation
        
        // Output layer: a1 @ W2 + b2  
        auto z2 = a1.dot(W2) + b2;
        auto output = softmax(z2, 1);  // Softmax along feature axis
        
        return output;
    }
    
    void train_step(const ArrayF& X, const ArrayF& y, float lr = 0.01f) {
        // Forward pass
        auto predictions = forward(X);
        
        // Compute loss
        float loss = cross_entropy_loss(predictions, y);
        std::cout << "Loss: " << loss << std::endl;
        
        // Backward pass would go here (you'd need gradient computation)
        // This is where you'd implement backpropagation
    }
};

int main() {
    using namespace np;
    
    // Create a simple neural network
    SimpleNN model(784, 128, 10);  // MNIST-like architecture
    
    // Create some dummy data
    auto X = random<float>({32, 784});  // Batch of 32 samples, 784 features
    auto y = zeros<float>({32, 10});    // One-hot encoded labels
    
    // Fill some random labels
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9);
    
    for (size_t i = 0; i < 32; ++i) {
        y[{i, static_cast<size_t>(dis(gen))}] = 1.0f;
    }
    
    // Train for a few steps
    for (int epoch = 0; epoch < 5; ++epoch) {
        std::cout << "Epoch " << epoch << ": ";
        model.train_step(X, y);
    }
    
    // Test convolution
    auto image_batch = random<float>({8, 3, 32, 32});  // 8 RGB images, 32x32
    auto conv_kernel = random<float>({16, 3, 3, 3});   // 16 filters, 3x3
    
    auto conv_output = image_batch.conv2d(conv_kernel);
    std::cout << "\nConv output shape: ";
    for (auto dim : conv_output.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return 0;
}