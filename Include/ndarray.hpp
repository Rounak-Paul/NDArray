#pragma once

#include <vector>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <type_traits>

template<typename T>
class NDArray {
private:
    std::shared_ptr<std::vector<T>> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t offset_;
    
    void calculate_strides() {
        strides_.resize(shape_.size());
        if (!shape_.empty()) {
            strides_.back() = 1;
            for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }
    
    size_t flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Index dimension mismatch");
        }
        size_t idx = offset_;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            idx += indices[i] * strides_[i];
        }
        return idx;
    }
    
    std::vector<size_t> unravel_index(size_t flat_idx) const {
        std::vector<size_t> indices(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) {
            indices[i] = (flat_idx / strides_[i]) % shape_[i];
        }
        return indices;
    }

public:
    // Constructors
    NDArray() : offset_(0) {}
    
    explicit NDArray(const std::vector<size_t>& shape, const T& fill_value = T{})
        : shape_(shape), offset_(0) {
        size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        data_ = std::make_shared<std::vector<T>>(total_size, fill_value);
        calculate_strides();
    }
    
    NDArray(const std::vector<size_t>& shape, const std::vector<T>& data)
        : shape_(shape), offset_(0) {
        size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        if (data.size() != total_size) {
            throw std::invalid_argument("Data size doesn't match shape");
        }
        data_ = std::make_shared<std::vector<T>>(data);
        calculate_strides();
    }
    
    // Initializer list constructors
    NDArray(std::initializer_list<T> list) {
        shape_ = {list.size()};
        data_ = std::make_shared<std::vector<T>>(list);
        offset_ = 0;
        calculate_strides();
    }
    
    template<typename U>
    NDArray(std::initializer_list<std::initializer_list<U>> list) {
        shape_ = {list.size(), list.begin()->size()};
        data_ = std::make_shared<std::vector<T>>();
        data_->reserve(shape_[0] * shape_[1]);
        
        for (const auto& row : list) {
            if (row.size() != shape_[1]) {
                throw std::invalid_argument("Inconsistent row sizes");
            }
            for (const auto& val : row) {
                data_->push_back(static_cast<T>(val));
            }
        }
        offset_ = 0;
        calculate_strides();
    }
    
    // Copy and move constructors
    NDArray(const NDArray& other) = default;
    NDArray(NDArray&& other) noexcept = default;
    NDArray& operator=(const NDArray& other) = default;
    NDArray& operator=(NDArray&& other) noexcept = default;
    
    // Basic properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { 
        return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
    }
    bool empty() const { return size() == 0; }
    
    // Element access
    template<typename... Indices>
    T& operator()(Indices... indices) {
        std::vector<size_t> idx_vec = {static_cast<size_t>(indices)...};
        return (*data_)[flat_index(idx_vec)];
    }
    
    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        std::vector<size_t> idx_vec = {static_cast<size_t>(indices)...};
        return (*data_)[flat_index(idx_vec)];
    }
    
    // Dynamic N-dimensional indexing
    T& operator[](const std::vector<size_t>& indices) {
        return (*data_)[flat_index(indices)];
    }
    
    const T& operator[](const std::vector<size_t>& indices) const {
        return (*data_)[flat_index(indices)];
    }
    
    T& at(const std::vector<size_t>& indices) {
        return (*data_)[flat_index(indices)];
    }
    
    const T& at(const std::vector<size_t>& indices) const {
        return (*data_)[flat_index(indices)];
    }
    
    // Flatten access
    T& flat(size_t index) {
        if (index >= size()) throw std::out_of_range("Flat index out of bounds");
        return (*data_)[offset_ + index];
    }
    
    const T& flat(size_t index) const {
        if (index >= size()) throw std::out_of_range("Flat index out of bounds");
        return (*data_)[offset_ + index];
    }
    
    // Reshape
    NDArray reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
        if (new_size != size()) {
            throw std::invalid_argument("Cannot reshape array: size mismatch");
        }
        
        NDArray result;
        result.data_ = data_;
        result.shape_ = new_shape;
        result.offset_ = offset_;
        result.calculate_strides();
        return result;
    }
    
    // Transpose
    NDArray T() const {
        std::vector<size_t> new_shape = shape_;
        std::reverse(new_shape.begin(), new_shape.end());
        
        NDArray result(new_shape);
        
        for (size_t i = 0; i < size(); ++i) {
            auto indices = unravel_index(i);
            std::reverse(indices.begin(), indices.end());
            result.at(indices) = flat(i);
        }
        
        return result;
    }
    
    // Arithmetic operations
    NDArray operator+(const NDArray& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for addition");
        }
        
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) + other.flat(i);
        }
        return result;
    }
    
    NDArray operator-(const NDArray& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for subtraction");
        }
        
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) - other.flat(i);
        }
        return result;
    }
    
    NDArray operator*(const NDArray& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for multiplication");
        }
        
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) * other.flat(i);
        }
        return result;
    }
    
    NDArray operator/(const NDArray& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Shape mismatch for division");
        }
        
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) / other.flat(i);
        }
        return result;
    }
    
    // Scalar operations
    NDArray operator+(const T& scalar) const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) + scalar;
        }
        return result;
    }
    
    NDArray operator-(const T& scalar) const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) - scalar;
        }
        return result;
    }
    
    NDArray operator*(const T& scalar) const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) * scalar;
        }
        return result;
    }
    
    NDArray operator/(const T& scalar) const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = flat(i) / scalar;
        }
        return result;
    }
    
    // In-place operations
    NDArray& operator+=(const NDArray& other) {
        *this = *this + other;
        return *this;
    }
    
    NDArray& operator-=(const NDArray& other) {
        *this = *this - other;
        return *this;
    }
    
    NDArray& operator*=(const NDArray& other) {
        *this = *this * other;
        return *this;
    }
    
    NDArray& operator/=(const NDArray& other) {
        *this = *this / other;
        return *this;
    }
    
    NDArray& operator+=(const T& scalar) {
        for (size_t i = 0; i < size(); ++i) {
            flat(i) += scalar;
        }
        return *this;
    }
    
    NDArray& operator-=(const T& scalar) {
        for (size_t i = 0; i < size(); ++i) {
            flat(i) -= scalar;
        }
        return *this;
    }
    
    NDArray& operator*=(const T& scalar) {
        for (size_t i = 0; i < size(); ++i) {
            flat(i) *= scalar;
        }
        return *this;
    }
    
    NDArray& operator/=(const T& scalar) {
        for (size_t i = 0; i < size(); ++i) {
            flat(i) /= scalar;
        }
        return *this;
    }
    
    // Neural Network utility functions
template<typename T>
NDArray<T> relu(const NDArray<T>& arr) {
    return arr.relu();
}

template<typename T>
NDArray<T> sigmoid(const NDArray<T>& arr) {
    return arr.sigmoid();
}

template<typename T>
NDArray<T> softmax(const NDArray<T>& arr, size_t axis = -1) {
    return arr.softmax(axis);
}

// Xavier/Glorot initialization
template<typename T>
NDArray<T> xavier_uniform(const std::vector<size_t>& shape, std::mt19937& rng) {
    size_t fan_in = shape.size() > 1 ? shape[shape.size()-2] : 1;
    size_t fan_out = shape.back();
    T limit = std::sqrt(T{6} / (fan_in + fan_out));
    
    std::uniform_real_distribution<T> dist(-limit, limit);
    
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    std::vector<T> data(total_size);
    
    for (auto& val : data) {
        val = dist(rng);
    }
    
    return NDArray<T>(shape, data);
}

// He initialization  
template<typename T>
NDArray<T> he_uniform(const std::vector<size_t>& shape, std::mt19937& rng) {
    size_t fan_in = shape.size() > 1 ? shape[shape.size()-2] : 1;
    T limit = std::sqrt(T{6} / fan_in);
    
    std::uniform_real_distribution<T> dist(-limit, limit);
    
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    std::vector<T> data(total_size);
    
    for (auto& val : data) {
        val = dist(rng);
    }
    
    return NDArray<T>(shape, data);
}

// Loss functions
template<typename T>
T mean_squared_error(const NDArray<T>& predictions, const NDArray<T>& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    T sum = T{};
    for (size_t i = 0; i < predictions.size(); ++i) {
        T diff = predictions.flat(i) - targets.flat(i);
        sum += diff * diff;
    }
    
    return sum / static_cast<T>(predictions.size());
}

template<typename T>
T cross_entropy_loss(const NDArray<T>& predictions, const NDArray<T>& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    T loss = T{};
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss -= targets.flat(i) * std::log(std::max(predictions.flat(i), T{1e-15}));
    }
    
    return loss / static_cast<T>(predictions.shape()[0]); // Assuming batch dimension is first
}
    NDArray abs() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::abs(flat(i));
        }
        return result;
    }
    
    NDArray sqrt() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::sqrt(flat(i));
        }
        return result;
    }
    
    NDArray exp() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::exp(flat(i));
        }
        return result;
    }
    
    NDArray log() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::log(flat(i));
        }
        return result;
    }
    
    NDArray sin() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::sin(flat(i));
        }
        return result;
    }
    
    NDArray cos() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::cos(flat(i));
        }
        return result;
    }
    
    NDArray pow(const T& exponent) const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::pow(flat(i), exponent);
        }
        return result;
    }
    
    // Reduction operations
    T sum() const {
        T result = T{};
        for (size_t i = 0; i < size(); ++i) {
            result += flat(i);
        }
        return result;
    }
    
    T mean() const {
        return sum() / static_cast<T>(size());
    }
    
    T min() const {
        if (size() == 0) throw std::runtime_error("Cannot find min of empty array");
        T result = flat(0);
        for (size_t i = 1; i < size(); ++i) {
            result = std::min(result, flat(i));
        }
        return result;
    }
    
    T max() const {
        if (size() == 0) throw std::runtime_error("Cannot find max of empty array");
        T result = flat(0);
        for (size_t i = 1; i < size(); ++i) {
            result = std::max(result, flat(i));
        }
        return result;
    }
    
    // Axis-wise operations
    NDArray sum(size_t axis) const {
        if (axis >= ndim()) {
            throw std::invalid_argument("Axis out of bounds");
        }
        
        std::vector<size_t> new_shape = shape_;
        new_shape.erase(new_shape.begin() + axis);
        
        if (new_shape.empty()) {
            return NDArray({1}, {sum()});
        }
        
        NDArray result(new_shape, T{});
        
        for (size_t i = 0; i < size(); ++i) {
            auto indices = unravel_index(i);
            auto result_indices = indices;
            result_indices.erase(result_indices.begin() + axis);
            result.at(result_indices) += flat(i);
        }
        
        return result;
    }
    
    NDArray mean(size_t axis) const {
        auto result = sum(axis);
        T divisor = static_cast<T>(shape_[axis]);
        result /= divisor;
        return result;
    }
    
    // Matrix operations (supports N-D tensors)
    NDArray dot(const NDArray& other) const {
        // Handle 1-D vectors
        if (ndim() == 1 && other.ndim() == 1) {
            if (shape_[0] != other.shape_[0]) {
                throw std::invalid_argument("Vector dimensions must match for dot product");
            }
            T result = T{};
            for (size_t i = 0; i < shape_[0]; ++i) {
                result += (*this)({i}) * other({i});
            }
            return NDArray({1}, {result});
        }
        
        // Handle matrix-vector multiplication
        if (ndim() == 2 && other.ndim() == 1) {
            if (shape_[1] != other.shape_[0]) {
                throw std::invalid_argument("Incompatible dimensions for matrix-vector multiplication");
            }
            NDArray result({shape_[0]}, T{});
            for (size_t i = 0; i < shape_[0]; ++i) {
                for (size_t j = 0; j < shape_[1]; ++j) {
                    result({i}) += (*this)({i, j}) * other({j});
                }
            }
            return result;
        }
        
        // Handle standard matrix multiplication
        if (ndim() == 2 && other.ndim() == 2) {
            if (shape_[1] != other.shape_[0]) {
                throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
            }
            
            std::vector<size_t> result_shape = {shape_[0], other.shape_[1]};
            NDArray result(result_shape, T{});
            
            for (size_t i = 0; i < shape_[0]; ++i) {
                for (size_t j = 0; j < other.shape_[1]; ++j) {
                    for (size_t k = 0; k < shape_[1]; ++k) {
                        result({i, j}) += (*this)({i, k}) * other({k, j});
                    }
                }
            }
            
            return result;
        }
        
        // Handle N-D tensor contraction (last axis of first, first axis of second)
        if (ndim() >= 2 && other.ndim() >= 2) {
            if (shape_.back() != other.shape_[0]) {
                throw std::invalid_argument("Incompatible dimensions for tensor dot product");
            }
            
            std::vector<size_t> result_shape = shape_;
            result_shape.pop_back(); // Remove last dimension
            for (size_t i = 1; i < other.shape_.size(); ++i) {
                result_shape.push_back(other.shape_[i]); // Add other's dimensions except first
            }
            
            NDArray result(result_shape, T{});
            
            // This is a simplified N-D implementation
            // For full tensor operations, you'd want optimized BLAS routines
            size_t contract_dim = shape_.back();
            size_t left_size = size() / contract_dim;
            size_t right_size = other.size() / contract_dim;
            
            for (size_t i = 0; i < left_size; ++i) {
                for (size_t j = 0; j < right_size; ++j) {
                    T sum = T{};
                    for (size_t k = 0; k < contract_dim; ++k) {
                        sum += flat(i * contract_dim + k) * other.flat(k * right_size + j);
                    }
                    result.flat(i * right_size + j) = sum;
                }
            }
            
            return result;
        }
        
        throw std::invalid_argument("Unsupported array dimensions for dot product");
    }
    
    // Slicing
    NDArray slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
        if (ranges.size() != ndim()) {
            throw std::invalid_argument("Number of slice ranges must match array dimensions");
        }
        
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i].first >= shape_[i] || ranges[i].second > shape_[i] || 
                ranges[i].first >= ranges[i].second) {
                throw std::invalid_argument("Invalid slice range");
            }
            new_shape.push_back(ranges[i].second - ranges[i].first);
        }
        
        NDArray result(new_shape);
        
        std::function<void(std::vector<size_t>&, size_t)> fill_slice = 
            [&](std::vector<size_t>& indices, size_t dim) {
                if (dim == ndim()) {
                    std::vector<size_t> src_indices = indices;
                    for (size_t i = 0; i < indices.size(); ++i) {
                        src_indices[i] += ranges[i].first;
                    }
                    result.at(indices) = at(src_indices);
                    return;
                }
                
                for (size_t i = 0; i < new_shape[dim]; ++i) {
                    indices[dim] = i;
                    fill_slice(indices, dim + 1);
                }
            };
        
        std::vector<size_t> indices(ndim());
        fill_slice(indices, 0);
        
        return result;
    }
    
    // Neural Network specific operations
    
    // Activation functions
    NDArray relu() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::max(T{}, flat(i));
        }
        return result;
    }
    
    NDArray sigmoid() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = T{1} / (T{1} + std::exp(-flat(i)));
        }
        return result;
    }
    
    NDArray tanh() const {
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            result.flat(i) = std::tanh(flat(i));
        }
        return result;
    }
    
    NDArray softmax(size_t axis = -1) const {
        if (axis == static_cast<size_t>(-1)) {
            axis = ndim() - 1;
        }
        
        // Subtract max for numerical stability
        auto max_vals = max_along_axis(axis);
        auto shifted = subtract_broadcast(max_vals, axis);
        auto exp_vals = shifted.exp();
        auto sum_vals = exp_vals.sum(axis);
        
        return exp_vals.divide_broadcast(sum_vals, axis);
    }
    
    // Convolution operation (basic 2D implementation)
    NDArray conv2d(const NDArray& kernel, size_t stride = 1, size_t padding = 0) const {
        if (ndim() != 4 || kernel.ndim() != 4) {
            throw std::invalid_argument("Conv2D requires 4D tensors (batch, channels, height, width)");
        }
        
        size_t batch_size = shape_[0];
        size_t in_channels = shape_[1];
        size_t in_height = shape_[2];
        size_t in_width = shape_[3];
        
        size_t out_channels = kernel.shape_[0];
        size_t kernel_height = kernel.shape_[2];
        size_t kernel_width = kernel.shape_[3];
        
        if (in_channels != kernel.shape_[1]) {
            throw std::invalid_argument("Input channels must match kernel input channels");
        }
        
        size_t out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        size_t out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        
        NDArray result({batch_size, out_channels, out_height, out_width}, T{});
        
        // Simple convolution implementation (not optimized)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        T sum = T{};
                        for (size_t ic = 0; ic < in_channels; ++ic) {
                            for (size_t kh = 0; kh < kernel_height; ++kh) {
                                for (size_t kw = 0; kw < kernel_width; ++kw) {
                                    size_t ih = oh * stride + kh - padding;
                                    size_t iw = ow * stride + kw - padding;
                                    
                                    if (ih < in_height && iw < in_width) {
                                        sum += (*this)({b, ic, ih, iw}) * kernel({oc, ic, kh, kw});
                                    }
                                }
                            }
                        }
                        result({b, oc, oh, ow}) = sum;
                    }
                }
            }
        }
        
        return result;
    }
    
    // Max pooling
    NDArray max_pool2d(size_t pool_size, size_t stride = 0) const {
        if (stride == 0) stride = pool_size;
        
        if (ndim() != 4) {
            throw std::invalid_argument("MaxPool2D requires 4D tensor");
        }
        
        size_t batch_size = shape_[0];
        size_t channels = shape_[1];
        size_t in_height = shape_[2];
        size_t in_width = shape_[3];
        
        size_t out_height = (in_height - pool_size) / stride + 1;
        size_t out_width = (in_width - pool_size) / stride + 1;
        
        NDArray result({batch_size, channels, out_height, out_width});
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_height; ++oh) {
                    for (size_t ow = 0; ow < out_width; ++ow) {
                        T max_val = std::numeric_limits<T>::lowest();
                        
                        for (size_t ph = 0; ph < pool_size; ++ph) {
                            for (size_t pw = 0; pw < pool_size; ++pw) {
                                size_t ih = oh * stride + ph;
                                size_t iw = ow * stride + pw;
                                max_val = std::max(max_val, (*this)({b, c, ih, iw}));
                            }
                        }
                        
                        result({b, c, oh, ow}) = max_val;
                    }
                }
            }
        }
        
        return result;
    }
    
    // Batch normalization (inference mode)
    NDArray batch_norm(const NDArray& mean, const NDArray& var, 
                      const NDArray& gamma, const NDArray& beta, T eps = T{1e-5}) const {
        // Assume channel dimension is 1 (standard for NCHW format)
        if (mean.size() != shape_[1] || var.size() != shape_[1] || 
            gamma.size() != shape_[1] || beta.size() != shape_[1]) {
            throw std::invalid_argument("BatchNorm parameters must match channel dimension");
        }
        
        NDArray result(shape_);
        
        for (size_t i = 0; i < size(); ++i) {
            auto indices = unravel_index(i);
            size_t channel = indices[1];
            
            T normalized = (flat(i) - mean.flat(channel)) / std::sqrt(var.flat(channel) + eps);
            result.flat(i) = gamma.flat(channel) * normalized + beta.flat(channel);
        }
        
        return result;
    }
    
    // Dropout (training mode)
    NDArray dropout(T probability, std::mt19937& rng) const {
        std::uniform_real_distribution<T> dist(T{0}, T{1});
        T scale = T{1} / (T{1} - probability);
        
        NDArray result(shape_);
        for (size_t i = 0; i < size(); ++i) {
            if (dist(rng) < probability) {
                result.flat(i) = T{};
            } else {
                result.flat(i) = flat(i) * scale;
            }
        }
        return result;
    }

private:
    // Helper functions for neural network operations
    NDArray max_along_axis(size_t axis) const {
        std::vector<size_t> new_shape = shape_;
        new_shape[axis] = 1;
        
        NDArray result(new_shape);
        
        // Implementation would need to iterate through axis
        // Simplified version - you'd want to optimize this
        for (size_t i = 0; i < result.size(); ++i) {
            auto result_indices = result.unravel_index(i);
            T max_val = std::numeric_limits<T>::lowest();
            
            for (size_t j = 0; j < shape_[axis]; ++j) {
                auto indices = result_indices;
                indices[axis] = j;
                max_val = std::max(max_val, at(indices));
            }
            
            result.flat(i) = max_val;
        }
        
        return result;
    }
    
    NDArray subtract_broadcast(const NDArray& other, size_t axis) const {
        NDArray result(shape_);
        
        for (size_t i = 0; i < size(); ++i) {
            auto indices = unravel_index(i);
            auto other_indices = indices;
            other_indices[axis] = 0;  // Broadcast along axis
            
            result.flat(i) = flat(i) - other.at(other_indices);
        }
        
        return result;
    }
    
    NDArray divide_broadcast(const NDArray& other, size_t axis) const {
        NDArray result(shape_);
        
        for (size_t i = 0; i < size(); ++i) {
            auto indices = unravel_index(i);
            auto other_indices = indices;
            other_indices[axis] = 0;  // Broadcast along axis
            
            result.flat(i) = flat(i) / other.at(other_indices);
        }
        
        return result;
    }
    static NDArray concatenate(const std::vector<NDArray>& arrays, size_t axis = 0) {
        if (arrays.empty()) {
            throw std::invalid_argument("Cannot concatenate empty array list");
        }
        
        const auto& first = arrays[0];
        if (axis >= first.ndim()) {
            throw std::invalid_argument("Axis out of bounds");
        }
        
        // Check shape compatibility
        for (size_t i = 1; i < arrays.size(); ++i) {
            if (arrays[i].ndim() != first.ndim()) {
                throw std::invalid_argument("All arrays must have same number of dimensions");
            }
            for (size_t j = 0; j < first.ndim(); ++j) {
                if (j != axis && arrays[i].shape_[j] != first.shape_[j]) {
                    throw std::invalid_argument("Array dimensions must match except for concatenation axis");
                }
            }
        }
        
        // Calculate new shape
        std::vector<size_t> new_shape = first.shape_;
        for (size_t i = 1; i < arrays.size(); ++i) {
            new_shape[axis] += arrays[i].shape_[axis];
        }
        
        NDArray result(new_shape);
        
        // Copy data
        size_t offset = 0;
        for (const auto& arr : arrays) {
            for (size_t i = 0; i < arr.size(); ++i) {
                auto indices = arr.unravel_index(i);
                indices[axis] += offset;
                result.at(indices) = arr.flat(i);
            }
            offset += arr.shape_[axis];
        }
        
        return result;
    }
    
    // Broadcasting check
    static bool can_broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
        size_t max_dims = std::max(shape1.size(), shape2.size());
        
        for (size_t i = 0; i < max_dims; ++i) {
            size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
            size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
            
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                return false;
            }
        }
        
        return true;
    }
    
    // String representation (handles any N dimensions)
    std::string to_string() const {
        std::ostringstream oss;
        
        if (ndim() == 0) {
            oss << flat(0);
            return oss.str();
        }
        
        if (ndim() == 1) {
            oss << "[";
            for (size_t i = 0; i < shape_[0]; ++i) {
                if (i > 0) oss << " ";
                oss << (*this)({i});
            }
            oss << "]";
        } else if (ndim() == 2) {
            oss << "[";
            for (size_t i = 0; i < shape_[0]; ++i) {
                if (i > 0) oss << "\n ";
                oss << "[";
                for (size_t j = 0; j < shape_[1]; ++j) {
                    if (j > 0) oss << " ";
                    oss << std::setw(8) << (*this)({i, j});
                }
                oss << "]";
            }
            oss << "]";
        } else {
            // For 3D and higher, show summary information and some elements
            oss << "NDArray(shape=[";
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << shape_[i];
            }
            oss << "], dtype=" << typeid(T).name() << ")\n";
            
            // Show first few elements for preview
            oss << "First elements: [";
            size_t preview_count = std::min(size_t(10), size());
            for (size_t i = 0; i < preview_count; ++i) {
                if (i > 0) oss << ", ";
                oss << flat(i);
            }
            if (size() > preview_count) {
                oss << ", ...";
            }
            oss << "]";
        }
        
        return oss.str();
    }
    
    // Recursive helper for N-dimensional printing (optional detailed view)
    void print_recursive(std::ostream& os, std::vector<size_t>& indices, size_t dim, 
                        size_t max_elements_per_dim = 5) const {
        if (dim == ndim()) {
            os << (*this)[indices];
            return;
        }
        
        os << "[";
        size_t elements_to_show = std::min(shape_[dim], max_elements_per_dim);
        bool truncated = shape_[dim] > max_elements_per_dim;
        
        for (size_t i = 0; i < elements_to_show; ++i) {
            if (i > 0) {
                os << (dim == ndim() - 1 ? " " : "\n");
                for (size_t d = 0; d <= dim; ++d) os << " ";
            }
            indices[dim] = i;
            print_recursive(os, indices, dim + 1, max_elements_per_dim);
        }
        
        if (truncated) {
            os << (dim == ndim() - 1 ? " ..." : "\n...");
        }
        
        os << "]";
    }
};

// Free functions (NumPy-style)
template<typename T>
NDArray<T> zeros(const std::vector<size_t>& shape) {
    return NDArray<T>(shape, T{});
}

template<typename T>
NDArray<T> ones(const std::vector<size_t>& shape) {
    return NDArray<T>(shape, T{1});
}

template<typename T>
NDArray<T> full(const std::vector<size_t>& shape, const T& value) {
    return NDArray<T>(shape, value);
}

template<typename T>
NDArray<T> arange(T start, T stop, T step = T{1}) {
    std::vector<T> data;
    for (T val = start; val < stop; val += step) {
        data.push_back(val);
    }
    return NDArray<T>({data.size()}, data);
}

template<typename T>
NDArray<T> linspace(T start, T stop, size_t num = 50) {
    std::vector<T> data;
    if (num == 0) return NDArray<T>({0});
    if (num == 1) return NDArray<T>({1}, {start});
    
    T step = (stop - start) / static_cast<T>(num - 1);
    for (size_t i = 0; i < num; ++i) {
        data.push_back(start + static_cast<T>(i) * step);
    }
    return NDArray<T>({num}, data);
}

template<typename T>
NDArray<T> eye(size_t n) {
    NDArray<T> result({n, n}, T{});
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

template<typename T>
NDArray<T> random(const std::vector<size_t>& shape, T min_val = T{0}, T max_val = T{1}) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(static_cast<double>(min_val), static_cast<double>(max_val));
    
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    std::vector<T> data(total_size);
    
    for (auto& val : data) {
        val = static_cast<T>(dis(gen));
    }
    
    return NDArray<T>(shape, data);
}

// Mathematical functions
template<typename T>
NDArray<T> abs(const NDArray<T>& arr) {
    return arr.abs();
}

template<typename T>
NDArray<T> sqrt(const NDArray<T>& arr) {
    return arr.sqrt();
}

template<typename T>
NDArray<T> exp(const NDArray<T>& arr) {
    return arr.exp();
}

template<typename T>
NDArray<T> log(const NDArray<T>& arr) {
    return arr.log();
}

template<typename T>
NDArray<T> sin(const NDArray<T>& arr) {
    return arr.sin();
}

template<typename T>
NDArray<T> cos(const NDArray<T>& arr) {
    return arr.cos();
}

template<typename T>
NDArray<T> pow(const NDArray<T>& arr, const T& exponent) {
    return arr.pow(exponent);
}

// Output stream operator
template<typename T>
std::ostream& operator<<(std::ostream& os, const NDArray<T>& arr) {
    os << arr.to_string();
    return os;
}

// N-dimensional array creation helpers
template<typename T>
NDArray<T> zeros_like(const NDArray<T>& arr) {
    return NDArray<T>(arr.shape(), T{});
}

template<typename T>
NDArray<T> ones_like(const NDArray<T>& arr) {
    return NDArray<T>(arr.shape(), T{1});
}

template<typename T>
NDArray<T> full_like(const NDArray<T>& arr, const T& value) {
    return NDArray<T>(arr.shape(), value);
}

// N-dimensional meshgrid
template<typename T>
std::vector<NDArray<T>> meshgrid(const std::vector<NDArray<T>>& arrays) {
    if (arrays.empty()) {
        throw std::invalid_argument("At least one array required for meshgrid");
    }
    
    // All input arrays must be 1D
    for (const auto& arr : arrays) {
        if (arr.ndim() != 1) {
            throw std::invalid_argument("All input arrays must be 1-dimensional");
        }
    }
    
    // Calculate output shape
    std::vector<size_t> output_shape;
    for (const auto& arr : arrays) {
        output_shape.push_back(arr.shape()[0]);
    }
    
    std::vector<NDArray<T>> result;
    
    for (size_t arr_idx = 0; arr_idx < arrays.size(); ++arr_idx) {
        NDArray<T> grid(output_shape);
        
        for (size_t i = 0; i < grid.size(); ++i) {
            auto indices = grid.unravel_index(i);
            grid.flat(i) = arrays[arr_idx][{indices[arr_idx]}];
        }
        
        result.push_back(grid);
    }
    
    return result;
}

// N-dimensional tensor operations
template<typename T>
NDArray<T> tensordot(const NDArray<T>& a, const NDArray<T>& b, 
                     const std::vector<size_t>& axes_a, const std::vector<size_t>& axes_b) {
    if (axes_a.size() != axes_b.size()) {
        throw std::invalid_argument("Number of axes to contract must be equal");
    }
    
    // Verify axes are valid
    for (size_t axis : axes_a) {
        if (axis >= a.ndim()) {
            throw std::invalid_argument("Axis out of bounds for first array");
        }
    }
    for (size_t axis : axes_b) {
        if (axis >= b.ndim()) {
            throw std::invalid_argument("Axis out of bounds for second array");
        }
    }
    
    // Verify dimensions match for contraction
    for (size_t i = 0; i < axes_a.size(); ++i) {
        if (a.shape()[axes_a[i]] != b.shape()[axes_b[i]]) {
            throw std::invalid_argument("Contracted dimensions must have same size");
        }
    }
    
    // For simplicity, use the standard dot product for common cases
    // Full N-dimensional tensor contraction would require complex reshaping
    if (axes_a.size() == 1 && axes_a[0] == a.ndim() - 1 && axes_b[0] == 0) {
        return a.dot(b);
    }
    
    throw std::runtime_error("General tensordot not fully implemented - use dot() for standard cases");
}

// Type aliases for convenience
using Array = NDArray<double>;
using ArrayF = NDArray<float>;
using ArrayI = NDArray<int>;
using ArrayL = NDArray<long>;