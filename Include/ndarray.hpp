#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <memory>
#include <initializer_list>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <sstream>
#include <random>
#include <cassert>
#include <type_traits>

template<typename T>
class NDArray {
private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    
    void calculate_strides();
    size_t flat_index(const std::vector<size_t>& indices) const;
    std::vector<size_t> unravel_index(size_t flat_idx) const;
    bool can_broadcast(const NDArray& other) const;
    std::vector<size_t> broadcast_shape(const NDArray& other) const;

public:
    // Constructors and Destructors
    NDArray();
    NDArray(const std::vector<size_t>& shape);
    NDArray(const std::vector<size_t>& shape, const T& value);
    NDArray(std::initializer_list<T> list);
    NDArray(const NDArray& other);
    NDArray(NDArray&& other) noexcept;
    ~NDArray() = default;
    
    // Assignment operators
    NDArray& operator=(const NDArray& other);
    NDArray& operator=(NDArray&& other) noexcept;
    NDArray& operator=(const T& value);
    
    // Factory methods
    static NDArray zeros(const std::vector<size_t>& shape);
    static NDArray ones(const std::vector<size_t>& shape);
    static NDArray full(const std::vector<size_t>& shape, const T& value);
    static NDArray empty(const std::vector<size_t>& shape);
    static NDArray arange(T start, T stop, T step = T(1));
    static NDArray linspace(T start, T stop, size_t num = 50);
    static NDArray logspace(T start, T stop, size_t num = 50, T base = T(10));
    static NDArray eye(size_t n, size_t m = 0);
    static NDArray identity(size_t n);
    static NDArray random(const std::vector<size_t>& shape, T min_val = T(0), T max_val = T(1));
    static NDArray randn(const std::vector<size_t>& shape);
    static NDArray from_vector(const std::vector<T>& vec, const std::vector<size_t>& shape);
    
    // Shape and dimension methods
    const std::vector<size_t>& shape() const;
    size_t ndim() const;
    size_t size() const;
    size_t itemsize() const;
    size_t nbytes() const;
    bool empty() const;
    
    // Indexing and slicing
    T& operator[](size_t idx);
    const T& operator[](size_t idx) const;
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;
    
    // Multi-dimensional indexing
    template<typename... Args>
    T& operator()(Args... args);
    template<typename... Args>
    const T& operator()(Args... args) const;
    
    // Slicing
    NDArray slice(const std::vector<std::pair<int, int>>& ranges) const;
    NDArray operator[](const std::vector<size_t>& indices) const;
    
    // Reshape and resize
    NDArray reshape(const std::vector<size_t>& new_shape) const;
    NDArray& reshape_inplace(const std::vector<size_t>& new_shape);
    NDArray resize(const std::vector<size_t>& new_shape) const;
    NDArray flatten() const;
    NDArray ravel() const;
    NDArray squeeze(int axis = -1) const;
    NDArray expand_dims(int axis) const;
    
    // Transpose and axis manipulation
    NDArray transpose() const;
    NDArray transpose(const std::vector<size_t>& axes) const;
    NDArray swapaxes(size_t axis1, size_t axis2) const;
    NDArray moveaxis(size_t source, size_t destination) const;
    NDArray rollaxis(size_t axis, size_t start = 0) const;
    
    // Arithmetic operations
    NDArray operator+(const NDArray& other) const;
    NDArray operator-(const NDArray& other) const;
    NDArray operator*(const NDArray& other) const;
    NDArray operator/(const NDArray& other) const;
    NDArray operator%(const NDArray& other) const;
    NDArray pow(const NDArray& other) const; // power
    
    // Scalar arithmetic operations
    NDArray operator+(const T& scalar) const;
    NDArray operator-(const T& scalar) const;
    NDArray operator*(const T& scalar) const;
    NDArray operator/(const T& scalar) const;
    NDArray operator%(const T& scalar) const;
    NDArray pow(const T& scalar) const;
    
    // In-place arithmetic operations
    NDArray& operator+=(const NDArray& other);
    NDArray& operator-=(const NDArray& other);
    NDArray& operator*=(const NDArray& other);
    NDArray& operator/=(const NDArray& other);
    NDArray& operator%=(const NDArray& other);
    NDArray& pow_inplace(const NDArray& other);
    
    // In-place scalar operations
    NDArray& operator+=(const T& scalar);
    NDArray& operator-=(const T& scalar);
    NDArray& operator*=(const T& scalar);
    NDArray& operator/=(const T& scalar);
    NDArray& operator%=(const T& scalar);
    NDArray& pow_inplace(const T& scalar);
    
    // Unary operations
    NDArray operator-() const;
    NDArray operator+() const;
    
    // Comparison operations
    NDArray operator==(const NDArray& other) const;
    NDArray operator!=(const NDArray& other) const;
    NDArray operator<(const NDArray& other) const;
    NDArray operator<=(const NDArray& other) const;
    NDArray operator>(const NDArray& other) const;
    NDArray operator>=(const NDArray& other) const;
    
    // Scalar comparison operations
    NDArray operator==(const T& scalar) const;
    NDArray operator!=(const T& scalar) const;
    NDArray operator<(const T& scalar) const;
    NDArray operator<=(const T& scalar) const;
    NDArray operator>(const T& scalar) const;
    NDArray operator>=(const T& scalar) const;
    
    // Logical operations
    NDArray operator&&(const NDArray& other) const;
    NDArray operator||(const NDArray& other) const;
    NDArray operator!() const;
    
    // Bitwise operations (for integer types)
    NDArray operator&(const NDArray& other) const;
    NDArray operator|(const NDArray& other) const;
    NDArray operator^(const NDArray& other) const; // XOR
    NDArray operator~() const;
    NDArray operator<<(const NDArray& other) const;
    NDArray operator>>(const NDArray& other) const;
    
    // Mathematical functions
    NDArray abs() const;
    NDArray sqrt() const;
    NDArray exp() const;
    NDArray exp2() const;
    NDArray log() const;
    NDArray log2() const;
    NDArray log10() const;
    NDArray sin() const;
    NDArray cos() const;
    NDArray tan() const;
    NDArray asin() const;
    NDArray acos() const;
    NDArray atan() const;
    NDArray sinh() const;
    NDArray cosh() const;
    NDArray tanh() const;
    NDArray floor() const;
    NDArray ceil() const;
    NDArray round() const;
    NDArray sign() const;
    NDArray reciprocal() const;
    NDArray square() const;
    NDArray power(const T& exponent) const;
    NDArray power(const NDArray& exponent) const;
    
    // Reduction operations
    T sum() const;
    T sum(int axis) const;
    NDArray sum_axis(int axis) const;
    T prod() const;
    T prod(int axis) const;
    NDArray prod_axis(int axis) const;
    T mean() const;
    T mean(int axis) const;
    NDArray mean_axis(int axis) const;
    T var() const;
    T var(int axis, int ddof = 0) const;
    NDArray var_axis(int axis, int ddof = 0) const;
    T std() const;
    T std(int axis, int ddof = 0) const;
    NDArray std_axis(int axis, int ddof = 0) const;
    T min() const;
    T min(int axis) const;
    NDArray min_axis(int axis) const;
    T max() const;
    T max(int axis) const;
    NDArray max_axis(int axis) const;
    size_t argmin() const;
    size_t argmin(int axis) const;
    NDArray argmin_axis(int axis) const;
    size_t argmax() const;
    size_t argmax(int axis) const;
    NDArray argmax_axis(int axis) const;
    
    // Cumulative operations
    NDArray cumsum() const;
    NDArray cumsum(int axis) const;
    NDArray cumprod() const;
    NDArray cumprod(int axis) const;
    
    // Sorting and searching
    NDArray sort() const;
    NDArray sort(int axis) const;
    NDArray argsort() const;
    NDArray argsort(int axis) const;
    NDArray unique() const;
    NDArray searchsorted(const T& value) const;
    NDArray searchsorted(const NDArray& values) const;
    
    // Linear algebra
    NDArray dot(const NDArray& other) const;
    NDArray matmul(const NDArray& other) const;
    NDArray cross(const NDArray& other) const;
    T trace() const;
    T det() const;
    NDArray inv() const;
    std::pair<NDArray, NDArray> eig() const;
    std::tuple<NDArray, NDArray, NDArray> svd() const;
    NDArray solve(const NDArray& b) const;
    
    // Array manipulation
    NDArray concatenate(const NDArray& other, int axis = 0) const;
    NDArray append(const NDArray& other, int axis = -1) const;
    NDArray insert(size_t index, const NDArray& values, int axis = -1) const;
    NDArray delete_elements(const std::vector<size_t>& indices, int axis = -1) const;
    NDArray split(size_t sections, int axis = 0) const;
    std::vector<NDArray> split_array(size_t sections, int axis = 0) const;
    NDArray repeat(size_t repeats, int axis = -1) const;
    NDArray tile(const std::vector<size_t>& reps) const;
    NDArray flip(int axis = -1) const;
    NDArray roll(int shift, int axis = -1) const;
    
    // Stacking and joining
    static NDArray stack(const std::vector<NDArray>& arrays, int axis = 0);
    static NDArray vstack(const std::vector<NDArray>& arrays);
    static NDArray hstack(const std::vector<NDArray>& arrays);
    static NDArray dstack(const std::vector<NDArray>& arrays);
    static NDArray concatenate(const std::vector<NDArray>& arrays, int axis = 0);
    
    // Broadcasting
    NDArray broadcast_to(const std::vector<size_t>& shape) const;
    static std::pair<NDArray, NDArray> broadcast_arrays(const NDArray& a, const NDArray& b);
    
    // Conditional operations
    NDArray where(const NDArray& condition, const NDArray& y) const;
    NDArray where(const NDArray& condition, const T& y) const;
    static NDArray where(const NDArray& condition, const NDArray& x, const NDArray& y);
    NDArray select(const std::vector<NDArray>& condlist, const std::vector<NDArray>& choicelist, const T& default_val = T(0)) const;
    
    // Type conversion
    template<typename U>
    NDArray<U> astype() const;
    NDArray copy() const;
    NDArray view() const;
    
    // I/O and string representation
    std::string str() const;
    std::string repr() const;
    void print() const;
    void save(const std::string& filename) const;
    static NDArray load(const std::string& filename);
    
    // Memory and data access
    T* data();
    const T* data() const;
    std::vector<T>& get_data();
    const std::vector<T>& get_data() const;
    bool is_contiguous() const;
    
    // Statistical functions
    T median() const;
    T percentile(T q) const;
    T quantile(T q) const;
    std::pair<T, T> histogram_range() const;
    NDArray histogram(size_t bins = 10) const;
    T correlation(const NDArray& other) const;
    T covariance(const NDArray& other) const;
    
    // Set operations
    NDArray intersect1d(const NDArray& other) const;
    NDArray union1d(const NDArray& other) const;
    NDArray setdiff1d(const NDArray& other) const;
    NDArray setxor1d(const NDArray& other) const;
    bool in1d(const T& value) const;
    NDArray isin(const NDArray& test_elements) const;
    
    // Iteration support
    class iterator {
    private:
        typename std::vector<T>::iterator it_;
    public:
        explicit iterator(typename std::vector<T>::iterator it) : it_(it) {}
        T& operator*() { return *it_; }
        iterator& operator++() { ++it_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++it_; return tmp; }
        bool operator==(const iterator& other) const { return it_ == other.it_; }
        bool operator!=(const iterator& other) const { return it_ != other.it_; }
    };
    
    class const_iterator {
    private:
        typename std::vector<T>::const_iterator it_;
    public:
        explicit const_iterator(typename std::vector<T>::const_iterator it) : it_(it) {}
        const T& operator*() const { return *it_; }
        const_iterator& operator++() { ++it_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++it_; return tmp; }
        bool operator==(const const_iterator& other) const { return it_ == other.it_; }
        bool operator!=(const const_iterator& other) const { return it_ != other.it_; }
    };
    
    iterator begin() { return iterator(data_.begin()); }
    iterator end() { return iterator(data_.end()); }
    const_iterator begin() const { return const_iterator(data_.begin()); }
    const_iterator end() const { return const_iterator(data_.end()); }
    const_iterator cbegin() const { return const_iterator(data_.cbegin()); }
    const_iterator cend() const { return const_iterator(data_.cend()); }
};

// Global functions for NDArray operations
template<typename T>
NDArray<T> operator+(const T& scalar, const NDArray<T>& arr);

template<typename T>
NDArray<T> operator-(const T& scalar, const NDArray<T>& arr);

template<typename T>
NDArray<T> operator*(const T& scalar, const NDArray<T>& arr);

template<typename T>
NDArray<T> operator/(const T& scalar, const NDArray<T>& arr);

template<typename T>
std::ostream& operator<<(std::ostream& os, const NDArray<T>& arr);

// Mathematical functions (global)
template<typename T>
NDArray<T> abs(const NDArray<T>& arr);

template<typename T>
NDArray<T> sqrt(const NDArray<T>& arr);

template<typename T>
NDArray<T> exp(const NDArray<T>& arr);

template<typename T>
NDArray<T> log(const NDArray<T>& arr);

template<typename T>
NDArray<T> sin(const NDArray<T>& arr);

template<typename T>
NDArray<T> cos(const NDArray<T>& arr);

template<typename T>
NDArray<T> tan(const NDArray<T>& arr);

template<typename T>
NDArray<T> power(const NDArray<T>& arr, const T& exponent);

template<typename T>
NDArray<T> power(const NDArray<T>& base, const NDArray<T>& exponent);

// Utility functions
template<typename T>
NDArray<T> maximum(const NDArray<T>& a, const NDArray<T>& b);

template<typename T>
NDArray<T> minimum(const NDArray<T>& a, const NDArray<T>& b);

template<typename T>
NDArray<T> clip(const NDArray<T>& arr, const T& min_val, const T& max_val);

template<typename T>
bool allclose(const NDArray<T>& a, const NDArray<T>& b, T rtol = T(1e-5), T atol = T(1e-8));

template<typename T>
bool array_equal(const NDArray<T>& a, const NDArray<T>& b);

// Type aliases for common array types
using Array = NDArray<double>;
using ArrayF = NDArray<float>;
using ArrayI = NDArray<int>;
using ArrayL = NDArray<long>;
using ArrayB = NDArray<bool>;

#endif // NDARRAY_H