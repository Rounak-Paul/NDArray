#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <vector>
#include <numeric>      // For std::accumulate, std::iota
#include <stdexcept>    // For std::out_of_range, std::invalid_argument
#include <memory>       // For std::shared_ptr, std::make_shared
#include <functional>   // For std::function, std::multiplies
#include <iostream>     // For std::ostream
#include <algorithm>    // For std::copy, std::equal, std::fill
#include <initializer_list> // For std::initializer_list
#include <string>       // For std::to_string in printing
#include <sstream>      // For std::ostringstream in printing

namespace nd {

template <typename T>
class NDArray {
private:
    std::shared_ptr<std::vector<T>> data_ptr_; // Pointer to the underlying flat data
    std::vector<size_t> shape_;                // Dimensions of the array
    std::vector<size_t> strides_;              // Strides for each dimension
    size_t offset_;                            // Offset into data_ptr_ for this view
    size_t total_size_;                        // Total number of elements in this view

    // Helper to calculate total size from shape
    static size_t calculate_total_size_internal(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return 1; // Scalar (0-D array)
        }
        for (size_t dim_size : shape) {
            if (dim_size == 0) return 0; // If any dimension is 0, total size is 0
        }
        return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    }

    // Helper to calculate strides from shape (C-style, row-major)
    static std::vector<size_t> calculate_strides_internal(const std::vector<size_t>& shape) {
        if (shape.empty()) {
            return {}; // No strides for a 0-D array
        }
        std::vector<size_t> strides(shape.size());
        if (shape.empty() || calculate_total_size_internal(shape) == 0) { // Also handle shapes like (2,0,3)
             std::fill(strides.begin(), strides.end(), 0); // or some other indicator for empty/zero-size
             if (!shape.empty()) strides.back() = 1; // Convention for last stride even if total size is 0
            return strides;
        }

        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    // Helper to get flat index in data_ptr_ from N-D indices
    size_t get_flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != ndim()) {
            throw std::out_of_range("Number of indices does not match array dimension.");
        }
        if (ndim() == 0) { // 0-D array
            if (!indices.empty()) throw std::out_of_range("Indices provided for 0-D array.");
            return offset_;
        }
        size_t flat_idx = offset_;
        for (size_t i = 0; i < ndim(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index " + std::to_string(indices[i]) +
                                        " is out of bounds for dimension " + std::to_string(i) +
                                        " with size " + std::to_string(shape_[i]));
            }
            flat_idx += indices[i] * strides_[i];
        }
        return flat_idx;
    }

    // Iterative helper for element-wise operations
    // Takes a function f that accepts current N-D indices
    void for_each_index_iterative(std::function<void(const std::vector<size_t>&)> f) const {
        if (total_size_ == 0 && ndim() > 0) return; // Empty array with dimensions
        if (ndim() == 0) { // Scalar (0-D array)
            if (total_size_ == 1) f({}); // Empty vector for indices
            return;
        }

        std::vector<size_t> current_indices(ndim(), 0);
        while (true) {
            f(current_indices);

            int dim_to_increment = static_cast<int>(ndim()) - 1;
            while (dim_to_increment >= 0) {
                current_indices[dim_to_increment]++;
                if (current_indices[dim_to_increment] < shape_[dim_to_increment]) {
                    break; // No carry-over needed
                }
                current_indices[dim_to_increment] = 0; // Reset current dim, carry to next
                dim_to_increment--;
            }
            if (dim_to_increment < 0) {
                break; // All indices have been iterated
            }
        }
    }
    
    // Recursive helper for printing
    void print_recursive(std::ostream& os, std::vector<size_t>& current_indices, size_t current_dim) const {
        if (current_dim == ndim()) { // Base case: print element
            os << this->at(current_indices);
            return;
        }

        os << "[";
        for (size_t i = 0; i < shape_[current_dim]; ++i) {
            current_indices[current_dim] = i;
            print_recursive(os, current_indices, current_dim + 1);
            if (i < shape_[current_dim] - 1) {
                os << ( (current_dim == ndim() - 1) ? ", " : ",\n" + std::string(current_dim + 1, ' ') );
            }
        }
        os << "]";
    }


public:
    // Default constructor: creates an empty array (0-D, 1 element, uninitialized or default T)
    NDArray() : offset_(0), total_size_(1) {
        shape_ = {}; // 0-D
        strides_ = calculate_strides_internal(shape_);
        data_ptr_ = std::make_shared<std::vector<T>>(1); // Uninitialized scalar
    }

    // Constructor: creates a 0-D array (scalar) from a single value
    explicit NDArray(const T& scalar_value)
        : shape_({}), offset_(0), total_size_(1) {
        strides_ = calculate_strides_internal(shape_);
        data_ptr_ = std::make_shared<std::vector<T>>(1, scalar_value);
    }

    // Constructor: creates an uninitialized array of a given shape
    explicit NDArray(const std::vector<size_t>& shape)
        : shape_(shape), offset_(0) {
        total_size_ = calculate_total_size_internal(shape_);
        strides_ = calculate_strides_internal(shape_);
        if (total_size_ > 0 || shape_.empty()) { // Allow 0-D array (total_size_ = 1)
             data_ptr_ = std::make_shared<std::vector<T>>(total_size_ > 0 ? total_size_ : 1); // Avoid 0-size vector if shape is like {}
        } else { // shape like (2,0,3)
             data_ptr_ = std::make_shared<std::vector<T>>(); // Empty data
        }
         if (shape_.empty() && total_size_ == 1 && data_ptr_->empty()) { // Ensure 0-D has space for 1 element
            data_ptr_->resize(1);
        }
    }
    // Convenience constructor with initializer_list for shape
    NDArray(std::initializer_list<size_t> shape_list)
        : NDArray(std::vector<size_t>(shape_list)) {}


    // Constructor: creates an array of a given shape, initialized with a value
    NDArray(const std::vector<size_t>& shape, const T& value)
        : shape_(shape), offset_(0) {
        total_size_ = calculate_total_size_internal(shape_);
        strides_ = calculate_strides_internal(shape_);
         if (total_size_ > 0 || shape_.empty()) {
            data_ptr_ = std::make_shared<std::vector<T>>(total_size_ > 0 ? total_size_ : 1, value);
        } else {
            data_ptr_ = std::make_shared<std::vector<T>>();
        }
        if (shape_.empty() && total_size_ == 1 && data_ptr_->empty()) {
            data_ptr_->resize(1, value);
        }
    }
    // Convenience constructor with initializer_list for shape and a value
    NDArray(std::initializer_list<size_t> shape_list, const T& value)
        : NDArray(std::vector<size_t>(shape_list), value) {}

    // Constructor: creates an array from a shape and flat data (copies data)
    NDArray(const std::vector<size_t>& shape, const std::vector<T>& flat_data)
        : shape_(shape), offset_(0) {
        total_size_ = calculate_total_size_internal(shape_);
        strides_ = calculate_strides_internal(shape_);
        if (flat_data.size() != total_size_) {
            if (!(shape.empty() && total_size_ == 1 && flat_data.size() == 1)) { // Allow scalar init
                 throw std::invalid_argument("Data size does not match shape's total size.");
            }
        }
        data_ptr_ = std::make_shared<std::vector<T>>(flat_data);
         if (shape_.empty() && total_size_ == 1 && data_ptr_->empty() && !flat_data.empty()){
             // This case should be covered by flat_data.size() == total_size_ check
         } else if (shape_.empty() && total_size_ == 1 && data_ptr_->empty() && flat_data.empty()){
             data_ptr_->resize(1); // Default construct the scalar if flat_data is empty for 0-D
         }
    }
     // Convenience constructor with initializer_list for shape and flat_data
    NDArray(std::initializer_list<size_t> shape_list, const std::vector<T>& flat_data)
        : NDArray(std::vector<size_t>(shape_list), flat_data) {}


    // Copy constructor (deep copy)
    NDArray(const NDArray& other)
        : shape_(other.shape_), strides_(other.strides_), offset_(0), total_size_(other.total_size_) {
        if (other.total_size_ > 0 || other.shape_.empty()) {
            data_ptr_ = std::make_shared<std::vector<T>>(other.total_size_ > 0 ? other.total_size_ : 1);
            
            // Efficient copy if 'other' is compact and C-contiguous from its offset
            // For simplicity, using element-wise copy via for_each_index on 'other'
            // This creates a new compact representation.
            size_t current_flat_idx = 0;
            other.for_each_index_iterative([&](const std::vector<size_t>& read_indices) {
                (*data_ptr_)[current_flat_idx++] = other.at(read_indices);
            });
        } else { // Source is empty (e.g. shape (2,0,3))
            data_ptr_ = std::make_shared<std::vector<T>>();
        }
         if (shape_.empty() && total_size_ == 1 && data_ptr_->empty()) {
            // This case implies other.total_size_ was 1 (0-D array)
            // and for_each_index_iterative correctly copied the single element.
            // If other.total_size_ was 0 but shape was empty, it's an issue.
            // The logic above should handle 0-D correctly.
            // If other was 0-D, total_size_ is 1. data_ptr_ is allocated with 1.
            // for_each_index_iterative calls f({}), other.at({}) gets the element.
        }
    }

    // Move constructor
    NDArray(NDArray&& other) noexcept
        : data_ptr_(std::move(other.data_ptr_)),
          shape_(std::move(other.shape_)),
          strides_(std::move(other.strides_)),
          offset_(other.offset_),
          total_size_(other.total_size_) {
        // Reset other to a valid (empty or default) state
        other.shape_.clear();
        other.strides_.clear();
        other.offset_ = 0;
        other.total_size_ = 0; 
        // other.data_ptr_ is already moved from, effectively null or points to moved-from vector
    }

    // Copy assignment (deep copy)
    NDArray& operator=(const NDArray& other) {
        if (this == &other) {
            return *this;
        }
        shape_ = other.shape_;
        strides_ = other.strides_;
        total_size_ = other.total_size_;
        offset_ = 0; // New data block is always at offset 0 for itself

        if (other.total_size_ > 0 || other.shape_.empty()) {
            data_ptr_ = std::make_shared<std::vector<T>>(other.total_size_ > 0 ? other.total_size_ : 1);
            size_t current_flat_idx = 0;
            other.for_each_index_iterative([&](const std::vector<size_t>& read_indices) {
                (*data_ptr_)[current_flat_idx++] = other.at(read_indices);
            });
        } else {
            data_ptr_ = std::make_shared<std::vector<T>>();
        }
        return *this;
    }

    // Move assignment
    NDArray& operator=(NDArray&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        data_ptr_ = std::move(other.data_ptr_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        offset_ = other.offset_;
        total_size_ = other.total_size_;

        other.shape_.clear();
        other.strides_.clear();
        other.offset_ = 0;
        other.total_size_ = 0;
        return *this;
    }

    // Element access
    T& at(const std::vector<size_t>& indices) {
        return (*data_ptr_)[get_flat_index(indices)];
    }
    const T& at(const std::vector<size_t>& indices) const {
        return (*data_ptr_)[get_flat_index(indices)];
    }

    // Convenience element access with initializer_list
    T& operator()(std::initializer_list<size_t> indices_list) {
        return at(std::vector<size_t>(indices_list));
    }
    const T& operator()(std::initializer_list<size_t> indices_list) const {
        return at(std::vector<size_t>(indices_list));
    }
    // For 0-D array access like arr()
    T& operator()() {
        if (ndim() != 0) throw std::invalid_argument("Parameterless operator() only for 0-D arrays.");
        return at({});
    }
    const T& operator()() const {
        if (ndim() != 0) throw std::invalid_argument("Parameterless operator() only for 0-D arrays.");
        return at({});
    }


    // Properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return total_size_; } // Total number of elements

    // Raw data pointer (use with caution, points to start of this view's data in shared block)
    // This is only safe if the view is C-contiguous.
    // For simplicity, we provide access to the underlying shared_vector and offset.
    std::shared_ptr<std::vector<T>> get_data_ptr() const { return data_ptr_; }
    size_t get_offset() const { return offset_; }
    const std::vector<size_t>& get_strides() const { return strides_; }


    // Fill all elements with a value
    void fill(const T& value) {
        if (total_size_ == 0) return;
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            this->at(indices) = value;
        });
    }

    // Reshape: returns a new NDArray that is a view of the original data if possible
    NDArray reshape(const std::vector<size_t>& new_shape) const {
        size_t new_total_size = calculate_total_size_internal(new_shape);
        if (new_total_size != this->total_size_) {
             // Special case: if current total_size_ is 1 (e.g. 0-D or shape {1,1,1})
             // and new_total_size is also 1 (e.g. different 0-D or shape {1}), allow it.
            if (! (this->total_size_ == 1 && new_total_size == 1) ) {
                throw std::invalid_argument("Total size must remain unchanged for reshape.");
            }
        }

        NDArray result; // Create a dummy to be overwritten
        result.data_ptr_ = this->data_ptr_; // Share data
        result.offset_ = this->offset_;     // Share offset
        result.shape_ = new_shape;
        result.total_size_ = new_total_size == 0 && new_shape.empty() ? 1 : new_total_size; // Handle 0-D target
        result.strides_ = calculate_strides_internal(new_shape);
        
        // A check for C-contiguity might be needed here if we want to be strict
        // about when a view is possible. For now, assume any reshape preserving
        // size can be represented by new strides on the same flat data segment.
        return result;
    }
    // Convenience reshape with initializer_list
    NDArray reshape(std::initializer_list<size_t> new_shape_list) const {
        return reshape(std::vector<size_t>(new_shape_list));
    }


    // --- Arithmetic Operators ---
    // NDArray + NDArray
    NDArray operator+(const NDArray& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Arrays must have the same shape for element-wise addition.");
        }
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) + other.at(indices);
        });
        return result;
    }

    // NDArray - NDArray
    NDArray operator-(const NDArray& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Arrays must have the same shape for element-wise subtraction.");
        }
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) - other.at(indices);
        });
        return result;
    }

    // NDArray * NDArray (element-wise)
    NDArray operator*(const NDArray& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Arrays must have the same shape for element-wise multiplication.");
        }
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) * other.at(indices);
        });
        return result;
    }

    // NDArray / NDArray (element-wise)
    NDArray operator/(const NDArray& other) const {
        if (this->shape_ != other.shape_) {
            throw std::invalid_argument("Arrays must have the same shape for element-wise division.");
        }
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            // Could add check for division by zero if T is numeric
            result.at(indices) = this->at(indices) / other.at(indices);
        });
        return result;
    }

    // NDArray + scalar
    NDArray operator+(const T& scalar) const {
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) + scalar;
        });
        return result;
    }
    // scalar + NDArray
    friend NDArray operator+(const T& scalar, const NDArray& arr) {
        return arr + scalar; // Utilize the above operator
    }


    // NDArray - scalar
    NDArray operator-(const T& scalar) const {
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) - scalar;
        });
        return result;
    }
    // scalar - NDArray
    friend NDArray operator-(const T& scalar, const NDArray& arr) {
        NDArray result(arr.shape());
        arr.for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = scalar - arr.at(indices);
        });
        return result;
    }

    // NDArray * scalar
    NDArray operator*(const T& scalar) const {
        NDArray result(this->shape_);
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) * scalar;
        });
        return result;
    }
    // scalar * NDArray
    friend NDArray operator*(const T& scalar, const NDArray& arr) {
        return arr * scalar;
    }

    // NDArray / scalar
    NDArray operator/(const T& scalar) const {
        NDArray result(this->shape_);
        // Could add check for division by zero if T is numeric and scalar is 0
        this->for_each_index_iterative([&](const std::vector<size_t>& indices) {
            result.at(indices) = this->at(indices) / scalar;
        });
        return result;
    }
     // scalar / NDArray
    friend NDArray operator/(const T& scalar, const NDArray& arr) {
        NDArray result(arr.shape());
        arr.for_each_index_iterative([&](const std::vector<size_t>& indices) {
            // Could add check for division by zero if T is numeric
            result.at(indices) = scalar / arr.at(indices);
        });
        return result;
    }


    // Output stream operator
    friend std::ostream& operator<<(std::ostream& os, const NDArray<T>& arr) {
        if (arr.ndim() == 0) { // Scalar
            os << "NDArray(" << arr.at({}) << ")";
            return os;
        }
        if (arr.total_size_ == 0) {
            os << "NDArray([])"; // Or based on shape, e.g. NDArray(shape=(2,0,3)) []
            return os;
        }
        
        // Use a recursive helper or an iterative one for complex N-D printing
        // For simplicity, a basic N-D print:
        os << "NDArray(";
        std::vector<size_t> current_indices(arr.ndim());
        arr.print_recursive(os, current_indices, 0);
        os << ")";
        return os;
    }

    // Static factory methods (like NumPy)
    static NDArray zeros(const std::vector<size_t>& shape) {
        return NDArray(shape, static_cast<T>(0));
    }
    static NDArray zeros(std::initializer_list<size_t> shape_list) {
        return NDArray(std::vector<size_t>(shape_list), static_cast<T>(0));
    }

    static NDArray ones(const std::vector<size_t>& shape) {
        return NDArray(shape, static_cast<T>(1));
    }
    static NDArray ones(std::initializer_list<size_t> shape_list) {
        return NDArray(std::vector<size_t>(shape_list), static_cast<T>(1));
    }
    
    static NDArray full(const std::vector<size_t>& shape, const T& value) {
        return NDArray(shape, value);
    }
    static NDArray full(std::initializer_list<size_t> shape_list, const T& value) {
        return NDArray(std::vector<size_t>(shape_list), value);
    }

    // Create an array with a range of values (similar to np.arange and then reshape)
    // Creates a 1D array [start, stop) with step.
    static NDArray arange(T start, T stop, T step = static_cast<T>(1)) {
        if (step == static_cast<T>(0)) {
            throw std::invalid_argument("Step cannot be zero.");
        }
        std::vector<T> values;
        if (step > 0) {
            for (T val = start; val < stop; val += step) {
                values.push_back(val);
            }
        } else { // step < 0
            for (T val = start; val > stop; val += step) { // val > stop for negative step
                values.push_back(val);
            }
        }
        if (values.empty()) return NDArray<T>({0}); // Empty range results in array of shape (0,)
        return NDArray<T>({values.size()}, values);
    }
};

} // namespace nd

#endif // NDARRAY_HPP
