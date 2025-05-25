
# NDArray (Header-only C++ N-Dimensional Array Library)

`NDArray<T>` is a header-only, flexible, N-dimensional array class in C++ designed to mimic NumPy-like behavior with modern C++ standards. It supports scalar (0-D) arrays, basic indexing, iteration, and shape/stride handling.

## Features

- 0-D to N-D arrays (any rank)
- C-style row-major storage
- Shared memory with copy-on-write support
- Indexing with bounds checking
- Flattened memory layout and offset-based views
- Pretty printing for nested arrays
- Fully implemented in a single header

## Usage

### Include
```cpp
#include "ndarray.hpp"
```

### Construction Examples

```cpp
nd::NDArray<int> scalar(42);                      // 0-D scalar
nd::NDArray<float> array({2, 3});                 // 2x3 uninitialized array
nd::NDArray<double> filled({2, 2}, 3.14);         // 2x2 initialized with 3.14
std::vector<int> flat_data = {1,2,3,4,5,6};
nd::NDArray<int> from_flat({2,3}, flat_data);     // Construct from flat vector
```

### Accessing Elements

```cpp
nd::NDArray<int> arr({2, 2}, 5);
arr.at({0, 1}) = 10;
int value = arr.at({0, 1});
```

### Printing

```cpp
std::cout << arr << std::endl;
```

### Iteration

```cpp
arr.for_each_index_iterative([](const std::vector<size_t>& indices) {
    // Access element using arr.at(indices);
});
```

## API Reference

### Constructors

- `NDArray()` — Creates a default 0-D array.
- `NDArray(const T& scalar_value)` — Scalar constructor.
- `NDArray(const std::vector<size_t>& shape)` — Shape-only constructor.
- `NDArray(const std::vector<size_t>& shape, const T& value)` — Initialized constructor.
- `NDArray(const std::vector<size_t>& shape, const std::vector<T>& flat_data)` — Flat data constructor.

### Methods

- `T& at(const std::vector<size_t>& indices)` — Access element by index (with bounds checking).
- `size_t ndim() const` — Number of dimensions.
- `const std::vector<size_t>& shape() const` — Returns shape.
- `const std::vector<size_t>& strides() const` — Returns strides.
- `size_t total_size() const` — Returns number of elements.

## License

This project is free to use under the MIT License.

---

Made with ❤️ in C++
