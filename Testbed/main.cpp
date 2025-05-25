// Example Usage (in a .cpp file)
#include <iostream>
#include <ndarray.hpp>

int main() {
    // Create a 2x3 array of integers, initialized with 0
    nd::NDArray<int> arr1 = nd::NDArray<int>::zeros({2, 3});
    std::cout << "arr1 (zeros):\n" << arr1 << std::endl;

    // Fill with a value
    arr1.fill(5);
    std::cout << "arr1 (filled with 5):\n" << arr1 << std::endl;

    // Create from shape and value
    nd::NDArray<double> arr2({2, 2}, 3.14);
    std::cout << "arr2 (2x2 filled with 3.14):\n" << arr2 << std::endl;

    // Element access
    arr2.at({0, 1}) = 1.618;
    arr2({1,0}) = 2.718; // Using operator()
    std::cout << "arr2 (modified):\n" << arr2 << std::endl;
    std::cout << "Element (0,1) of arr2: " << arr2({0,1}) << std::endl;

    // Reshape (creates a view)
    nd::NDArray<double> arr2_reshaped = arr2.reshape({4, 1});
    std::cout << "arr2_reshaped (4x1 view of arr2):\n" << arr2_reshaped << std::endl;
    
    // Modifying view affects original (if data is shared)
    arr2_reshaped.at({2,0}) = 9.99;
    std::cout << "arr2 (after modifying reshaped view at logical index (2,0) -> original (1,0)):\n" << arr2 << std::endl;
    std::cout << "arr2_reshaped (confirming modification):\n" << arr2_reshaped << std::endl;


    // Arithmetic
    nd::NDArray<int> a({2, 2}, 10);
    nd::NDArray<int> b({2, 2}, 2);
    nd::NDArray<int> c = a + b;
    std::cout << "c = a + b:\n" << c << std::endl;
    nd::NDArray<int> d = a * 5;
    std::cout << "d = a * 5:\n" << d << std::endl;
    nd::NDArray<int> e = 3 * a;
     std::cout << "e = 3 * a:\n" << e << std::endl;


    // Arange and reshape
    nd::NDArray<int> range_arr = nd::NDArray<int>::arange(0, 12).reshape({3,4});
    std::cout << "range_arr (arange(0,12).reshape({3,4})):\n" << range_arr << std::endl;

    // 0-D (scalar) array
    nd::NDArray<float> scalar_arr(7.5f); // Implicitly { } shape, value 7.5f
    // Note: The default constructor NDArray() makes an uninitialized 0-D array.
    // To make a 0-D array with a specific value directly, you can use:
    nd::NDArray<float> scalar_arr_explicit({}, 7.5f);
    std::cout << "scalar_arr_explicit:\n" << scalar_arr_explicit << std::endl;
    std::cout << "Value of scalar_arr_explicit: " << scalar_arr_explicit() << std::endl;
    nd::NDArray<float> scalar_arr_reshaped = scalar_arr_explicit.reshape({1,1,1});
    std::cout << "scalar_arr_reshaped from 0-D:\n" << scalar_arr_reshaped << std::endl;

    try {
        nd::NDArray<int> err_arr({2,2});
        // err_arr.at({2,0}); // This would throw std::out_of_range
        // nd::NDArray<int> err_sum = err_arr + nd::NDArray<int>({2,3}); // This would throw std::invalid_argument
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}