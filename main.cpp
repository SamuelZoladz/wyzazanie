#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "functions.h"
#include "algorithm.h"


int main()
{

    // Quadratic equation
    uint32_t n = 200000;
    std::vector<double> x_0 = make_quadratic_x0(n);
    auto start_q = std::chrono::high_resolution_clock::now();

    auto quadratic_result = perform_sequential_algorithm(calc_quadratic_function, x_0, n, -5, 5);

    auto end_q = std::chrono::high_resolution_clock::now();
    auto dur_q = std::chrono::duration_cast<std::chrono::milliseconds>( end_q - start_q ).count();
    std::cout << "Sequential execution time for quadratic equation: " << dur_q << std::endl;

    // Woods equation
    n = 100000;
    x_0 = make_woods_x0(n);
    auto start_woods = std::chrono::high_resolution_clock::now();

    auto woods_result = perform_sequential_algorithm(calc_woods_function, x_0, n, -5, 5);

    auto end_woods = std::chrono::high_resolution_clock::now();
    auto dur_woods = std::chrono::duration_cast<std::chrono::milliseconds>( end_woods - start_woods ).count();

    std::cout << "Sequential execution time for woods equation: " << dur_woods << std::endl;

    // Powell Singular equation
    n = 100000;
    x_0 = make_powell_x0(n);

    auto start_powell = std::chrono::high_resolution_clock::now();

    auto powell_singular_result = perform_sequential_algorithm(calc_powell_singular_function, x_0, n, -4, 4);

    auto end_powell = std::chrono::high_resolution_clock::now();
    auto dur_powell = std::chrono::duration_cast<std::chrono::milliseconds>( end_powell - start_powell ).count();

    std::cout << "Sequential execution time for powell equation: " << dur_powell << std::endl;

    return 0;
}
