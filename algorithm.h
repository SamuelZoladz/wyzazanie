#ifndef SIMULATED_ANNEALING_H
#define SIMULATED_ANNEALING_H

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

typedef std::function<double(const std::vector<double> &, uint32_t)>
    calc_function_t;

typedef std::function<double(const std::vector<double> &, uint32_t, uint32_t)>
    calc_function_partial_t;

std::pair<std::vector<double>, double>
perform_sequential_algorithm(const calc_function_t &calc_value,
                             std::vector<double> starting_x_0, const uint32_t n,
                             const int a, const int b);

std::pair<std::vector<double>, double>
perform_parallel_algorithm(const calc_function_partial_t &calc_value_partial,
                           std::vector<double> starting_x_0, const uint32_t n,
                           const int a, const int b,
                           const uint32_t block_alignment = 1);

#endif // SIMULATED_ANNEALING_H
