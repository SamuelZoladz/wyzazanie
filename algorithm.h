#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

typedef std::function<double(const std::vector<double> &, uint32_t)>
    calc_function_t;
typedef std::function<double(const std::vector<double> &, uint32_t, uint32_t)>
    calc_function_partial_t;

// Struktura pamięci współdzielonej dla synchronizacji i danych wspólnych
struct SharedState {
  // Synchronizacja
  volatile long barrier_count;
  volatile long barrier_generation;
  volatile long exit_flag;

  // Dane wspólne
  double f_x0;
  double f_star;
  double global_norm_sq;
  int accepted;
  int cauchy_steps;

  // Redukcja (sumowanie lokalnych wartości)
  volatile double local_contributions[64];
};

std::pair<std::vector<double>, double>
perform_sequential_algorithm(const calc_function_t &calc_value,
                             std::vector<double> starting_x_0, uint32_t n,
                             int a, int b, bool debug);

std::pair<std::vector<double>, double> perform_parallel_algorithm_win(
    const calc_function_partial_t &calc_value_partial,
    std::vector<double> starting_x_0, uint32_t n, int a, int b,
    uint32_t block_alignment, bool debug, int rank, int size,
    SharedState *shared_state, double *full_x_ptr);

#endif // ALGORITHM_H
