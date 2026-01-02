#include "algorithm.h"
#include "functions.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

int main(int argc, char *argv[]) {

  // Inicjalizacja MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const uint32_t n = 800000;

  // Równanie kwadratowe
  if (rank == 0) {
    std::cout << "=== Testing Quadratic Function ===" << std::endl;
    std::cout << "n = " << n << ", processes = " << size << std::endl;
  }

  // RK wersja sekwencyjna - realizacja tylko w procesie z rank 0
  if (rank == 0) {
    std::vector<double> x_0 = make_quadratic_x0(n);
    auto start_q = std::chrono::high_resolution_clock::now();

    auto quadratic_result =
        perform_sequential_algorithm(calc_quadratic_function, x_0, n, -5, 5);

    auto end_q = std::chrono::high_resolution_clock::now();
    auto dur_q =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_q - start_q)
            .count();
    std::cout << "Sequential execution time: " << dur_q
              << "ms, Euclidean norm: " << l2_norm(quadratic_result.first)
              << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // RK wersja równoległa - realizacja w wszystkich procesach
  {
    std::vector<double> x_0 = make_quadratic_x0(n);

    auto start_p = std::chrono::high_resolution_clock::now();

    auto parallel_result = perform_parallel_algorithm(
        calc_quadratic_function_partial, x_0, n, -5, 5, 1);

    auto end_p = std::chrono::high_resolution_clock::now();
    auto dur_p =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();

    if (rank == 0) {
      std::cout << "Parallel execution time:   " << dur_p
                << "ms, Euclidean norm: " << l2_norm(parallel_result.first)
                << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Równanie Woodsa
  if (rank == 0) {
    std::cout << "\n=== Testing Woods Function ===" << std::endl;
  }

  // RW wersja sekwencyjna - realizacja tylko w procesie z rank 0
  if (rank == 0) {
    std::vector<double> x_0 = make_woods_x0(n);
    auto start_w = std::chrono::high_resolution_clock::now();

    auto woods_result =
        perform_sequential_algorithm(calc_woods_function, x_0, n, -5, 5);

    auto end_w = std::chrono::high_resolution_clock::now();
    auto dur_w =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_w - start_w)
            .count();
    std::cout << "Sequential execution time: " << dur_w
              << "ms, Distance to minimum: "
              << l2_norm_distance_to_woods_min(woods_result.first, n)
              << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // RW wersja równoległa - realizacja w wszystkich procesach
  {
    std::vector<double> x_0 = make_woods_x0(n);

    auto start_p = std::chrono::high_resolution_clock::now();

    auto parallel_result = perform_parallel_algorithm(
        calc_woods_function_partial, x_0, n, -5, 5, 4); // block_alignment = 4

    auto end_p = std::chrono::high_resolution_clock::now();
    auto dur_p =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();

    if (rank == 0) {
      std::cout << "Parallel execution time:   " << dur_p
                << "ms, Distance to minimum: "
                << l2_norm_distance_to_woods_min(parallel_result.first, n)
                << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Równanie Powella (osobliwe)
  if (rank == 0) {
    std::cout << "\n=== Testing Powell Singular Function ===" << std::endl;
  }

  // RP wersja sekwencyjna - realizacja tylko w procesie z rank 0
  if (rank == 0) {
    std::vector<double> x_0 = make_powell_x0(n);
    auto start_p = std::chrono::high_resolution_clock::now();

    auto powell_result = perform_sequential_algorithm(
        calc_powell_singular_function, x_0, n, -4, 4);

    auto end_p = std::chrono::high_resolution_clock::now();
    auto dur_p =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();
    std::cout << "Sequential execution time: " << dur_p
              << "ms, Distance to minimum: "
              << l2_norm_distance_to_powell_min(powell_result.first, n)
              << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // RP wersja równoległa - realizacja w wszystkich procesach
  {
    std::vector<double> x_0 = make_powell_x0(n);

    auto start_p = std::chrono::high_resolution_clock::now();

    auto parallel_result =
        perform_parallel_algorithm(calc_powell_singular_function_partial, x_0,
                                   n, -4, 4, 4); // block_alignment = 4

    auto end_p = std::chrono::high_resolution_clock::now();
    auto dur_p =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();

    if (rank == 0) {
      std::cout << "Parallel execution time:   " << dur_p
                << "ms, Distance to minimum: "
                << l2_norm_distance_to_powell_min(parallel_result.first, n)
                << std::endl;
    }
  }

  // Zakończenie środowiska MPI
  MPI_Finalize();
  return 0;
}
