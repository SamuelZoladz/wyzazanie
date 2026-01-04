#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>
#include "algorithm.h"
#include "functions.h"


using clock_type = std::chrono::high_resolution_clock;

void run_test(const std::string &name, const calc_function_t &func,
              std::vector<double> (*make_x0)(uint32_t),
              double (*norm_func)(const std::vector<double> &, uint32_t),
              uint32_t n, int a, int b) {
    std::cout << "\n===== " << name << " =====\n";

    //SEQUENTIAL
    omp_set_num_threads(1);
    auto x0 = make_x0(n);

    auto t1_start = clock_type::now();
    auto result_seq = perform_sequential_algorithm(func, x0, n, a, b);
    auto t1_end = clock_type::now();

    double T1 = std::chrono::duration<double>(t1_end - t1_start).count();
    std::cout << "SEQ time: " << T1 << " s"
              << ", norm = " << norm_func(result_seq.first, n) << "\n";

    //PARALLEL
    for (int threads : {2, 4}) {
        omp_set_num_threads(threads);

        x0 = make_x0(n);
        auto tp_start = clock_type::now();
        auto result_par =
            perform_parallel_algorithm_threads(func, x0, n, a, b, threads);
        auto tp_end = clock_type::now();

        double Tp = std::chrono::duration<double>(tp_end - tp_start).count();
        double speedup = T1 / Tp;

        std::cout << "Threads: " << threads << " | time: " << Tp << " s"
                  << " | speedup S(n,p) = " << speedup << "\n";
    }
}

int main() {
    
    const uint32_t n = 800000;

    run_test(
        "QUADRATIC FUNCTION", calc_quadratic_function, make_quadratic_x0,
        [](const std::vector<double> &x, uint32_t) { return l2_norm(x); }, n, -5,
        5);

    run_test("WOODS FUNCTION", calc_woods_function, make_woods_x0,
             l2_norm_distance_to_woods_min, n, -5, 5);

    run_test("POWELL SINGULAR FUNCTION", calc_powell_singular_function,
             make_powell_x0, l2_norm_distance_to_powell_min, n, -4, 4);

    return 0;
}
