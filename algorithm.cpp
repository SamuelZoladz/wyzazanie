#include "algorithm.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>



// Nie ma sprawdzania czy n jest poprawne, jeżeli nie jest to powinniśmy się wywalić;
std::pair<std::vector<double>, double> perform_sequential_algorithm(const calc_function_t& calc_value,
                                                                    std::vector<double> starting_x_0, const uint32_t n,
                                                                    const int a, const int b)
{
    // Step 1
    uint16_t k = 0;
    const uint16_t iteration_limit = 30;
    double temperature = 500;
    const double alpha = 0.3;
    const double epsilon = 0.1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(0, 1);

    std::vector<double> x_0 = starting_x_0;

    std::vector<double> x_optimal = x_0;
    while (true)
    {
        // Step 2
        k = k + 1;
        std::vector<double> x_prim = {};
        x_prim.resize(n);

        for (int i = 0; i < n; ++i)
        {
            const double s_i = (distrib(gen));
            x_prim[i] = s_i * (b - a) + a;
        }

        // Step 3
        const auto f_of_x_prim = calc_value(x_prim, n);
        const auto f_of_x_0 = calc_value(x_prim, n);
        if (f_of_x_prim < f_of_x_0)
        {
            std::swap(x_0, x_prim);
            std::swap(x_optimal, x_prim);
        }
        else
        {
            // Step 4
            double r = distrib(gen);
            if (r < std::exp((f_of_x_0 - f_of_x_prim) / temperature))
            {
                std::swap(x_0, x_prim);
            }
        }
        // Step 5
        if (k > iteration_limit)
        {
            // Step 6
            temperature = temperature * (1 - alpha);
            if (temperature > epsilon)
            {
                k = 0;
            }
            else
            {
                break;
            }
        }
    }

    return std::make_pair(x_optimal, calc_value(x_optimal, n));
}
