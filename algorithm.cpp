#include "algorithm.h"

#include <cmath>
#include <random>
#include <vector>

// Norma euklidesowa różnicy dwóch wektorów (kryterium Cauchy'ego)
static double l2_norm_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    const size_t n = a.size();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// Symulowane wyżarzanie (wersja sekwencyjna).
// Uwaga: generowanie x* jest globalne (jednostajnie w [a,b]^n), bo tak jest w treści zadania.
std::pair<std::vector<double>, double> perform_sequential_algorithm(const calc_function_t& calc_value,
                                                                    std::vector<double> starting_x_0,
                                                                    const uint32_t n,
                                                                    const int a,
                                                                    const int b)
{
    // Krok 1: parametry (wg propozycji z treści)
    const uint32_t L = 30;
    double T = 500.0;
    const double alpha = 0.3;
    const double epsT = 0.1;

    const double cauchy_eps = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> U(0.0, 1.0);

    if (starting_x_0.size() != n)
    {
        starting_x_0.resize(n, 0.0);
    }

    std::vector<double> x0 = std::move(starting_x_0);
    double f_x0 = calc_value(x0, n);

    std::vector<double> xopt = x0;
    double f_opt = f_x0;

    while (T > epsT)
    {
        for (uint32_t k = 0; k < L; ++k)
        {
            // Krok 2: losowanie x*
            std::vector<double> x_star(n);
            for (uint32_t i = 0; i < n; ++i)
            {
                const double s_i = U(gen);
                x_star[i] = static_cast<double>(a) + s_i * (static_cast<double>(b) - static_cast<double>(a));
            }

            const double f_star = calc_value(x_star, n);

            bool accepted = false;
            double step_norm = 0.0;

            // Krok 3
            if (f_star < f_x0)
            {
                accepted = true;
                if (cauchy_eps > 0.0) step_norm = l2_norm_diff(x0, x_star);

                x0 = x_star;
                f_x0 = f_star;

                if (f_star < f_opt)
                {
                    xopt = x_star;
                    f_opt = f_star;
                }
            }
            else
            {
                // Krok 4
                const double r = U(gen);
                const double p = std::exp((f_x0 - f_star) / T);

                if (r < p)
                {
                    accepted = true;
                    if (cauchy_eps > 0.0) step_norm = l2_norm_diff(x0, x_star);

                    x0 = x_star;
                    f_x0 = f_star;
                }
            }

            // kryterium Cauchy'ego
            if (cauchy_eps > 0.0 && accepted)
            {
                if (step_norm < cauchy_eps)
                {
                    return {xopt, f_opt};
                }
            }
        }

        // Krok 6
        T *= (1.0 - alpha);
    }

    return {xopt, f_opt};
}
