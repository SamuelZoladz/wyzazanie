#include "algorithm.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>


static double l2_norm_diff(const std::vector<double> &a,
                           const std::vector<double> &b) {
  const size_t n = a.size();
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = a[i] - b[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

std::pair<std::vector<double>, double>
perform_sequential_algorithm(const calc_function_t &calc_value,
                             std::vector<double> starting_x_0, const uint32_t n,
                             const int a, const int b) {
  // Parameters
  const uint32_t L = 30;
  double T = 500.0;
  const double alpha = 0.3;
  const double epsT = 0.1;

  const double cauchy_eps =
      (b - a) * std::sqrt(n / 6.0) *
      1e-3;
  const uint16_t cauchy_max_steps = 10;
  uint16_t cauchy_steps = 0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> U(0.0, 1.0);

  if (starting_x_0.size() != n) {
    starting_x_0.resize(n, 0.0);
  }

  std::vector<double> x0 = std::move(starting_x_0);
  double f_x0 = calc_value(x0, n);

  std::vector<double> xopt = x0;
  double f_opt = f_x0;

  while (T > epsT) {
    for (uint32_t k = 0; k < L; ++k) {

      std::vector<double> x_star(n);
      for (uint32_t i = 0; i < n; ++i) {
        const double s_i = U(gen);
        x_star[i] = static_cast<double>(a) +
                    s_i * (static_cast<double>(b) - static_cast<double>(a));
      }

      const double f_star = calc_value(x_star, n);

      bool accepted = false;
      double step_norm = 0.0;


      if (f_star < f_x0) {
        accepted = true;
        if (cauchy_eps > 0.0) {
          step_norm = l2_norm_diff(x0, x_star);
        }

        x0 = x_star;
        f_x0 = f_star;

        if (f_star < f_opt) {
          xopt = x_star;
          f_opt = f_star;
        }
      } else {

        const double r = U(gen);

        if (r < std::exp((f_x0 - f_star) / T)) {
          accepted = true;
          if (cauchy_eps > 0.0) {
            step_norm = l2_norm_diff(x0, x_star);
          }

          x0 = x_star;
          f_x0 = f_star;
        }
      }

      // Cauch Criterion
      if (cauchy_eps > 0.0 && accepted) {
        if (step_norm < cauchy_eps) {
          cauchy_steps++;
        } else {
          cauchy_steps = 0;
        }
        if (cauchy_steps > cauchy_max_steps) {
          std::cout << std::endl
                    << "Quitting algorithm due to Cauchy criterion"
                    << std::endl;
          return {xopt, f_opt};
        }
      }
    }


    T *= (1.0 - alpha);
  }

  return {xopt, f_opt};
}
#include <limits>
#include <mutex>
#include <thread>


std::pair<std::vector<double>, double>
perform_parallel_algorithm_threads(const calc_function_t &calc_value,
                                   const std::vector<double> &starting_x_0,
                                   uint32_t n, int a, int b, int num_threads) {
  // Parameters
  const uint32_t L = 30;
  double T = 500.0;
  const double alpha = 0.3;
  const double epsT = 0.1;

  const double cauchy_eps = (b - a) * std::sqrt(n / 6.0) * 1e-3;
  const uint16_t cauchy_max_steps = 10;
  uint16_t cauchy_steps = 0;

  std::vector<double> x0 = starting_x_0;
  if (x0.size() != n)
    x0.resize(n, 0.0);
  double f_x0 = calc_value(x0, n);

  std::vector<double> xopt = x0;
  double f_opt = f_x0;


  std::vector<std::mt19937> gens;
  for (int i = 0; i < num_threads; ++i) {
    gens.emplace_back(std::random_device{}());
  }
  std::uniform_real_distribution<double> U(0.0, 1.0);

  while (T > epsT) {

    for (uint32_t k = 0; k < L; k += num_threads) {
      std::vector<std::thread> threads;
      std::vector<std::vector<double>> x_stars(num_threads,
                                               std::vector<double>(n));
      std::vector<double> f_stars(num_threads);
      std::vector<bool> thread_done(num_threads, false);

      for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {

          for (uint32_t i = 0; i < n; ++i) {
            x_stars[t][i] =
                static_cast<double>(a) +
                U(gens[t]) * (static_cast<double>(b) - static_cast<double>(a));
          }
          f_stars[t] = calc_value(x_stars[t], n);
          thread_done[t] = true;
        });
      }

      for (auto &th : threads)
        th.join();


      bool accepted_in_round = false;
      double best_step_norm = 0.0;

      for (int t = 0; t < num_threads; ++t) {
        bool thread_accepted = false;
        if (f_stars[t] < f_x0) {
          thread_accepted = true;
        } else {
          const double r =
              U(gens[0]);
          if (r < std::exp((f_x0 - f_stars[t]) / T)) {
            thread_accepted = true;
          }
        }

        if (thread_accepted) {
          accepted_in_round = true;

          if (f_stars[t] < f_x0) {
            double step_norm = l2_norm_diff(x0, x_stars[t]);
            x0 = x_stars[t];
            f_x0 = f_stars[t];
            best_step_norm = step_norm;

            if (f_x0 < f_opt) {
              xopt = x0;
              f_opt = f_x0;
            }
          }
        }
      }


      if (cauchy_eps > 0.0) {
        if (accepted_in_round) {
          if (best_step_norm < cauchy_eps)
            cauchy_steps++;
          else
            cauchy_steps = 0;
        }
        if (cauchy_steps > cauchy_max_steps) {
          std::cout << "Quitting algorithm due to Cauchy criterion"
                    << std::endl;
          return {xopt, f_opt};
        }
      }
    }
    T *= (1.0 - alpha);
  }
  return {xopt, f_opt};
}
