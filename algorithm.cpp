#include "algorithm.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

// Norma euklidesowa różnicy dwóch wektorów (kryterium Cauchy'ego)
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

// Symulowane wyżarzanie (wersja sekwencyjna).
// Uwaga: generowanie x* jest globalne (jednostajnie w [a,b]^n), bo tak jest w
// treści zadania.
std::pair<std::vector<double>, double>
perform_sequential_algorithm(const calc_function_t &calc_value,
                             std::vector<double> starting_x_0, const uint32_t n,
                             const int a, const int b) {
  // Krok 1: parametry (wg propozycji z treści)
  const uint32_t L = 30;
  double T = 500.0;
  const double alpha = 0.3;
  const double epsT = 0.1;

  const double cauchy_eps =
      (b - a) * std::sqrt(n / 6.0) * 1e-3; // 1000 razy mniejsz niż krok
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
      // Krok 2: losowanie x*
      std::vector<double> x_star(n);
      for (uint32_t i = 0; i < n; ++i) {
        const double s_i = U(gen);
        x_star[i] = static_cast<double>(a) +
                    s_i * (static_cast<double>(b) - static_cast<double>(a));
      }

      const double f_star = calc_value(x_star, n);

      bool accepted = false;
      double step_norm = 0.0;

      // Krok 3
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
        // Krok 4
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

      // Kryterium Cauchy'ego
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

    // Krok 6
    T *= (1.0 - alpha);
  }

  return {xopt, f_opt};
}

// Symulowane wyżarzanie (wersja równoległa).
// Przyjęta implementacja, podobnie jak wersja sekwencyjna sprawdza jednego
// kandydata naraz. Pierwotnie losowano x* globalnie i przesyłano do każdego
// procesu - było to bardzo kosztowne. Teraz każdy proces losuje swój fragment
// x*. Funkcje testujące opierają się na sumowaniu wartości częściowych
// wartości. Stąd każdy proces może obliczyć sumę swojej części wektora x* i
// zwrócić ją do procesu głównego. Proces główny następnie ocenia kandydata.
std::pair<std::vector<double>, double>
perform_parallel_algorithm(const calc_function_partial_t &calc_value_partial,
                           std::vector<double> starting_x_0, const uint32_t n,
                           const int a, const int b,
                           const uint32_t block_alignment) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Parametry algorytmu (takie same jak w wersji sekwencyjnej)
  const uint32_t L = 30;
  double T = 500.0;
  const double alpha = 0.3;
  const double epsT = 0.1;

  const double cauchy_eps = (b - a) * std::sqrt(n / 6.0) * 1e-3;
  const uint16_t cauchy_max_steps = 10;
  uint16_t cauchy_steps = 0;

  // Podział danych z wyrównaniem do bloków (funkcje Woodsa i Powella obliczane
  // są w blokach o rozmiarze 4)
  const uint32_t total_blocks = n / block_alignment;
  const uint32_t blocks_per_proc = total_blocks / static_cast<uint32_t>(size);
  const uint32_t remainder_blocks = total_blocks % static_cast<uint32_t>(size);

  // Każdy proces otrzymuje blocks_per_proc bloków, a pierwsze remainder_blocks
  // procesów dostaje po 1 dodatkowym bloku.
  uint32_t local_blocks =
      blocks_per_proc +
      (static_cast<uint32_t>(rank) < remainder_blocks ? 1 : 0);
  uint32_t local_n = local_blocks * block_alignment;

  // Obliczenie indeksu startowego dla procesu
  uint32_t global_start = 0;
  for (int r = 0; r < rank; ++r) {
    uint32_t r_blocks =
        blocks_per_proc + (static_cast<uint32_t>(r) < remainder_blocks ? 1 : 0);
    global_start += r_blocks * block_alignment;
  }

  // Przygotowanie generatorów dla każdego procesu (wspólny seed przesunięty o
  // rank procesu)
  const uint64_t base_seed = 42;
  std::mt19937 gen(base_seed + static_cast<uint64_t>(rank));
  std::uniform_real_distribution<double> U(0.0, 1.0);

  // Przygotowanie generatora dla procesu 0 na potrzeby akceptacji
  std::mt19937 decision_gen(base_seed + 1000);
  std::uniform_real_distribution<double> U_decision(0.0, 1.0);

  // Lokalne wektory (tylko fragment danych)
  std::vector<double> local_x0(local_n);
  std::vector<double> local_xopt(local_n);

  // Inicjalizacja z punktu startowego - innego dla każdego procesu
  if (starting_x_0.size() >= global_start + local_n) {
    for (uint32_t i = 0; i < local_n; ++i) {
      local_x0[i] = starting_x_0[global_start + i];
    }
  } else {
    std::fill(local_x0.begin(), local_x0.end(), 0.0);
  }
  local_xopt = local_x0;

  // Obliczenie początkowego kosztu lokalnego i zsumowanie globalnie
  double local_cost = calc_value_partial(local_x0, local_n, global_start);
  double f_x0 = 0.0;
  MPI_Allreduce(&local_cost, &f_x0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double f_opt = f_x0;

  while (T > epsT) {
    for (uint32_t k = 0; k < L; ++k) {
      // Obliczenie przez każdy proces swojej części x*
      std::vector<double> local_x_star(local_n);
      for (uint32_t i = 0; i < local_n; ++i) {
        const double s_i = U(gen);
        local_x_star[i] =
            static_cast<double>(a) +
            s_i * (static_cast<double>(b) - static_cast<double>(a));
      }

      // Obliczenie lokalnego kosztu i zsumowanie globalnie
      double local_f_star =
          calc_value_partial(local_x_star, local_n, global_start);
      double f_star = 0.0;
      MPI_Allreduce(&local_f_star, &f_star, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      // Obliczenie lokalnej normy różnicy dla kryterium Cauchy'ego
      double local_norm_sq = 0.0;
      for (uint32_t i = 0; i < local_n; ++i) {
        double d = local_x0[i] - local_x_star[i];
        local_norm_sq += d * d;
      }
      double global_norm_sq = 0.0;
      MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      double step_norm = std::sqrt(global_norm_sq);

      // Proces 0 decyduje o akceptacji i przekazuje decyzję do wszystkich
      // procesów
      int accepted = 0;
      if (rank == 0) {
        if (f_star < f_x0) {
          accepted = 1;
        } else {
          const double r = U_decision(decision_gen);
          if (r < std::exp((f_x0 - f_star) / T)) {
            accepted = 1;
          }
        }
      }
      MPI_Bcast(&accepted, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // Aktualizacja lokalnych danych jeśli zaakceptowano kandydata
      if (accepted) {
        local_x0 = local_x_star;
        f_x0 = f_star;

        if (f_star < f_opt) {
          local_xopt = local_x_star;
          f_opt = f_star;
        }

        // Kryterium Cauchy'ego
        if (cauchy_eps > 0.0) {
          if (step_norm < cauchy_eps) {
            cauchy_steps++;
          } else {
            cauchy_steps = 0;
          }

          if (cauchy_steps > cauchy_max_steps) {
            if (rank == 0) {
              std::cout
                  << std::endl
                  << "Quitting algorithm due to Cauchy criterion (parallel)"
                  << std::endl;
            }
            goto finalize;
          }
        }
      }
    }
    T *= (1.0 - alpha);
  }

finalize:
  // Złożenie pełnego wektora xopt na procesie 0 za pomocą MPI_Gatherv
  std::vector<double> full_xopt;
  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  // Zebranie rozmiarów lokalnych na wszystkich procesach
  int local_n_int = static_cast<int>(local_n);
  MPI_Allgather(&local_n_int, 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                MPI_COMM_WORLD);

  // Obliczenie przesunięć
  displs[0] = 0;
  for (int r = 1; r < size; ++r) {
    displs[r] = displs[r - 1] + recvcounts[r - 1];
  }

  if (rank == 0) {
    full_xopt.resize(n);
  }

  MPI_Gatherv(local_xopt.data(), local_n_int, MPI_DOUBLE, full_xopt.data(),
              recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return {full_xopt, f_opt};
}
