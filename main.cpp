#include "algorithm.h"
#include "functions.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <windows.h>

// Przyjęcie nazwy pamięci współdzielonej - typ Local
const char *SHM_NAME = "Local\\SA_SharedMemory";

// Funkcja tworząca procesy potomne
// W Windows nie ma funkcji fork. Zamiast tego tworzymy nowe kopie niniejszej
// aplikacji poprzez wielokrotne uruchamianie pliku .exe za pomocą funkcji
// CreateProcess. Każda z instancji wie jakie jest jej zadanie poprzez
// przekazaną do niej wartość rank i size.
void spawn_processes(int size, uint64_t seed, const std::string &exe_path) {
  for (int i = 1; i < size; ++i) {
    std::string cmd = exe_path + " --rank " + std::to_string(i) + " --size " +
                      std::to_string(size) + " --seed " + std::to_string(seed);
    // Parametry do funkcji CreateProcess wymagane w środowisku Windows
    STARTUPINFOA si = {sizeof(si)};
    PROCESS_INFORMATION pi;

    // Tworzenie nowego procesu
    if (!CreateProcessA(NULL, (char *)cmd.c_str(), NULL, NULL, FALSE, 0, NULL,
                        NULL, &si, &pi)) {
      std::cerr << "Failed to spawn process " << i
                << ". Error: " << GetLastError() << std::endl;
    } else {
      // Zamykanie "uchwytów" procesu i wątku - nie są potrzebne. Nie usuwa to
      // samych procesów i wątków.
      CloseHandle(pi.hProcess);
      CloseHandle(pi.hThread);
    }
  }
}

int main(int argc, char *argv[]) {
  int rank = 0;
  int size = 4;
  uint64_t base_seed = 0;
  bool seed_provided = false;

  // Parsowanie argumentów
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--rank" && i + 1 < argc)
      rank = std::stoi(argv[++i]);
    if (arg == "--size" && i + 1 < argc)
      size = std::stoi(argv[++i]);
    if (arg == "--seed" && i + 1 < argc) {
      base_seed = std::stoull(argv[++i]);
      seed_provided = true;
    }
  }

  // Jeśli rank 0 i nie podano ziarna, wygeneruj losowe
  if (rank == 0 && !seed_provided) {
    std::random_device rd;
    base_seed = rd();
  }

  const uint32_t n = 800000;
  const bool debug = false;

  // Pamięć współdzielona
  // Rozmiar pamięci współdzielonej na shared state i wektor x_opt
  size_t shm_size = sizeof(SharedState) + n * sizeof(double);

  HANDLE hMapFile;
  if (rank == 0) {
    // Stworzenie obszaru pamięci w pliku stronicowania z uprawnieniami do
    // odczytu i zapisu
    hMapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
                                  (DWORD)shm_size, SHM_NAME);
  } else {
    // Otwarcie obszaru pamięci w pliku stronicowania
    hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, SHM_NAME);
  }

  if (hMapFile == NULL) {
    std::cerr << "Could not map file. Error: " << GetLastError() << std::endl;
    return 1;
  }

  // Stworzenie wskaźnika do obszaru pamięci współdzielonej
  void *pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, shm_size);
  if (pBuf == NULL) {
    std::cerr << "Could not map view of file. Error: " << GetLastError()
              << std::endl;
    CloseHandle(hMapFile);
    return 1;
  }

  // Stworzenie wskaźników, które organizują pamięć współdzieloną
  SharedState *shared_state = (SharedState *)pBuf;
  double *full_x_ptr = (double *)((char *)pBuf + sizeof(SharedState));

  if (rank == 0) {
    // Reset stanu
    memset(shared_state, 0, sizeof(SharedState));

    // Tworzenie pozostałych procesów
    spawn_processes(size, base_seed, argv[0]);
  }

  // Równanie kwadratowe
  if (rank == 0) {
    std::cout << "=== Testing Quadratic Function ===" << std::endl;
    std::cout << "n = " << n << ", processes = " << size << std::endl;
  }

  // RK wersja sekwencyjna
  long long dur_q_quadratic = 0;
  if (rank == 0) {
    std::vector<double> x_0 = make_quadratic_x0(n);
    auto start_q = std::chrono::high_resolution_clock::now();
    auto res = perform_sequential_algorithm(calc_quadratic_function, x_0, n, -5,
                                            5, base_seed, debug);
    auto end_q = std::chrono::high_resolution_clock::now();
    dur_q_quadratic =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_q - start_q)
            .count();
    std::cout << "Sequential execution time: " << dur_q_quadratic
              << "ms, Euclidean norm: " << l2_norm(res.first) << std::endl;
  }

  // RK wersja równoległa
  long long dur_p_quadratic = 0;
  {
    std::vector<double> x_0 = make_quadratic_x0(n);
    auto start_p = std::chrono::high_resolution_clock::now();

    auto parallel_result = perform_parallel_algorithm_win(
        calc_quadratic_function_partial, x_0, n, -5, 5, 1, base_seed, debug,
        rank, size, shared_state, full_x_ptr);

    auto end_p = std::chrono::high_resolution_clock::now();
    dur_p_quadratic =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();
    if (rank == 0) {
      std::cout << "Parallel execution time:   " << dur_p_quadratic
                << "ms, Euclidean norm: " << l2_norm(parallel_result.first)
                << std::endl;
      if (dur_p_quadratic > 0) {
        std::cout << "Speedup: " << (double)dur_q_quadratic / dur_p_quadratic
                  << "x" << std::endl;
      }
    }
  }

  // Równanie Woodsa
  if (rank == 0) {
    std::cout << "=== Testing Woods Function ===" << std::endl;
    std::cout << "n = " << n << ", processes = " << size << std::endl;
  }

  // RW wersja sekwencyjna
  long long dur_q_woods = 0;
  if (rank == 0) {
    std::vector<double> x_0 = make_woods_x0(n);
    auto start_w = std::chrono::high_resolution_clock::now();
    auto res = perform_sequential_algorithm(calc_woods_function, x_0, n, -5, 5,
                                            base_seed, debug);
    auto end_w = std::chrono::high_resolution_clock::now();
    dur_q_woods =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_w - start_w)
            .count();
    std::cout << "Sequential execution time: " << dur_q_woods
              << "ms, Distance to minimum: "
              << l2_norm_distance_to_woods_min(res.first, n) << std::endl;
  }

  // RW wersja równoległa
  long long dur_p_woods = 0;
  {
    std::vector<double> x_0 = make_woods_x0(n);
    auto start_p = std::chrono::high_resolution_clock::now();
    auto parallel_result = perform_parallel_algorithm_win(
        calc_woods_function_partial, x_0, n, -5, 5, 4, base_seed, debug, rank,
        size, shared_state, full_x_ptr);
    auto end_p = std::chrono::high_resolution_clock::now();
    dur_p_woods =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();
    if (rank == 0) {
      std::cout << "Parallel execution time:   " << dur_p_woods
                << "ms, Distance to minimum: "
                << l2_norm_distance_to_woods_min(parallel_result.first, n)
                << std::endl;
      if (dur_p_woods > 0) {
        std::cout << "Speedup: " << (double)dur_q_woods / dur_p_woods << "x"
                  << std::endl;
      }
    }
  }

  // Równanie Powella
  if (rank == 0) {
    std::cout << "=== Testing Powell Singular Function ===" << std::endl;
    std::cout << "n = " << n << ", processes = " << size << std::endl;
  }

  // RP wersja sekwencyjna
  long long dur_q_powell = 0;
  if (rank == 0) {
    std::vector<double> x_0 = make_powell_x0(n);
    auto start_p = std::chrono::high_resolution_clock::now();
    auto res = perform_sequential_algorithm(calc_powell_singular_function, x_0,
                                            n, -4, 4, base_seed, debug);
    auto end_p = std::chrono::high_resolution_clock::now();
    dur_q_powell =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();
    std::cout << "Sequential execution time: " << dur_q_powell
              << "ms, Distance to minimum: "
              << l2_norm_distance_to_powell_min(res.first, n) << std::endl;
  }

  // RP wersja równoległa
  long long dur_p_powell = 0;
  {
    std::vector<double> x_0 = make_powell_x0(n);
    auto start_p = std::chrono::high_resolution_clock::now();
    auto parallel_result = perform_parallel_algorithm_win(
        calc_powell_singular_function_partial, x_0, n, -4, 4, 4, base_seed,
        debug, rank, size, shared_state, full_x_ptr);
    auto end_p = std::chrono::high_resolution_clock::now();
    dur_p_powell =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_p - start_p)
            .count();
    if (rank == 0) {
      std::cout << "Parallel execution time:   " << dur_p_powell
                << "ms, Distance to minimum: "
                << l2_norm_distance_to_powell_min(parallel_result.first, n)
                << std::endl;
      if (dur_p_powell > 0) {
        std::cout << "Speedup: " << (double)dur_q_powell / dur_p_powell << "x"
                  << std::endl;
      }
    }
  }

  // Sprzątanie pamięci współdzielonej
  UnmapViewOfFile(pBuf);
  CloseHandle(hMapFile);

  return 0;
}
