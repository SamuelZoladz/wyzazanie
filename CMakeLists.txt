cmake_minimum_required(VERSION 3.20)
project(simulated_annealing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(simulated_annealing main.cpp algorithm.cpp functions.cpp)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(simulated_annealing PUBLIC OpenMP::OpenMP_CXX)
endif()
