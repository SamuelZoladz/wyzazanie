cmake_minimum_required(VERSION 3.20)
project(rozproszone)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rozproszone main.cpp algorithm.cpp functions.cpp)