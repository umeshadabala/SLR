cmake_minimum_required(VERSION 3.22.1)
project(slr_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
else()
    set(OpenMP_FLAGS "-fopenmp")
endif()

add_library(slr_native SHARED
    slr_native.cpp
)

target_compile_options(slr_native PRIVATE
    -O3
    -ffast-math
    ${OpenMP_FLAGS}
)

target_link_libraries(slr_native PRIVATE
    log
    ${OpenMP_FLAGS}
    -static-openmp
)
