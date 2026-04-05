cmake_minimum_required(VERSION 3.18)
project(RTRExtractor_GPU LANGUAGES CXX CUDA)

# --- 1. Framework Standards ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# --- 2. Dependencies ---
find_package(OpenMP REQUIRED)

# --- 3. The Executable ---
add_executable(tric_gpu
    src/main.cpp
    src/tricount_gpu.cu
    src/log.cpp
)

# --- 4. Include Directories ---
target_include_directories(tric_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- 5. Compiler Flags ---
target_compile_options(tric_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --extended-lambda --expt-relaxed-constexpr -DSM_TARGET=86>
)

# --- 6. Linking ---
target_link_libraries(tric_gpu PRIVATE OpenMP::OpenMP_CXX)