cmake_minimum_required(VERSION 3.18)
project(RTRExtractor_GPU LANGUAGES CXX CUDA)

# --- 1. Framework Standards ---
# Gunrock requires modern C++. Bumping from your original C++11.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 2. Architecture Settings ---
# Replaces your -rdc=true flag for proper cross-kernel linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# --- 3. Dependencies ---
find_package(OpenMP REQUIRED)

# --- 4. The Executable ---
add_executable(tric_gpu
    src/main.cpp
    src/tricount_gpu.cu
    src/log.cpp
)

# --- 5. Include Directories ---
# Links your local headers PLUS the newly cloned Gunrock and CUB frameworks
target_include_directories(tric_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/gunrock
    ${CMAKE_CURRENT_SOURCE_DIR}/externals/gunrock/externals/cub
)

# --- 6. Compiler Flags ---
# Preserving your -O2 optimization and native architecture flags
target_compile_options(tric_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 -arch=native>
)

# --- 7. Linking ---
target_link_libraries(tric_gpu PRIVATE OpenMP::OpenMP_CXX)