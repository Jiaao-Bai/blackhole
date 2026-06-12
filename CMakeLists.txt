cmake_minimum_required(VERSION 3.24)

project(blackhole LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BLACKHOLE_SM100      "Build sm_100 (datacenter Blackwell) kernels" ON)
option(BLACKHOLE_SM120      "Build sm_120 (consumer Blackwell) kernels"   ON)
option(BLACKHOLE_BUILD_TESTS      "Build correctness tests" ON)
option(BLACKHOLE_BUILD_BENCHMARKS "Build benchmarks"        ON)
option(BLACKHOLE_BUILD_EXAMPLES   "Build examples"          ON)

# ---------------------------------------------------------------------------
# Standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# CUDA architectures
# ---------------------------------------------------------------------------
set(BLACKHOLE_CUDA_ARCHS "")
if(BLACKHOLE_SM100)
  list(APPEND BLACKHOLE_CUDA_ARCHS 100a)
endif()
if(BLACKHOLE_SM120)
  list(APPEND BLACKHOLE_CUDA_ARCHS 120a)
endif()
if(NOT BLACKHOLE_CUDA_ARCHS)
  message(FATAL_ERROR "Enable at least one of BLACKHOLE_SM100 / BLACKHOLE_SM120")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${BLACKHOLE_CUDA_ARCHS})

# ---------------------------------------------------------------------------
# CUTLASS / CuTe (header-only, from submodule)
# ---------------------------------------------------------------------------
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
if(NOT EXISTS ${CUTLASS_DIR}/include/cute/tensor.hpp)
  message(FATAL_ERROR
    "CUTLASS submodule not found. Run: git submodule update --init --recursive")
endif()

add_library(cutlass_cute INTERFACE)
target_include_directories(cutlass_cute INTERFACE
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include)
target_compile_features(cutlass_cute INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# blackhole interface (header-only public API)
# ---------------------------------------------------------------------------
add_library(blackhole INTERFACE)
add_library(blackhole::blackhole ALIAS blackhole)
target_include_directories(blackhole INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(blackhole INTERFACE cutlass_cute)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(BLACKHOLE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(BLACKHOLE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
if(BLACKHOLE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
