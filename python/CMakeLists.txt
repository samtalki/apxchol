cmake_minimum_required(VERSION 3.16)
project(apxchol_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(APXCHOL_ENABLE_OPENMP "Enable OpenMP parallelism when a runtime is available" ON)

# Repo root (this CMakeLists lives in python/).
set(APX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Eigen (system or fetched, mirrors root) ──
include(${APX_ROOT}/cmake/ApxCholEigen.cmake)
apxchol_find_or_fetch_eigen()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
# The two core TUs depend only on Eigen + OpenMP (no fast_matrix_market, no
# generated version.h, no CLI11/spdlog), so we compile them straight into the
# extension. The benchmark-critical apxchol::solve in solve.cpp is reused as-is.
pybind11_add_module(_apxchol
    src/apxchol_ext.cpp
    ${APX_ROOT}/src/factorization.cpp
    ${APX_ROOT}/src/solve.cpp
)
target_include_directories(_apxchol PRIVATE ${APX_ROOT}/include)
target_link_libraries(_apxchol PRIVATE Eigen3::Eigen)
include(${APX_ROOT}/cmake/ApxCholOpenMP.cmake)
apxchol_enable_openmp(_apxchol PRIVATE)

# Match the root library defaults (both ON by default there).
target_compile_definitions(_apxchol PRIVATE APXCHOL_SPTRSV_FP32 APXCHOL_POOL_FP32)
if (NOT MSVC)
    # Distributable wheels (cibuildwheel sets APXCHOL_PORTABLE_BUILD=1) must not
    # use -march=native — it would SIGILL on older CPUs. Local source builds keep it.
    if (DEFINED ENV{APXCHOL_PORTABLE_BUILD})
        target_compile_options(_apxchol PRIVATE -O3)
    else()
        target_compile_options(_apxchol PRIVATE -O3 -march=native)
    endif()
endif()

install(TARGETS _apxchol LIBRARY DESTINATION apxchol)
