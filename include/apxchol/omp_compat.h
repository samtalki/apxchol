#pragma once

// OpenMP is optional. Compilers ignore OpenMP pragmas when it is disabled, but
// a few shared headers also query the runtime directly. Keep those headers
// usable in serial builds without scattering _OPENMP guards across the code.
#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() noexcept { return 1; }
inline int omp_get_num_threads() noexcept { return 1; }
inline int omp_get_thread_num() noexcept { return 0; }
inline int omp_in_parallel() noexcept { return 0; }
inline void omp_set_num_threads(int) noexcept {}
#endif
