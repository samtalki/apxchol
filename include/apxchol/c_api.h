#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxchol_solve_info {
    int32_t iterations;
    double relative_residual;
    double setup_seconds;
    double solve_seconds;
} apxchol_solve_info;

typedef int32_t apxchol_status;
enum {
    APXCHOL_STATUS_SUCCESS = 0,
    APXCHOL_STATUS_ERROR = 1,
    APXCHOL_STATUS_UNKNOWN_ERROR = 2,
    APXCHOL_STATUS_NOT_CONVERGED = 3
};

/// Solve a Laplacian or SDDM system supplied in CSC form.
///
/// `index_base` must be 0 or 1 and applies to both `colptr` and `rowval`. The
/// caller supplies `n + 1` column pointers, `nnz` row indices and values, and
/// `n` entries in each dense vector. The function copies the matrix and right
/// hand side into Eigen storage, factorizes the matrix, and writes `n` entries
/// to `solution`; `rhs` and `solution` may alias. A positive `threads` value
/// applies for this call and the prior OpenMP thread setting is restored before
/// return; `threads <= 0` leaves it unchanged. Returns
/// `APXCHOL_STATUS_SUCCESS` only when the requested tolerance is reached.
/// `APXCHOL_STATUS_NOT_CONVERGED` leaves the best solution and statistics in
/// the output buffers. Other nonzero statuses report an error, leave `info`
/// zero-initialized, and do not guarantee the contents of `solution`. If
/// `error_buffer` is non-null, `error_capacity` is its writable size in bytes.
apxchol_status apxchol_solve_csc64(
    int64_t n,
    int64_t nnz,
    const int64_t* colptr,
    const int64_t* rowval,
    const double* values,
    int32_t index_base,
    const double* rhs,
    double tolerance,
    int32_t max_iterations,
    uint32_t seed,
    int32_t threads,
    double* solution,
    apxchol_solve_info* info,
    char* error_buffer,
    int64_t error_capacity);

#ifdef __cplusplus
}
#endif
