#include "apxchol/c_api.h"

#include "apxchol/solver/solve.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;

double seconds(clock_type::time_point start, clock_type::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

void set_error(char* buffer, std::int64_t capacity, const char* message) noexcept {
    if (buffer == nullptr || capacity <= 0) return;
    const auto available = static_cast<std::uint64_t>(capacity - 1);
    std::size_t length = std::strlen(message);
    if (available < length) length = static_cast<std::size_t>(available);
    std::memcpy(buffer, message, length);
    buffer[length] = '\0';
}

class openmp_thread_limit_guard {
public:
    explicit openmp_thread_limit_guard(std::int32_t threads) noexcept
        : previous_(threads > 0 ? omp_get_max_threads() : 1),
          restore_(threads > 0) {
        if (restore_) omp_set_num_threads(threads);
    }

    ~openmp_thread_limit_guard() {
        if (restore_) omp_set_num_threads(previous_);
    }

    openmp_thread_limit_guard(const openmp_thread_limit_guard&) = delete;
    openmp_thread_limit_guard& operator=(const openmp_thread_limit_guard&) = delete;

private:
    int previous_;
    bool restore_;
};
}  // namespace

extern "C" apxchol_status apxchol_solve_csc64(
    std::int64_t n,
    std::int64_t nnz,
    const std::int64_t* colptr,
    const std::int64_t* rowval,
    const double* values,
    std::int32_t index_base,
    const double* rhs,
    double tolerance,
    std::int32_t max_iterations,
    std::uint32_t seed,
    std::int32_t threads,
    double* solution,
    apxchol_solve_info* info,
    char* error_buffer,
    std::int64_t error_capacity) {
    if (info != nullptr) *info = {};
    set_error(error_buffer, error_capacity, "");
    try {
        if (n <= 0 || nnz < 0 || colptr == nullptr || rowval == nullptr ||
            values == nullptr || rhs == nullptr || solution == nullptr) {
            throw std::invalid_argument("invalid null pointer or matrix dimension");
        }
        if (index_base != 0 && index_base != 1)
            throw std::invalid_argument("index_base must be 0 or 1");
        if (n > std::numeric_limits<int>::max() ||
            nnz > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("matrix exceeds Eigen's 32-bit index range");
        }
        const std::int64_t first_index = index_base;
        const std::int64_t past_last_index = nnz + index_base;
        if (colptr[0] != first_index || colptr[n] != past_last_index)
            throw std::invalid_argument("invalid CSC column pointers");
        if (!std::isfinite(tolerance) || !(tolerance > 0.0) ||
            max_iterations <= 0) {
            throw std::invalid_argument("tolerance and max_iterations must be positive");
        }

        const openmp_thread_limit_guard thread_limit(threads);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<std::size_t>(nnz));
        for (std::int64_t column = 0; column < n; ++column) {
            if (!std::isfinite(rhs[column]))
                throw std::invalid_argument("right hand side contains a nonfinite value");
            const std::int64_t raw_begin = colptr[column];
            const std::int64_t raw_end = colptr[column + 1];
            if (raw_begin < first_index || raw_end < raw_begin ||
                raw_end > past_last_index) {
                throw std::invalid_argument("invalid CSC column pointers");
            }
            const std::int64_t begin = raw_begin - first_index;
            const std::int64_t end = raw_end - first_index;
            for (std::int64_t p = begin; p < end; ++p) {
                if (!std::isfinite(values[p]))
                    throw std::invalid_argument("matrix contains a nonfinite value");
                const std::int64_t raw_row = rowval[p];
                if (raw_row < first_index || raw_row >= n + index_base)
                    throw std::invalid_argument("CSC row index out of range");
                const std::int64_t row = raw_row - first_index;
                triplets.emplace_back(
                    static_cast<int>(row), static_cast<int>(column), values[p]);
            }
        }

        Eigen::SparseMatrix<double> matrix(
            static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        const Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(
            rhs, static_cast<Eigen::Index>(n));
        Eigen::Map<Eigen::VectorXd> x(solution, static_cast<Eigen::Index>(n));

        apxchol::solve_options options;
        options.tol = tolerance;
        options.max_iter = max_iterations;
        options.stagnation_window = 0;
        options.factor_opts.seed = seed;

        const auto setup_start = clock_type::now();
        apxchol::cpu_solver solver(matrix, options);
        const auto setup_stop = clock_type::now();
        const apxchol::solve_result result = solver.solve(
            b, x, tolerance, max_iterations);
        const auto solve_stop = clock_type::now();

        if (info != nullptr) {
            info->iterations = static_cast<std::int32_t>(result.iterations);
            info->relative_residual = result.residual;
            info->setup_seconds = seconds(setup_start, setup_stop);
            info->solve_seconds = seconds(setup_stop, solve_stop);
        }
        if (!(result.residual <= tolerance)) {
            set_error(error_buffer, error_capacity,
                      "solver did not converge to the requested tolerance");
            return APXCHOL_STATUS_NOT_CONVERGED;
        }
        set_error(error_buffer, error_capacity, "");
        return APXCHOL_STATUS_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_buffer, error_capacity, error.what());
        return APXCHOL_STATUS_ERROR;
    } catch (...) {
        set_error(error_buffer, error_capacity, "unknown C++ exception");
        return APXCHOL_STATUS_UNKNOWN_ERROR;
    }
}
