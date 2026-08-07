#include <gtest/gtest.h>

#include "apxchol/c_api.h"
#include "apxchol/omp_compat.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

TEST(CApi, SolvesOneBasedLaplacianCsc) {
    const std::array<std::int64_t, 3> colptr{1, 3, 5};
    const std::array<std::int64_t, 4> rowval{1, 2, 1, 2};
    const std::array<double, 4> values{1.0, -1.0, -1.0, 1.0};
    const std::array<double, 2> rhs{1.0, -1.0};
    std::array<double, 2> solution{};
    apxchol_solve_info info{};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        2, 4, colptr.data(), rowval.data(), values.data(), 1, rhs.data(),
        1e-10, 100, 7, 1, solution.data(), &info, error.data(), error.size());

    ASSERT_EQ(status, 0) << error.data();
    EXPECT_NEAR(solution[0] - solution[1], 1.0, 1e-12);
    EXPECT_LT(info.relative_residual, 1e-10);
    EXPECT_GE(info.iterations, 1);
}

TEST(CApi, RejectsInvalidIndexBase) {
    const std::array<std::int64_t, 2> colptr{0, 0};
    const std::array<std::int64_t, 1> rowval{0};
    const std::array<double, 1> values{0.0};
    const std::array<double, 1> rhs{0.0};
    std::array<double, 1> solution{};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        1, 0, colptr.data(), rowval.data(), values.data(), 2, rhs.data(),
        1e-8, 10, 1, 1, solution.data(), nullptr, error.data(), error.size());

    EXPECT_NE(status, 0);
    EXPECT_NE(std::string(error.data()).find("index_base"), std::string::npos);
}

TEST(CApi, SupportsAliasedRhsAndSolution) {
    const std::array<std::int64_t, 3> colptr{0, 2, 4};
    const std::array<std::int64_t, 4> rowval{0, 1, 0, 1};
    const std::array<double, 4> values{1.0, -1.0, -1.0, 1.0};
    std::array<double, 2> rhs_and_solution{1.0, -1.0};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        2, 4, colptr.data(), rowval.data(), values.data(), 0,
        rhs_and_solution.data(), 1e-10, 100, 7, 1,
        rhs_and_solution.data(), nullptr, error.data(), error.size());

    ASSERT_EQ(status, 0) << error.data();
    EXPECT_NEAR(rhs_and_solution[0] - rhs_and_solution[1], 1.0, 1e-12);
}

TEST(CApi, ReportsIterationExhaustionAsNotConverged) {
    const std::array<std::int64_t, 6> colptr{0, 5, 10, 15, 20, 25};
    const std::array<std::int64_t, 25> rowval{
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4};
    const std::array<double, 25> values{
        10.0, -1.0, -2.0, -3.0, -4.0,
        -1.0, 19.0, -5.0, -6.0, -7.0,
        -2.0, -5.0, 24.0, -8.0, -9.0,
        -3.0, -6.0, -8.0, 27.0, -10.0,
        -4.0, -7.0, -9.0, -10.0, 30.0};
    const std::array<double, 5> rhs{1.0, -2.0, 3.0, -4.0, 2.0};
    std::array<double, 5> solution{};
    apxchol_solve_info info{};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        5, 25, colptr.data(), rowval.data(), values.data(), 0, rhs.data(),
        1e-30, 1, 7, 1, solution.data(), &info, error.data(), error.size());

    EXPECT_EQ(status, APXCHOL_STATUS_NOT_CONVERGED);
    EXPECT_GE(info.relative_residual, 1e-30);
    EXPECT_NE(std::string(error.data()).find("did not converge"), std::string::npos);
}

TEST(CApi, RestoresOpenMPThreadLimitAfterSuccessAndError) {
    const std::array<std::int64_t, 3> colptr{1, 3, 5};
    const std::array<std::int64_t, 4> rowval{1, 2, 1, 2};
    const std::array<double, 4> values{1.0, -1.0, -1.0, 1.0};
    const std::array<double, 2> rhs{1.0, -1.0};
    std::array<double, 2> solution{};
    std::array<char, 256> error{};
    const int previous = omp_get_max_threads();
    const int requested = previous == 1 ? 2 : 1;

    const int success = apxchol_solve_csc64(
        2, 4, colptr.data(), rowval.data(), values.data(), 1, rhs.data(),
        1e-10, 100, 7, requested, solution.data(), nullptr, error.data(),
        error.size());

    ASSERT_EQ(success, 0) << error.data();
    EXPECT_EQ(omp_get_max_threads(), previous);

    const std::array<std::int64_t, 4> invalid_rowval{1, 2, 1, 3};
    const int failure = apxchol_solve_csc64(
        2, 4, colptr.data(), invalid_rowval.data(), values.data(), 1, rhs.data(),
        1e-10, 100, 7, requested, solution.data(), nullptr, error.data(),
        error.size());

    EXPECT_NE(failure, 0);
    EXPECT_EQ(omp_get_max_threads(), previous);
}

TEST(CApi, RejectsExtremeOneBasedIndicesWithoutOverflow) {
    const std::array<std::int64_t, 2> colptr{
        1, std::numeric_limits<std::int64_t>::min()};
    const std::array<std::int64_t, 1> rowval{
        std::numeric_limits<std::int64_t>::min()};
    const std::array<double, 1> values{1.0};
    const std::array<double, 1> rhs{1.0};
    std::array<double, 1> solution{};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        1, 0, colptr.data(), rowval.data(), values.data(), 1, rhs.data(),
        1e-8, 10, 1, 1, solution.data(), nullptr, error.data(), error.size());

    EXPECT_NE(status, 0);
    EXPECT_NE(std::string(error.data()).find("column pointers"), std::string::npos);
}

TEST(CApi, RejectsExtremeOneBasedRowIndexWithoutOverflow) {
    const std::array<std::int64_t, 2> colptr{1, 2};
    const std::array<std::int64_t, 1> rowval{
        std::numeric_limits<std::int64_t>::min()};
    const std::array<double, 1> values{1.0};
    const std::array<double, 1> rhs{1.0};
    std::array<double, 1> solution{};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        1, 1, colptr.data(), rowval.data(), values.data(), 1, rhs.data(),
        1e-8, 10, 1, 1, solution.data(), nullptr, error.data(), error.size());

    EXPECT_NE(status, 0);
    EXPECT_NE(std::string(error.data()).find("row index"), std::string::npos);
}

TEST(CApi, ClearsInfoAndBoundsAOneByteErrorBuffer) {
    const std::array<std::int64_t, 2> colptr{0, 0};
    const std::array<std::int64_t, 1> rowval{0};
    const std::array<double, 1> values{0.0};
    const std::array<double, 1> rhs{0.0};
    std::array<double, 1> solution{42.0};
    apxchol_solve_info info{7, 1.0, 2.0, 3.0};
    std::array<char, 1> error{'x'};

    const int status = apxchol_solve_csc64(
        0, 0, colptr.data(), rowval.data(), values.data(), 0, rhs.data(),
        1e-8, 10, 1, 1, solution.data(), &info, error.data(), error.size());

    EXPECT_EQ(status, APXCHOL_STATUS_ERROR);
    EXPECT_EQ(error[0], '\0');
    EXPECT_EQ(solution[0], 42.0);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.relative_residual, 0.0);
    EXPECT_EQ(info.setup_seconds, 0.0);
    EXPECT_EQ(info.solve_seconds, 0.0);
}

TEST(CApi, RejectsNonfiniteTolerance) {
    const std::array<std::int64_t, 2> colptr{0, 1};
    const std::array<std::int64_t, 1> rowval{0};
    const std::array<double, 1> values{1.0};
    const std::array<double, 1> rhs{1.0};
    std::array<double, 1> solution{};
    std::array<char, 256> error{};

    const int status = apxchol_solve_csc64(
        1, 1, colptr.data(), rowval.data(), values.data(), 0, rhs.data(),
        std::numeric_limits<double>::infinity(), 10, 1, 1, solution.data(),
        nullptr, error.data(), error.size());

    EXPECT_EQ(status, APXCHOL_STATUS_ERROR);
    EXPECT_NE(std::string(error.data()).find("tolerance"), std::string::npos);
}

TEST(CApi, RejectsNonfiniteMatrixAndRightHandSideData) {
    const std::array<std::int64_t, 2> colptr{0, 1};
    const std::array<std::int64_t, 1> rowval{0};
    const std::array<double, 1> finite{1.0};
    const std::array<double, 1> nonfinite{
        std::numeric_limits<double>::quiet_NaN()};
    std::array<double, 1> solution{};
    std::array<char, 256> error{};

    const int matrix_status = apxchol_solve_csc64(
        1, 1, colptr.data(), rowval.data(), nonfinite.data(), 0,
        finite.data(), 1e-8, 10, 1, 1, solution.data(), nullptr,
        error.data(), error.size());
    EXPECT_EQ(matrix_status, APXCHOL_STATUS_ERROR);
    EXPECT_NE(std::string(error.data()).find("matrix"), std::string::npos);

    const int rhs_status = apxchol_solve_csc64(
        1, 1, colptr.data(), rowval.data(), finite.data(), 0,
        nonfinite.data(), 1e-8, 10, 1, 1, solution.data(), nullptr,
        error.data(), error.size());
    EXPECT_EQ(rhs_status, APXCHOL_STATUS_ERROR);
    EXPECT_NE(std::string(error.data()).find("right hand side"),
              std::string::npos);
}
