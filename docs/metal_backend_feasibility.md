# Apple GPU backend feasibility

Status: design note only. A Metal backend does not belong in the macOS
portability change.

## Current library support

No reviewed Apple GPU library supplies the sparse triangular solve needed by
apxchol:

- Apple
  [`MPSMatrixSolveTriangular`](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixsolvetriangular)
  accepts `MPSMatrix` operands. The
  [Metal.jl wrapper](https://github.com/JuliaGPU/Metal.jl/blob/265ec1d275af89293da7759a0635e71ce3971904/lib/mps/solve.jl#L50-L71)
  and its
  [matrix descriptor](https://github.com/JuliaGPU/Metal.jl/blob/265ec1d275af89293da7759a0635e71ce3971904/lib/mps/matrix.jl#L3-L21)
  expose dense matrices, not CSC or CSR factors.
- [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph)
  can construct sparse tensor representations, but its documented operation
  list has no sparse triangular solve.
- GPUArrays.jl defines CSC, CSR, COO, and BSR device types, while explicitly
  stating that it provides
  [no device-compatible sparse operations](https://github.com/JuliaGPU/GPUArrays.jl/blob/82ed1dadd19cddfcdc28e586406fb1839fb7ca07/src/device/sparse.jl#L1-L7).
  Metal.jl does not implement a sparse array backend on top of those types.
- MLX exposes
  [`solve_triangular`](https://github.com/ml-explore/mlx/blob/447bb0f9ef20789894c7786e23358578db2f247b/mlx/linalg.h#L91-L98)
  for dense arrays and has no sparse matrix representation or sparse solve.
- Accelerate has sparse CPU solvers, including triangular solves, but
  [Accelerate executes on the CPU](https://developer.apple.com/documentation/accelerate).

Metal.jl
[rejects `Float64` arrays](https://github.com/JuliaGPU/Metal.jl/blob/265ec1d275af89293da7759a0635e71ce3971904/src/array.jl#L29-L35),
and its opinionated
[array adaptor converts floating point inputs to `Float32`](https://github.com/JuliaGPU/Metal.jl/blob/265ec1d275af89293da7759a0635e71ce3971904/src/array.jl#L439-L452).
The existing CUDA path stores factor values in `Float32`, but keeps PCG vectors,
sparse matrix products, dot products, and residual checks in `Float64`. A fully
GPU-resident Metal implementation cannot copy that accuracy model.

## Scoped prototype

A custom Metal compute path is technically possible because the factor already
has dependency level sets. It should begin as a benchmark target, not a public
backend or C ABI extension.

The prototype would reuse the CPU factorization and prepare these persistent
buffers once:

1. CSR for `L` and CSR for `Lᵀ`, with 32-bit row offsets and column indices.
2. `Float32` factor values.
3. Forward and backward row order plus level boundary arrays.
4. Work vectors in shared or private Metal buffers, selected by measurement.

The triangular kernel would assign one SIMD group to each row. Threads would
stride over the row entries, reduce the partial sum within the SIMD group, and
write one result. Encode one dispatch per dependency level into a single command
buffer so dependencies remain ordered without a CPU synchronization between
levels. This follows the current CUDA level set algorithm and gives a direct
correctness comparison.

Two execution models need separate measurement:

- CPU PCG with a Metal `Float32` preconditioner. This preserves `Float64` outer
  arithmetic but pays a command buffer completion at every preconditioner call.
- Metal PCG with custom sparse matrix vector, vector update, and reduction
  kernels. This avoids per-iteration CPU synchronization but requires an
  accuracy strategy. Periodic CPU `Float64` residual replacement or iterative
  refinement is acceptable for the prototype; a relaxed residual target is not.

The first prototype must not change the installed library, public headers,
default build, Python module, or C ABI. If it passes the gates below, a later
change can introduce a mutually exclusive `APXCHOL_USE_METAL` build option and
an Objective-C++ implementation hidden behind the existing solver interface.

## Benchmark plan

Use generated 2D and 3D grids plus publicly downloadable SuiteSparse matrices
already named by the benchmark suite. Cover block greedy, Luby, and rootset
factors so level width and depth vary. Record factor order, factor nonzeros,
level count, maximum level width, and work per level with every result.

Run on at least two Apple GPU families, including one base chip and one
Max or Ultra chip. Compare these configurations:

1. Serial CPU triangular solve.
2. OpenMP triangular solve at 1, 2, 4, 8, and all physical performance cores.
3. Metal triangular solve with CPU PCG.
4. Metal resident PCG, if its accuracy gate passes.

Measure factorization, Metal buffer preparation, first solve, median warm solve,
end-to-end time, PCG iterations, true relative residual, and peak buffer memory.
Use one right hand side to test the current interface and batches of 10 and 100
right hand sides to determine whether setup reuse changes the result. Take at
least 10 warm measurements and report the median and interquartile range.

The prototype advances only if all of these hold:

- Every accepted solve reaches a true relative residual at or below `1e-8` on
  the original operator.
- Results are free of Metal validation errors, data races, and nonfinite values.
- PCG iteration growth is no more than 20 percent over the CPU `Float32` factor
  baseline.
- Metal is faster end to end for at least one defined workload class after
  buffer preparation is charged. A speedup limited to the triangular kernel is
  not enough.
- The winning region is stable across both tested GPU families and exceeds run
  variance.

If the accuracy gate fails, stop. If accuracy passes but end-to-end time does
not, retain the benchmark results and do not add a backend.
