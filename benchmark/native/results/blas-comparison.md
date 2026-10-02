# Apple Accelerate BLAS enabled

The current local Release shared library was rebuilt with `SPECTRALMM_USE_BLAS=ON` and `BLA_VENDOR=Apple`, using Eigen 3.4.0 headers. `otool -L` confirms its Accelerate dependency, and `nm -u` lists BLAS references including dgemm, dgemv, dsyrk and dtrsm. CMake retains this configuration in build/native. The portable CMake option still defaults to OFF; the documented build command explicitly enables it.

Python, R and the native Julia wrapper share this library. Restart language sessions after rebuilding a library already loaded at the same path. The original Julia implementation is not modified. Sparse products continue to use Eigen.

## Numerical comparison

224 before/after comparisons covered 14 families, eight solvers and dense/CSC inputs on the saved 100-by-3 example data. Coefficients, loss and covariance met the comparison tolerances; gradient convergence flags and inference availability agreed. The maximum coefficient difference was 1.4685e-8. Iteration counts changed in four Probit configurations (PCG, MM, CGLS and LSQR); bitwise-identical traces are not expected when the linear algebra backend changes. Coefficient tolerances were rtol=1e-5 and atol=1e-7; loss tolerances were rtol=1e-10 and atol=1e-9.

Fresh R and Julia processes also each passed sixteen calls across the eight solvers and dense/CSC inputs using the new library. R inference and final trace consistency in both wrappers were checked.

## Warmed dense Logistic timings

The Eigen-only library was saved before rebuilding. Both versions were loaded in one Python process and measured in nine randomized interleaved rounds after four warm-up calls per version. Rank was min(5, features), starting Lanczos dimension min(features+1, 12), with zero initial coefficients, default intercept, gtol=1e-6, relgtol=1e-8, and negligible/stalled-step acceptance disabled. Inference, printing and trace capture were disabled. Batches contained 20 fits for the smallest case and three otherwise. OPENBLAS_NUM_THREADS and VECLIB_MAXIMUM_THREADS were both set to 1. Inputs were Fortran-contiguous Float64 matrices; generation was excluded.

| Input | Eigen-only ms | BLAS ms | Time reduction | Iterations (both) |
|---|---:|---:|---:|---:|
| 100 x 3 | 0.217 | 0.215 | 0.9% | 4 |
| 2000 x 30 | 11.368 | 8.585 | 24.5% | 10 |
| 2000 x 100 | 28.548 | 16.218 | 43.2% | 18 |
| 5000 x 300 | 127.109 | 72.937 | 42.6% | 19 |

The 100-by-3 case was effectively unchanged; supported dense BLAS operations helped the larger cases. These are local before/after C++ timings through Python, not a new comparison against original Julia. The earlier 1.43 ratio cannot be updated from these different workloads. Maximum coefficient differences in the four timed cases were below 4e-16. Timings depend on workload, host load and backend; no universal speedup is claimed.

Current library SHA-256: `c02929b1b7c698b9dedde55d10845a0a539f6852bcc16ab2e72496875e158ba1`.
