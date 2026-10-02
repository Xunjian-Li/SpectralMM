# Small-data Julia versus C++ comparison

Measured on 2026-09-30, macOS arm64, Julia 1.12.6. The existing dense
`examples/data.csv` provides 100 observations and 3 features; both interfaces
add an intercept (4 coefficients). C++ is called through the Julia package.

Timing excludes startup, JIT compilation and native building. It includes input
preparation, solving and result construction, plus inference when enabled.
Each configuration receives 10 warm-up calls. Reported times are medians of 21
randomly interleaved rounds, each averaging 100 complete calls. GC is run before
each batch; allocation and any collection inside a batch remain in the timing.
Interquartile ranges are retained in the CSV.

Both use PCG, rank 3, Krylov dimension 4, zero initial coefficients, floor/rho
1e-6, no ridge, maximum 500 outer and 100 inner iterations, absolute gradient
threshold 1e-6 and relative threshold 1e-8. Stalled-step acceptance is disabled;
C++ negligible-step acceptance is disabled. All returned fits meet the gradient
criterion. Verbose output and optional C++ trace collection are disabled.
All corresponding fits have identical iteration counts; the maximum coefficient
difference is 1.11e-16. Standard errors agree within 1e-5 tolerances.

Julia and its OpenBLAS report one thread. C++ links Apple Accelerate, with
VECLIB_MAXIMUM_THREADS=1 set before starting Julia. These are the current package
configurations, not identical BLAS implementations or a measurement of isolated
foreign-function-call overhead. No claim about other sizes or platforms follows.

| Model | Inference | Julia (microseconds) | C++ via Julia (microseconds) | C++ / Julia | Iterations (both) |
|---|---|---:|---:|---:|---:|
| bernoulli | false | 28.02 | 28.28 | 1.009 | 3 |
| bernoulli | true | 30.58 | 31.86 | 1.042 | 3 |
| poisson | false | 31.77 | 23.88 | 0.752 | 4 |
| poisson | true | 33.74 | 26.75 | 0.793 | 4 |
| pseudo_huber | false | 31.96 | 30.14 | 0.943 | 4 |
| pseudo_huber | true | 35.83 | 33.70 | 0.940 | 4 |
| student_t | false | 42.68 | 58.17 | 1.363 | 13 |
| student_t | true | 46.02 | 61.20 | 1.330 | 13 |

Logistic timings are close, with overlapping batch interquartile ranges.
C++ is faster for Poisson and modestly faster for Pseudo-Huber in this run;
Julia is faster for Student-t. There is no universal wrapper penalty ratio.

Reproduce from the repository root after installing the Julia dependencies:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 \
  julia --project=. benchmark/native/small_current_backends.jl
```

The normal interface builds or locates its cached C++ library before measured
calls. The recorded run used an explicit SPECTRALMM_LIBRARY path to the existing
package-built cached library to avoid sandbox restrictions on depot log writes.
