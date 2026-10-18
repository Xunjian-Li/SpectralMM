> **Historical record.** Commands and paths below describe the original run and may be obsolete. For current experiments, follow the [benchmark guide](../../../README.md). Measurements and provenance files are retained unchanged.

# Python/R PCG and MM comparison

Measured on 2026-10-15 using current local sources; n=2000, 1000 predictors plus one supplied intercept. Dense and sparse inputs are shared across interfaces. Five interleaved timed rounds after warm-up; reported values are medians in milliseconds. Inference is disabled. Threads are requested as one, retaining the native benchmark settings. Python uses an Apple Accelerate C++ build; R uses its configured BLAS (see environment_R.txt). These are not measurements of identical BLAS providers.

Only PCG and MM were rerun. Other methods and historical timings are not combined into this comparison. Initial coefficients, rank=5, krylovdim=12, rho=1e-6, ridge=0, tolerances, and iteration limits match between the two methods.

**An asterisk means the result did not meet the gradient criterion** (absolute gradient <= 1e-6 or relative gradient <= 1e-8). Solver-declared convergence can also reflect small-step/stagnation stopping; those times must not be interpreted as equal-accuracy speedups. Raw CSVs include termination reasons, iteration counts and final gradients.

## Dense

| Model | Python PCG (ms) | Python MM (ms) | R PCG (ms) | R MM (ms) |
|---|---:|---:|---:|---:|
| gaussian | 15.83 | 34.87 | 78.00 | 178.00 |
| bernoulli | 72.38 | 185.11 | 317.00 | 959.00 |
| probit | 70.90 | 145.47 | 241.00 | 707.00 |
| poisson | 33.39* | 107.55 | 163.00* | 560.00 |
| gamma | 28.99* | 48.96* | 127.00* | 217.00* |
| negative_binomial | 27.94* | 79.87 | 120.00* | 369.00 |
| smooth_quantile | 72.21* | 344.62 | 387.00* | 1871.00 |
| expectile | 18.89* | 35.28* | 94.00* | 180.00* |
| pseudo_huber | 22.04 | 45.52* | 102.00 | 217.00* |
| student_t | 27.86* | 62.89* | 113.00* | 284.00* |

## Sparse

| Model | Python PCG (ms) | Python MM (ms) | R PCG (ms) | R MM (ms) |
|---|---:|---:|---:|---:|
| gaussian | 5.17* | 6.74 | 5.13* | 6.75 |
| bernoulli | 52.19 | 121.43 | 52.00 | 116.00 |
| probit | 97.17 | 79.33 | 93.00 | 78.00 |
| poisson | 7.53 | 11.93 | 7.50 | 10.60 |
| gamma | 16.90* | 33.92 | 17.33* | 34.00 |
| negative_binomial | 9.07* | 15.82 | 9.33* | 16.00 |
| smooth_quantile | 20.52 | 172.77 | 19.33* | 162.00 |
| expectile | 6.04 | 9.14 | 6.00 | 9.17 |
| pseudo_huber | 7.61 | 12.18 | 7.57 | 12.20 |
| student_t | 9.70 | 13.92 | 9.67 | 13.50 |
