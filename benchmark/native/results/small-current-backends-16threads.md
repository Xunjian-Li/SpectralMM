# Small-data maximum-thread comparison

2026-09-30, same 100-by-3 data, PCG settings and timing protocol as
[the single-thread run](small-current-backends.md). The host reports 16 logical
CPUs. This run sets Julia, OpenBLAS, OpenMP and Accelerate thread limits to 16.
Julia reports 16 execution threads and 16 OpenBLAS threads; Accelerate is
configured with VECLIB_MAXIMUM_THREADS=16. This is permission to use up to 16
threads, not a measurement showing that each small operation used 16 workers.
The solver's serial loops do not automatically become parallel.

The single-thread baseline and this run are separate sequential processes;
small timing changes can reflect run-to-run noise. Both use 21 interleaved
rounds of 100 calls per backend/configuration, after warm-up. Compilation and
startup are excluded. All coefficient, convergence and inference checks passed,
with the same iteration counts as the baseline.

Times below include standard-error inference and are microseconds per fit.

| Model | Julia, 1 thread | Julia, up to 16 | C++ via Julia, 1 thread | C++ via Julia, up to 16 |
|---|---:|---:|---:|---:|
| bernoulli | 30.58 | 31.98 | 31.86 | 33.60 |
| poisson | 33.74 | 35.85 | 26.75 | 27.26 |
| pseudo_huber | 35.83 | 38.29 | 33.70 | 34.13 |
| student_t | 46.02 | 46.94 | 61.20 | 61.96 |

No speedup was observed from raising the thread limit on these small examples.
This does not establish scaling behavior for larger matrices. Actual worker
utilization was not profiled. Julia uses OpenBLAS and C++ uses Accelerate.

Reproduce from the repository root:

```sh
VECLIB_MAXIMUM_THREADS=16 OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
JULIA_NUM_THREADS=16 SPECTRALMM_BENCH_THREADS=16 \
SPECTRALMM_BENCH_OUTPUT=benchmark/native/results/small-current-backends-16threads.csv \
  julia --project=. benchmark/native/small_current_backends.jl
```

These settings apply only to the benchmark process; package defaults were not
changed. Both inference modes and timing quartiles are retained in the CSV.
