# Benchmarks

For Python/R native comparisons and final results, see [native/README.md](native/README.md).
The English LaTeX usage/reproduction guide is [here](../docs/usage_and_benchmark.tex).

## Original Julia benchmark

From the repository root:

```sh
julia --project=benchmark benchmark/setup.jl
julia --threads=12 --project=benchmark benchmark/scaling.jl
julia --threads=12 --project=benchmark benchmark/scaling_sparse.jl
```

These scripts use `benchmark/common.jl` and include their own timing loop.
They write the original scaling CSV outputs; the existing CSV files are retained.

The benchmark environment contains its own CSV, DataFrames, BenchmarkTools and
GLM dependencies. Setup uses the local `julia/` package without adding benchmark
dependencies to its runtime requirements. Restart a notebook kernel before
loading this environment if packages were already loaded from another environment.

From a notebook started in the repository root, run setup once:

```julia
include("benchmark/setup.jl")
```

Then load the helpers or run one full benchmark:

```julia
include("benchmark/common.jl") # Defines functions only; does not write CSV.
# include("benchmark/scaling.jl") # Runs dense experiments and overwrites their CSV.
# include("benchmark/scaling_sparse.jl") # Runs sparse experiments and overwrites their CSV.
```

## Thread configuration

The Julia scaling benchmarks set BLAS to 12 threads by default and print the
actual Julia thread count, BLAS thread count and BLAS implementation before fitting.
Set `SPECTRALMM_BENCH_THREADS` to a positive integer or `max` before loading
`common.jl` or either scaling script. `max` requests `Sys.CPU_THREADS` BLAS threads.
For example, on macOS:

```sh
SPECTRALMM_BENCH_THREADS=max julia --threads="$(sysctl -n hw.logicalcpu)" --project=benchmark benchmark/scaling.jl
```

In a notebook:

```julia
ENV["SPECTRALMM_BENCH_THREADS"] = "12" # Or "max".
include("benchmark/scaling.jl")
```

For ordinary Julia fits outside the benchmark scripts, configure BLAS explicitly:

```julia
using LinearAlgebra
BLAS.set_num_threads(12) # Or Sys.CPU_THREADS for all logical CPUs.
using SpectralMM
```

BLAS settings affect the entire Julia process. Julia worker thread counts are set
at process startup with `--threads`; changing BLAS threads does not change them.
Loading SpectralMM alone does not change thread settings. Maximum thread counts
are not necessarily fastest; use the same settings when comparing methods.
This configuration applies to Julia BLAS, not independently loaded C++/R/Python
BLAS runtimes. The native benchmark scripts retain their own thread settings.
