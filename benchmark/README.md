# Benchmarks

For Python/R native comparisons and final results, see [native/README.md](native/README.md).
The English LaTeX usage/reproduction guide is [here](../docs/usage_and_benchmark.tex).

## Original Julia benchmark

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add("BenchmarkTools")'
julia --project=. benchmark/scaling.jl
julia --project=. benchmark/scaling_sparse.jl
```

These scripts use `benchmark/common.jl` and include their own timing loop.
They write the original scaling CSV outputs; the existing CSV files are retained.
