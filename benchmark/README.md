Run from Julia:
```julia
using Pkg
Pkg.activate("benchmark")
Pkg.develop(path="..")
Pkg.add(["BenchmarkTools","DataFrames","GLM","Distributions"])
include("benchmark/benchmark.jl")
```
