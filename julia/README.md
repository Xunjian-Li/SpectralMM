# SpectralMM for Julia

Julia 1.10 or later is required. Pure Julia is the default backend;
use `backend=:cpp` to select the shared C++ implementation.

```julia-install
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SpectralMM", subdir="julia")
Pkg.add(["RDatasets", "Distributions", "GLM"])
```

## Logistic regression

```julia
using SpectralMM
using RDatasets
using Distributions
using GLM

birthwt = dataset("MASS", "birthwt")

X = Matrix{Float64}(birthwt[:, [:Age, :LWt, :Smoke]])
y = Float64.(birthwt.Low)

model = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); intercept = true, verbose = true)
```

## Smoothed quantile regression

```julia
using SpectralMM
using RDatasets

mtcars = dataset("datasets", "mtcars")

X = Matrix{Float64}(mtcars[:, [:WT, :HP, :Disp]])
y = Float64.(mtcars.MPG)

model = SpectralMM.fit(X, y, SmoothQuantile(0.5, 0.1); intercept = true, verbose = true)
```

From a local repository checkout, use `Pkg.develop(path="julia")` and run
`julia --project=julia -e 'using Pkg; Pkg.test()'` from the repository root.
Edit Julia code directly in `src/`. The `cpp/` directory contains generated
copies of the shared native sources; update those through the repository's
`tools/sync_package_sources.py` script.

See the [full documentation](https://xunjian-li.github.io/SpectralMM/).
