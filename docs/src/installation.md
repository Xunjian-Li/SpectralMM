# Installation

## Julia

Julia 1.10 or later is required. General registration is pending. Install the
self-contained Julia subpackage from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SpectralMM", subdir="julia")
Pkg.add(["Distributions", "GLM", "StatsModels"])
```

Load it with `using SpectralMM`. Pure Julia is the default backend. An explicit
`backend=:cpp` call builds and caches the native library, requiring a C++17
compiler. CMake and Eigen artifacts are currently unconditional dependencies;
installing the package downloads them even if C++ is never used.

## Python

```sh
python -m pip install SpectralMM
```

```python
import numpy as np
import spectralmm
rng = np.random.default_rng(71)
X = rng.normal(size=(100, 3))
y = rng.binomial(1, 0.5, size=100)
model = spectralmm.glm(X, y, family="binomial", link="logit")
print(model.summary())
```

The source distribution requires a C++17 compiler. pip installs build tools
and runtime dependencies automatically. Formula input uses Patsy.

## R

```r
install.packages("remotes")
remotes::install_github("Xunjian-Li/SpectralMM", subdir="R")
library(SpectralMM)
set.seed(71)
X <- matrix(rnorm(300), 100, 3)
y <- rbinom(100, 1, 0.5)
model <- spectralmm_glm(X, y, family=binomial("logit"))
summary(model)
```

Source installation uses R's build system and requires a C++17 compiler
(Rtools on Windows). Rcpp and RcppEigen are package dependencies. CRAN
registration is pending. Use `help(package="SpectralMM")` for installed help.

## Local development

Run these commands from the repository root to install a language package:

```sh
python -m pip install ./python
R CMD INSTALL R
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

R source installation requires Rcpp and RcppEigen in the active R library.
Use `julia --project=julia` to start a Julia session in the local package environment.
To use the local Julia package from another environment (such as a notebook), run
`Pkg.develop(path="/absolute/path/to/SpectralMM/julia")` in that environment.

Language code is maintained in `R/`, `python/` and `julia/`. Shared C++ files
are maintained in root `cpp/`; synchronize their bundled copies with
`python3 tools/sync_package_sources.py` before committing.
