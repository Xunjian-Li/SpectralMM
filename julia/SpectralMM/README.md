# SpectralMM

[![CI](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/package-checks.yml/badge.svg)](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/package-checks.yml)
[![Documentation](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/documentation.yml/badge.svg)](https://xunjian-li.github.io/SpectralMM/)
[![Coverage](https://codecov.io/gh/Xunjian-Li/SpectralMM/branch/main/graph/badge.svg)](https://codecov.io/gh/Xunjian-Li/SpectralMM)

SpectralMM provides **Julia, Python, and R** interfaces for generalized linear
and robust regression using spectral majorization and iterative linear solvers.
All three support matrix and formula inputs, prediction, and optional standard
errors and Wald inference. Julia defaults to a pure Julia backend and can select
C++ with `backend=:cpp`; Python and R use the shared C++ backend.

## Installation

### Julia

Requires Julia 1.10 or later. Install from GitHub:

```julia-install
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SpectralMM", subdir="julia/SpectralMM")
Pkg.add(["Distributions", "GLM"]) # Used by the example below
```

### Python

Requires Python 3.9 or later. Install from PyPI:

```sh
python -m pip install SpectralMM
```

### R

Requires R 4.0 or later. Install from GitHub:

```r-install
install.packages("remotes")
remotes::install_github("Xunjian-Li/SpectralMM")
```

R and Python source installations require a C++17 compiler (Rtools for R on
Windows); compilation is automatic. Julia's optional C++ backend also requires
a compiler on first use. Pure Julia fitting does not compile C++, although
CMake and Eigen artifacts are currently installed as dependencies.

## Logistic regression

These examples use the same 100-by-3 design and binary response. Matrix fitting
adds an intercept by default; each example returns four coefficients and fitted
probabilities. See the manual for formula inputs and intercept controls.

### Julia

```julia
using SpectralMM, Distributions, GLM
X = [sin(i*j) for i in 1:100, j in 1:3]
y = Float64.([isodd(i) for i in 1:100])
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink())
coef(model)
predict(model, X)[1:5]
```

### Python

```python
import numpy as np
import spectralmm
X = np.sin(np.arange(1, 101)[:, None] * np.arange(1, 4))
y = (np.arange(1, 101) % 2 == 1).astype(float)
model = spectralmm.glm(X, y, family="binomial", link="logit")
print(model.params)
print(model.predict(X)[:5])
```

### R

```r
library(SpectralMM)
X <- outer(1:100, 1:3, function(i, j) sin(i*j))
y <- as.numeric(1:100 %% 2 == 1)
model <- spectralmm_glm(X, y, family=binomial("logit"))
coef(model)
head(predict(model, X, type="response"), 5)
```

## Supported models

| GLM family | Supported links |
|---|---|
| Gaussian | identity, log |
| Binomial (binary response) | logit, probit |
| Poisson | log |
| Gamma | log, inverse |
| Negative binomial (fixed shape) | log |

The generic fitting interface also supports binomial success counts with known
trials, Tweedie regression with a log link, and the non-GLM losses **Pseudo-Huber,
expectile, smoothed quantile, and Student-t**. Use `fit` in Julia/Python or
`spectralmm_fit` in R for these models. See the manual for response conventions,
family parameters, and inference availability.

## Documentation

See the [full user manual](docs/src/index.md) for installation details, modeling
interfaces, solvers, diagnostics, and statistical inference.

## License

SpectralMM is licensed under the GNU General Public License v3; see [LICENSE](LICENSE).
