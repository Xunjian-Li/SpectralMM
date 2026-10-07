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
Pkg.add(["RDatasets", "Distributions", "GLM"]) # Used by the examples below
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

The Julia example uses the `birthwt` dataset; the Python and R examples use
a synthetic 100-by-3 design and binary response. Matrix fitting adds an intercept
by default. See the manual for formula inputs and intercept controls.

### Julia

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

## Smoothed quantile regression

This Julia example fits the conditional median of fuel economy using `mtcars`.

```julia
using SpectralMM
using RDatasets

mtcars = dataset("datasets", "mtcars")

X = Matrix{Float64}(mtcars[:, [:WT, :HP, :Disp]])
y = Float64.(mtcars.MPG)

model = SpectralMM.fit(X, y, SmoothQuantile(0.5, 0.1); intercept = true, verbose = true)
```

## Supported models

All three interfaces support the following **14 model/link or loss choices**.
In the call templates below, `X` contains predictors and `y` is the appropriate
response for that model; do not reuse a binary response for every family.

### Generalized linear models (GLMs)

Supported families are Gaussian, Bernoulli/binomial, Poisson, Gamma,
negative binomial (fixed shape), and Tweedie (fixed power).

Use the arguments in each row with `SpectralMM.glm(X, y, ...)` in Julia,
`spectralmm.glm(X, y, ...)` in Python, or `spectralmm_glm(X, y, ...)` in R.
The imports in the logistic examples above also cover these calls.

| Model / link | Julia arguments | Python arguments | R arguments |
|---|---|---|---|
| Gaussian / identity | `Normal(), IdentityLink()` | `family="gaussian", link="identity"` | `family=gaussian("identity")` |
| Gaussian / log | `Normal(), LogLink()` | `family="gaussian", link="log"` | `family=gaussian("log")` |
| Bernoulli (binary) / logit | `Bernoulli(), LogitLink()` | `family="binomial", link="logit"` | `family=binomial("logit")` |
| Bernoulli (binary) / probit | `Bernoulli(), ProbitLink()` | `family="binomial", link="probit"` | `family=binomial("probit")` |
| Poisson / log | `Poisson(), LogLink()` | `family="poisson", link="log"` | `family=poisson("log")` |
| Gamma / log | `Gamma(), LogLink()` | `family="gamma", link="log"` | `family=Gamma("log")` |
| Gamma / inverse | `Gamma(), InverseLink()` | `family="gamma", link="inverse"` | `family=Gamma("inverse")` |
| Negative binomial / log | `NegativeBinomial(4, 0.5), LogLink()` | `family="negative_binomial", link="log", family_options={"theta": 4}` | `family="negative_binomial", link="log", family_options=list(theta=4)` |

Binary GLMs require 0/1 responses; Poisson and negative binomial use counts,
and Gamma requires positive responses. The negative-binomial shape is fixed
at `theta=4` in the example; Julia's `0.5` is a distribution constructor argument,
not the fitted mean.

Binomial success-count and Tweedie regressions are also GLMs. These two
specifications currently use the generic fitting interface:
`SpectralMM.fit(X, y; ...)`, `spectralmm.fit(X, y, ...)`, or
`spectralmm_fit(X, y, ...)`, with the following keywords:

| Model / link | Julia keywords | Python keywords | R keywords |
|---|---|---|---|
| Binomial success counts / logit | `family=:binomial, family_options=(trials=4,)` | `family="binomial", family_options={"trials": 4}` | `family="binomial", family_options=list(trials=4)` |
| Tweedie / log | `family=:tweedie, family_options=(power=1.5,)` | `family="tweedie", family_options={"power": 1.5}` | `family="tweedie", family_options=list(power=1.5)` |

Binomial `y` contains success counts (0 through 4 here), not proportions;
prediction returns expected counts. Trials can also be supplied per observation.
Tweedie uses nonnegative responses and `1 < power < 2`; normalized likelihood
and deviance accessors are not implemented for it.

### Non-GLM losses

For continuous responses, pass a loss object using
`SpectralMM.fit(X, y, loss)` in Julia, `spectralmm.fit(X, y, family=loss)` in
Python, or `spectralmm_fit(X, y, family=loss)` in R:

| Loss | Julia object | Python object | R object |
|---|---|---|---|
| Pseudo-Huber | `PseudoHuber(1.0)` | `spectralmm.PseudoHuber(delta=1.0)` | `pseudo_huber(delta=1)` |
| Expectile | `Expectile(0.25)` | `spectralmm.Expectile(q=0.25)` | `expectile(q=0.25)` |
| Smoothed quantile | `SmoothQuantile(0.25, 0.1)` | `spectralmm.SmoothQuantile(q=0.25, epsilon=0.1)` | `smooth_quantile(q=0.25, epsilon=0.1)` |
| Student-t | `StudentT(4.0, 1.0)` | `spectralmm.StudentT(nu=4.0)` | `student_t(nu=4)` |

These are example tuning values, not estimates of the tuning parameters.
Student-t uses unit residual scale here. Coefficients and predictions are
accessed as in the logistic examples; inference depends on the model and fit.

### Solvers

All model choices accept `solver`: a symbol in Julia (for example,
`solver=:cg`) or a string in Python/R (`solver="cg"`). Add this keyword to a
fitting call; Julia separates keywords with `;`.

| Solver | Method |
|---|---|
| `pcg` (default; alias `spectral`) | Spectral-preconditioned conjugate gradient |
| `mm` | Spectral majorization-minimization |
| `cg` | Conjugate gradient, with Jacobi scaling by default |
| `cho` (alias `cholesky`) | Direct Cholesky solve; forms a full curvature matrix |
| `cgls`, `crls`, `lsqr`, `lsmr` | Alternative iterative least-squares solvers |

For example: `SpectralMM.glm(X, y, Bernoulli(), LogitLink(); solver=:cho)`,
`spectralmm.glm(X, y, family="binomial", link="logit", solver="cho")`, or
`spectralmm_glm(X, y, family=binomial("logit"), solver="cho")`.
See the [solver guide](docs/src/solvers.md) for numerical controls.

## Documentation

See the [full user manual](docs/src/index.md) for installation details, modeling
interfaces, solvers, diagnostics, and statistical inference.

## License

SpectralMM is licensed under the GNU General Public License v3; see [LICENSE](LICENSE).
