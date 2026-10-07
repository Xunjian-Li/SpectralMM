# SpectralMM

[![CI](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/package-checks.yml/badge.svg)](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/package-checks.yml)
[![Documentation](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/documentation.yml/badge.svg)](https://xunjian-li.github.io/SpectralMM/)
[![Coverage](https://codecov.io/gh/Xunjian-Li/SpectralMM/branch/main/graph/badge.svg)](https://codecov.io/gh/Xunjian-Li/SpectralMM)

SpectralMM provides **Julia, Python, and R** interfaces for generalized linear
and robust regression using spectral majorization and iterative linear solvers.
All three interfaces support matrix and formula inputs, prediction, and optional
standard errors and Wald inference. Julia uses a pure Julia backend by default
and can select the C++ backend with `backend=:cpp`; Python and R use the shared
C++ backend.

## Installation

### Julia

Requires Julia 1.10 or later. Install from GitHub:

```julia-install id="4sqw23"
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SpectralMM", subdir="julia")
Pkg.add(["RDatasets", "Distributions", "GLM"])  # Used by the examples below
```

### Python

Requires Python 3.9 or later. Install from PyPI:

```sh id="dl4q2x"
python -m pip install SpectralMM
python -m pip install statsmodels  # Used by the examples below
```

### R

Requires R 4.0 or later. Install from GitHub:

```r id="87z7f1"
install.packages("remotes")
remotes::install_github("Xunjian-Li/SpectralMM", subdir="R")
```

R and Python source installations require a C++17 compiler (Rtools for R on
Windows); compilation is automatic. Julia's optional C++ backend also requires
a compiler on first use. Pure Julia fitting does not compile C++, although
CMake and Eigen artifacts are currently installed as dependencies.

## Logistic regression

The following examples fit the same logistic regression to the `birthwt`
dataset from the R `MASS` package. The binary response is low birth weight,
and the predictors are maternal age, maternal weight, and smoking status.
Matrix fitting adds an intercept by default.

### Julia

```julia id="c6zfz4"
using SpectralMM, RDatasets, Distributions, GLM

birthwt = dataset("MASS", "birthwt")
X = Matrix{Float64}(birthwt[:, [:Age, :LWt, :Smoke]])
y = Float64.(birthwt.Low)

model = SpectralMM.glm(X, y, Bernoulli(), LogitLink();
                       intercept=true, verbose=true)

coef(model)
predict(model, X; type=:response)
```

### Python

```python id="6h7d07"
import statsmodels.api as sm
import spectralmm

birthwt = sm.datasets.get_rdataset("birthwt", "MASS").data
X = birthwt[["age", "lwt", "smoke"]].to_numpy(dtype=float)
y = birthwt["low"].to_numpy(dtype=float)

model = spectralmm.glm(
    X, y, family="binomial", link="logit", intercept=True, verbose=True
)

print(model.params)
print(model.predict(X)[:5])
```

### R

```r id="ammyrc"
library(SpectralMM)
data("birthwt", package="MASS")

X <- as.matrix(birthwt[, c("age", "lwt", "smoke")])
y <- as.numeric(birthwt$low)

model <- spectralmm_glm(
  X, y, family=binomial("logit"), intercept=TRUE, verbose=TRUE
)

coef(model)
head(predict(model, X, type="response"), 5)
```

### Formula interface

The same model can also be specified using a formula. An intercept is included
by default in the formula specification.

#### Julia

```julia id="e3px4m"
using SpectralMM, RDatasets, Distributions, GLM

birthwt = dataset("MASS", "birthwt")

model = SpectralMM.glm(@formula(Low ~ Age + LWt + Smoke), birthwt,
                       Bernoulli(), LogitLink(); verbose=true)
```

#### Python

```python id="u55nr8"
model = spectralmm.glm("low ~ age + lwt + smoke", data=birthwt,
                       family="binomial", link="logit", verbose=True)
```

#### R

```r id="ybcj1r"
model <- spectralmm_glm(low ~ age + lwt + smoke, data=birthwt,
                        family=binomial("logit"), verbose=TRUE)
```

## Smoothed quantile regression

The following examples fit a smoothed median regression for fuel economy using
the `mtcars` dataset. The response is miles per gallon, and the predictors are
weight, horsepower, and displacement.

### Julia

```julia id="0b6szz"
using SpectralMM, RDatasets

mtcars = dataset("datasets", "mtcars")
X = Matrix{Float64}(mtcars[:, [:WT, :HP, :Disp]])
y = Float64.(mtcars.MPG)

model = SpectralMM.fit(X, y, SmoothQuantile(0.5, 0.1);
                       intercept=true, verbose=true)

coef(model)
predict(model, X)
```

### Python

```python id="s09iyh"
import statsmodels.api as sm
import spectralmm

mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data
X = mtcars[["wt", "hp", "disp"]].to_numpy(dtype=float)
y = mtcars["mpg"].to_numpy(dtype=float)

model = spectralmm.fit(
    X, y, family=spectralmm.SmoothQuantile(q=0.5, epsilon=0.1),
    intercept=True, verbose=True
)

print(model.params)
print(model.predict(X)[:5])
```

### R

```r id="c6ez2v"
library(SpectralMM)
data("mtcars", package="datasets")

X <- as.matrix(mtcars[, c("wt", "hp", "disp")])
y <- as.numeric(mtcars$mpg)

model <- spectralmm_fit(
  X, y, family=smooth_quantile(q=0.5, epsilon=0.1),
  intercept=TRUE, verbose=TRUE
)

coef(model)
head(predict(model, X), 5)
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
at `theta=4` in the example; Julia's `0.5` is a distribution constructor
argument, not the fitted mean.

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

For example:

```julia id="06k7j7"
SpectralMM.glm(X, y, Bernoulli(), LogitLink(); solver=:cho)
```

```python id="8n0dm4"
spectralmm.glm(X, y, family="binomial", link="logit", solver="cho")
```

```r id="8g3j2e"
spectralmm_glm(X, y, family=binomial("logit"), solver="cho")
```

See the [solver guide](docs/src/solvers.md) for numerical controls.

## Repository layout and local installation

Each language package is maintained directly in its own directory:

- `julia/`: Julia source, metadata, and tests; install locally with `Pkg.develop(path="julia")`.
- `python/`: Python source, metadata, and tests; install locally with `python -m pip install ./python`.
- `R/`: R source, metadata, help, and tests; install locally with `remotes::install_local("R")`.
- `cpp/`: shared C++ source. Run `python3 tools/sync_package_sources.py` after changes.
- `docs/`, `examples/`, and `benchmark/`: documentation, examples, and comparisons.

Run these commands from the repository root. The bundled C++ copies let each
language package install independently. CI checks that they match `cpp/`.

## Documentation

See the [full user manual](docs/src/index.md) for installation details,
modeling interfaces, solvers, diagnostics, and statistical inference.

## License

SpectralMM is licensed under the GNU General Public License v3; see
[LICENSE](LICENSE).