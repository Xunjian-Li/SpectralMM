# SpectralMM

[![CI](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/package-checks.yml/badge.svg)](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/package-checks.yml)
[![Documentation](https://github.com/Xunjian-Li/SpectralMM/actions/workflows/documentation.yml/badge.svg)](https://xunjian-li.github.io/SpectralMM/)
[![Coverage](https://codecov.io/gh/Xunjian-Li/SpectralMM/branch/main/graph/badge.svg)](https://codecov.io/gh/Xunjian-Li/SpectralMM)

SpectralMM fits generalized linear and robust regression models using spectral
majorization and iterative linear solvers. Julia uses a pure Julia backend by
default; an optional C++ backend is shared with the R and Python interfaces.
Matrix and formula inputs are supported. Standard errors and Wald inference
are optional and computed automatically for eligible fits with at most 50
coefficients, including the intercept.

## Installation

- **Julia:** requires Julia 1.10 or later. While General registration is pending, install
  the Julia package from this repository using the command below.
- **Python:** `pip install SpectralMM`
- **R:** `remotes::install_github("Xunjian-Li/SpectralMM")`

```julia-install
using Pkg
Pkg.add(url="https://github.com/Xunjian-Li/SpectralMM", subdir="julia/SpectralMM")
Pkg.add(["Distributions", "GLM", "StatsModels"])
```

R and Python source installations require a C++17 compiler. The Julia C++
backend also requires a compiler on first use. Pure Julia fitting does not
compile C++, although build-tool artifacts are currently installed as dependencies.

## Julia quick start

```julia
using SpectralMM, Distributions, GLM, Random
rng = MersenneTwister(71)
X = randn(rng, 100, 3)
y = Float64.(rand(rng, 100) .< 0.5)
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink())
coef(model)
predict(model, X)
stderror(model)
```

Matrix fitting adds an intercept by default. Pass `intercept=false` when `X`
already contains the complete design. Formula fitting follows the formula's
intercept specification. Use `backend=:cpp` to select C++ explicitly.

```julia
using SpectralMM, Distributions, GLM, StatsModels
X = [sin(i*j) for i in 1:100, j in 1:3]
y = Float64.([isodd(i) for i in 1:100])
data = (y=y, x1=X[:, 1], x2=X[:, 2], x3=X[:, 3])
formula_model = SpectralMM.glm(@formula(y ~ x1 + x2 + x3), data,
                              Bernoulli(), LogitLink())
predict(formula_model, data)
```

## Documentation

The [user manual](docs/src/index.md) covers [modeling interfaces](docs/src/api.md),
[solvers](docs/src/solvers.md), [inference](docs/src/inference.md), and
[self-contained Julia examples](docs/src/examples.md), including sparse inputs.
See [installation](docs/src/installation.md) for language-specific requirements.
Use `SpectralMM.fit` for non-GLM losses such as `PseudoHuber()`.

## License

[GNU General Public License v3](LICENSE).
