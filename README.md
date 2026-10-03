# SpectralMM

`SpectralMM.jl` is a Julia package for large-scale statistical estimation problems that are solved through a sequence of weighted least-squares (WLS) or quadratic subproblems. The package combines matrix-free curvature operations, low-rank spectral majorization, and iterative linear solvers, with spectral information reused and updated across successive reweighting iterations.

The main computational goal is to avoid repeatedly forming and factorizing dense curvature matrices. Operations are expressed primarily through matrix-vector products and low-rank updates, so the same implementation can work with both dense and sparse design matrices.

## Statistical interfaces (development version)

Use the native statistical conventions of each language. These calls assume
`X` contains three feature columns and `dat` contains `y`, `x1`, `x2`, `x3`.

```julia
using SpectralMM, GLM, Distributions, StatsModels
m = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); rank=3, solver=:pcg)
f = SpectralMM.glm(@formula(y ~ x1 + x2 + x3), dat,
                  Bernoulli(), LogitLink(); rank=3, solver=:pcg)
coef(f); fitted(f); coeftable(f); diagnostics(f)
# Both routes default to Julia; backend=:cpp explicitly selects C++.
```

```r
library(SpectralMM)
m <- spectralmm_glm(X, y, family=binomial("logit"), rank=3, solver="pcg")
f <- spectralmm_glm(y ~ x1 + x2 + x3, data=dat,
                    family=binomial("logit"), rank=3, solver="pcg")
coef(f); fitted(f); summary(f); spectralmm_diagnostics(f)
```

```python
import spectralmm
m = spectralmm.glm(X, y, family="binomial", link="logit", rank=3, solver="pcg")
f = spectralmm.glm("y ~ x1 + x2 + x3", data=dat,
                   family="binomial", link="logit", rank=3, solver="pcg")
f.params; f.fittedvalues; f.bse; f.diagnostics
print(f.summary())
```

Formula parsing preserves coefficient names and categorical encoding for new
predictions. Advanced numerical settings belong in `SpectralMMControl`,
`spectralmm_control`, or `Control`; `rank` and `solver` remain direct arguments.
Use `start` for initial coefficients. Generic `fit`/`spectralmm_fit` remains
available for residual models and legacy aliases. Automatic inference retains
the 50-coefficient cutoff. Observation weights and offsets are explicitly
unsupported. See [the statistical API guide](docs/Statistical_API.md) for complete
examples, family/link mapping, prediction defaults and likelihood definitions.
Registry publication is paused while this interface is reviewed.

For non-GLM models, use the loss constructors with the generic fitting API:

```r
fit <- spectralmm_fit(y ~ ., data=dat, family=pseudo_huber(delta=1))
```

```python
fit = spectralmm.fit(X, y, family=spectralmm.PseudoHuber(delta=1.0))
```

Also available: R `expectile(q)`, `smooth_quantile(q, epsilon)`, `student_t(nu)`;
Python `spectralmm.Expectile(q)`, `spectralmm.SmoothQuantile(q, epsilon)`,
`spectralmm.StudentT(nu)`. Matrix and formula inputs share the same fitting path.
The existing string plus `family_options` interface remains supported; it cannot
be combined with a constructor object. These losses are not GLM families.

## Main features

- **Spectral-MM** for iteratively reweighted statistical estimation.
- **Matrix-free curvature operations** without explicitly forming `X' * W * X` in the main spectral solver.
- **Low-rank spectral majorization** based on Lanczos iterations.
- **Spectral reuse and correction** across successive reweighting iterations.
- **PCG and MM inner solvers** for the majorized quadratic subproblems.
- **Dense and sparse design matrices** through the same high-level interface.
- **GLM-style interface** using `Distribution` and `Link` objects from `Distributions.jl` and `GLM.jl`.
- **Formula interface** through `StatsModels.jl`.
- **Asymmetric and robust regression** including expectile, smooth quantile, pseudo-Huber, and Student-t models.
- Reference implementations based on **Cholesky** and matrix-free **Krylov methods** are included for numerical comparisons.

## Python and Julia installation

```sh
python -m pip install .
```

```python
import spectralmm
fit = spectralmm.fit(X, y, family="bernoulli")
```

```julia
using Pkg
Pkg.develop(path="/path/to/SpectralMM")
using SpectralMM
model = SpectralMM.fit(X, y; family=:bernoulli) # Pure Julia by default
model_cpp = SpectralMM.fit(X, y; family=:bernoulli, backend=:cpp)
```

Python installation builds and bundles the C++ library. Julia builds it during
the first explicit C++ use and reuses its cache. A C++17 toolchain is required for
source builds; users do not manually invoke CMake or locate Eigen/library files.
See [automatic installation](docs/Automatic_Installation.md) for remote installs,
platform scope and the distinction between original Julia and C++ methods.

## R package

Install the updated GitHub source with:

```r
install.packages("remotes")
remotes::install_github("Xunjian-Li/SpectralMM")
library(SpectralMM)
fit <- spectralmm_fit(X, y, family = "bernoulli")
summary(fit)
```

R automatically compiles the C++ core during installation. No CMake, manual
library loading or Julia runtime is needed for R usage. Source installation
requires a C++17 toolchain; R dependencies include Rcpp and RcppEigen.
For the current local checkout use `remotes::install_local("/path/to/SpectralMM")`.
The GitHub command uses the new package layout after these changes are published.
See the [R package guide](docs/R_Package.md) for a complete example and developer checks.

## Experimental C++ core

An opt-in [C++ prototype](cpp/README.md) provides a shared native solver with
Python and R wrappers for all 14 Julia family types, including the GLM, quantile,
expectile, PseudoHuber, and Student-t models, with dense and sparse inputs.
The Julia package exposes the C++ core through `SpectralMM.fit(X, y; family=:..., backend=:cpp)`
for all fourteen families. Existing original Julia methods remain available.
See the prototype guide for build instructions, examples, validation, and scope.

The [LaTeX quick-start guide](docs/usage_and_benchmark.tex) gives Python/R usage and
benchmark commands. The [final comparison](benchmark/native/results/family-performance/report.md)
and its LaTeX tables are retained; intermediate experiments have been cleaned up.

## Small-data examples

[Fourteen runnable model examples](examples/README.md) use the same 100-by-3 data
in Python, R and Julia, with fitted coefficients, inference and iteration logs.
The [standalone English LaTeX source](docs/Small_Model_Examples.tex) is ready for Overleaf.

## Installation

From the Julia package manager, the development version can be installed locally with

```julia
using Pkg
Pkg.develop(path="/path/to/SpectralMM")
Pkg.instantiate()
```

Then load the package with

```julia
using SpectralMM
```

## Generalized linear models

The high-level `glm` interface follows the calling convention of `GLM.jl`: the response distribution and link function are supplied explicitly.

```julia
using SpectralMM
using GLM
using Distributions
using Random

rng = MersenneTwister(3124)

n, p = 1000, 5
X = randn(rng, n, p)
β = [0.5, 1.0, -1.0, 0.5, 0.0, -0.5]

Xfit = hcat(ones(n), X)
η = Xfit * β
π = 1.0 ./ (1.0 .+ exp.(-η))
y = Float64.([rand(rng) < pi for pi in π])

model = SpectralMM.glm(
    X,
    y,
    Bernoulli(),
    LogitLink();
    rank=5,
    solver=:pcg,
    verbose=true,
)

coef(model)
predict(model)
```

Other supported GLM distribution-link combinations can be fitted by replacing the distribution and link objects. The current high-level interface supports Bernoulli-logit, Bernoulli-probit, Gaussian-identity, Gaussian-log, Poisson-log, Gamma-log, and Gamma-inverse models.

### Intercept convention

Matrix interfaces now accept **feature columns only** and fit an intercept by default:
Python `fit_intercept=True`, R `intercept=TRUE`, and Julia `intercept=true`.
Predict with the same feature columns; the fitted intercept is added automatically.
Ridge excludes the automatic intercept unless `penalize_intercept` is enabled.
`beta0` and full coefficient vectors include the intercept first. Python also exposes
`model.intercept_` and `model.coef_` (slopes only); R/Julia label it `(Intercept)`.

For existing code with a column of ones, pass `fit_intercept=False` (Python) or
`intercept=false` / `FALSE` (Julia/R). This keeps design-matrix semantics, including
penalizing every supplied column when ridge is nonzero. A duplicate constant column
triggers a warning, not silent removal. Zero feature columns fit an intercept-only
model. Native wrappers implement the added column implicitly; the original Julia
backend constructs a sparse-preserving design matrix.

Julia formulas control the intercept through `y ~ x` (included), `y ~ 0 + x`
(omitted), or `y ~ 1` (intercept only). Python/R/C++-backed Julia interfaces still accept
matrices only. Low-level `spectral_mm`, `irls_krylov`, `irls_cholesky` and existing
C ABI entry points keep explicit design-matrix semantics; Julia's low-level
`penalize_intercept=false` excludes the **first** supplied column (for
`spectral_mm`, set it in `SpectralOptions`).

### Formula interface

`SpectralMM.jl` also supports the `StatsModels.jl` formula interface.

```julia
using DataFrames
using StatsModels

 data = DataFrame(
    y=y,
    x1=X[:,1],
    x2=X[:,2],
    x3=X[:,3],
    x4=X[:,4],
    x5=X[:,5],
)

model = SpectralMM.glm(
    @formula(y ~ x1 + x2 + x3 + x4 + x5),
    data,
    Bernoulli(),
    LogitLink();
    verbose=true,
)

coef(model)
```

The returned `SpectralGLM` implements standard `StatsAPI` functionality including `coef`, `predict`, `fitted`, `response`, `modelmatrix`, and `nobs`.

## Asymmetric and robust regression

Residual-based models are fitted with `fit`.

```julia
model = SpectralMM.fit(
    X,
    y,
    PseudoHuber(1.0);
    rank=5,
    solver=:pcg,
    verbose=true,
)
```

Currently available families are

- `SmoothQuantile(tau, epsilon)`
- `Expectile(tau)`
- `PseudoHuber(delta)`
- `StudentT(nu)`

For Student-t regression, the MM construction uses a positive scale-mixture weight rather than the exact Hessian curvature, which can become negative in the tails.

## Low-level Spectral-MM interface

For direct control over the spectral approximation, inner solver, and outer iteration, use `spectral_mm` together with `SpectralOptions`, `InnerOptions`, and `OuterOptions`.

```julia
spectral = SpectralOptions{Float64}(
    k=5,
    krylovdim=20,
    ridge=1e-6,
    correction_tol=5e-2,
)

inner = InnerOptions{Float64}(
    solver=:pcg,
    maxiter=100,
    eta_max=0.6,
)

outer = OuterOptions{Float64}(
    maxiter=100,
    relgtol=1e-8,
    verbose=true,
)

result = spectral_mm(
    Xfit,
    y;
    family=PseudoHuber(1.0),
    spectral=spectral,
    inner=inner,
    outer=outer,
)
```

At each outer iteration, the method works with a local WLS curvature of the form

```math
H_k = X^\top W_k X,
```

but applies it matrix-free through products with `X` and `X'`. A low-rank approximation to the dominant spectral subspace is obtained by Lanczos iteration and used to construct a computationally tractable curvature majorizer. Dominant spectral information can be reused across successive outer iterations and corrected when the weights change sufficiently.

## Sparse design matrices

The main spectral solver accepts `AbstractMatrix` inputs and performs curvature operations through matrix-vector multiplication. Consequently, sparse matrices can be passed directly without converting the design matrix to dense form.

```julia
using SparseArrays

X = sprandn(10_000, 2_000, 0.01)

model = SpectralMM.glm(
    X,
    y,
    Bernoulli(),
    LogitLink();
    rank=10,
    solver=:pcg,
)
```

The spectral basis itself is stored as a low-rank dense factor, while multiplication by the original design matrix retains its sparse structure.

## Standard errors and Wald inference

High-level fits now offer optional post-fit inference. The default `inference="auto"`
(Python/R; `:auto` in Julia) attempts inference for at most 50 total coefficients,
including the intercept. Use `inference_max_p` to configure this engineering cutoff;
boolean true forces computation past the cutoff, and false disables it.

Python: `model.summary()`, `model.bse`, `model.pvalues`, `model.cov_params()`.
R: `summary(model)`, `vcov(model)`, `confint(model)`.
Julia: `stderror(model)`, `vcov(model)`, `confint(model)`, `coeftable(model)`.
The inference result records why a calculation was disabled, skipped or unavailable.

GLMs default to model-based covariance; residual models use HC1 sandwich covariance.
This optional stage forms full statistical matrices after the matrix-free fit.
It does not substitute the low-rank preconditioner for a covariance matrix.
Ordinary Wald results are withheld for ridge fits and failed validity checks.
See [inference guide](docs/README.md) for assumptions, options and references.

## Benchmarks

Dense and sparse scaling experiments are provided in

```text
benchmark/scaling.jl
benchmark/scaling_sparse.jl
```

The benchmark suite compares Spectral-MM with matrix-free Krylov methods (`CGLS`, `CRLS`, `LSQR`, and `LSMR`), direct Cholesky-based IRLS, and `GLM.jl` where applicable. Sparse experiments retain the sparse design matrix throughout the matrix-free solver path.

Run, for example,

```julia
include("benchmark/scaling.jl")
include("benchmark/scaling_sparse.jl")
```

Benchmark results are written to

```text
benchmark/scaling_results.csv
benchmark/scaling_sparse_results.csv
```

## Repository structure

```text
SpectralMM/
├── src/
│   ├── SpectralMM.jl
│   ├── families.jl
│   ├── spectral.jl
│   ├── lanczos.jl
│   ├── krylov.jl
│   ├── cholesky.jl
│   ├── glm.jl
│   ├── statsapi.jl
│   └── simulation.jl
├── benchmark/
│   ├── common.jl
│   ├── scaling.jl
│   └── scaling_sparse.jl
├── SpectralMM.ipynb
├── Poisson equation.ipynb
└── Project.toml
```

## Solver selection

High-level `glm` and `fit` support `:pcg` (alias `:spectral`), `:mm`, `:cg`,
`:cho` (alias `:cholesky`), `:cgls`, `:crls`, `:lsqr` and `:lsmr`.
Python/R accept the same names as strings. See [the unified guide](docs/README.md#choosing-a-solver)
for algorithm definitions and examples. Cholesky forms a full curvature matrix.

## License

SpectralMM.jl is released under the GNU General Public License version 3 (GPL-3.0). See the `LICENSE` file for details.


## Julia backend selection

The Julia package currently requires Julia 1.12 or later.
The main matrix API is `SpectralMM.fit(X, y; family=:gaussian, backend=:julia)`.
Pure Julia is the default. Select `backend=:cpp` explicitly to use the shared
C++ core; its first use builds the library automatically, and subsequent calls
reuse the cache. Installing or fitting with the Julia backend does not compile
C++ code. Build-tool dependencies are still resolved by the package manager.

```julia
julia_model = SpectralMM.fit(X, y; family=:pseudo_huber,
    family_options=(delta=1.0,), backend=:julia, solver=:pcg)
cpp_model = SpectralMM.fit(X, y; family=:pseudo_huber,
    family_options=(delta=1.0,), backend=:cpp, solver=:pcg)
julia_model.backend  # :julia
cpp_model.backend    # :cpp
coef(julia_model)
predict(cpp_model, X)
```

Both backends support all fourteen named families and the eight solvers. Common
matrix defaults are `intercept=true`, `maxiter=200`, `inner_maxiter=100`,
`gtol=1e-7`, `relgtol=1e-8`, `floor=1e-6`, and `solver=:pcg`. The default spectral
rank is `p-1` for `p<20` and 10 otherwise, where `p` includes the intercept.
Inference remains automatic for at most 50 coefficients. Backend-specific
advanced options are not interchangeable: for example, `trace=true` and
`krylovdim` are C++ interface options. Unsupported options raise an error rather
than switching the backend. Use an explicit `beta0` when comparing initialization.

`glm` and generic `fit` support matrix and formula input and default to pure Julia.
Use `backend=:cpp` explicitly to choose C++.
The earlier two-argument keyword API used C++; callers wanting that implementation
must now specify `backend=:cpp`. Legacy development loader shims have been removed; use the installed package
and request C++ explicitly. R and Python keep their C++ backend.


## Numerical correctness update (2026-10-01)

Probit objective/gradient consistency, rounding-scale line-search comparisons,
and convergence reporting on the final allowed step have been corrected.
The previously documented small-data inference discrepancies are resolved at
unchanged gradient thresholds. See the [numerical audit](benchmark/native/results/numerical-correctness-audit.md)
for before/after evidence and validation scope. Outdated archives under `output/` have been removed. Generate independent
packages from current sources with `python3 tools/build_packages.py`.
See [packaging instructions](docs/Packaging.md).
