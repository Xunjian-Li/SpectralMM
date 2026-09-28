# SpectralMM.jl

`SpectralMM.jl` is a Julia package for large-scale statistical estimation problems that are solved through a sequence of weighted least-squares (WLS) or quadratic subproblems. The package combines matrix-free curvature operations, low-rank spectral majorization, and iterative linear solvers, with spectral information reused and updated across successive reweighting iterations.

The main computational goal is to avoid repeatedly forming and factorizing dense curvature matrices. Operations are expressed primarily through matrix-vector products and low-rank updates, so the same implementation can work with both dense and sparse design matrices.

## Main features

- **Spectral-MM** for iteratively reweighted statistical estimation.
- **Matrix-free curvature operations** without explicitly forming `X' * W * X` in the main spectral solver.
- **Low-rank spectral majorization** based on Lanczos iterations.
- **Spectral reuse and correction** across successive reweighting iterations.
- **PCG and PSD inner solvers** for the majorized quadratic subproblems.
- **Dense and sparse design matrices** through the same high-level interface.
- **GLM-style interface** using `Distribution` and `Link` objects from `Distributions.jl` and `GLM.jl`.
- **Formula interface** through `StatsModels.jl`.
- **Asymmetric and robust regression** including expectile, smooth quantile, pseudo-Huber, and Student-t models.
- Reference implementations based on **Cholesky** and matrix-free **Krylov methods** are included for numerical comparisons.

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
    Xfit,
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
    Xfit,
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
Xfit = hcat(sparse(ones(10_000)), X)

model = SpectralMM.glm(
    Xfit,
    y,
    Bernoulli(),
    LogitLink();
    rank=10,
    solver=:pcg,
)
```

The spectral basis itself is stored as a low-rank dense factor, while multiplication by the original design matrix retains its sparse structure.

## Structured quadratic problems: Poisson equation

The repository also contains `Poisson equation.ipynb`, illustrating how the same curvature-majorization principle can be used in a structured quadratic problem.

Consider the two-dimensional Poisson equation

```math
-\Delta u(x)=f(x), \quad x\in(0,1)^2,
```

with homogeneous Dirichlet boundary conditions. With `N` interior grid points per coordinate, the standard five-point discretization gives

```math
A u=b, \quad
A=\frac{1}{h^2}\left(I_N\otimes T_N+T_N\otimes I_N\right),
```

where

```math
T_N=\operatorname{tridiag}(-1,2,-1).
```

Writing

```math
A=D-R,
```

a structure-preserving curvature majorizer is

```math
B=D+\rho I=I_N\otimes M,
\quad
\rho=\frac{2}{h^2}\cos\left(\frac{\pi}{N+1}\right),
```

with

```math
M=\frac{1}{h^2}\operatorname{tridiag}(-1,\alpha,-1),
\quad
\alpha=4+2\cos\left(\frac{\pi}{N+1}\right).
```

Since `B - A` is positive semidefinite, the corresponding MM iteration is

```math
u^{(k+1)}=u^{(k)}+B^{-1}(b-Au^{(k)}).
```

The matrix `A` is applied by the five-point stencil without forming the `N^2 × N^2` Kronecker matrix. Moreover,

```math
M=\frac{1}{h^2}LL^\top
```

has a lower-bidiagonal Cholesky factor. The factorization is computed once and reused, so applying `B^{-1}` requires only forward and backward bidiagonal solves. The notebook compares the resulting MM iteration with CG and with PCG using the same majorizer as a preconditioner.

This example is intentionally structure-aware: it illustrates the broader curvature-majorization principle rather than replacing the Poisson operator by a generic low-rank approximation.

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

## Tests

Run the package tests with

```julia
using Pkg
Pkg.test("SpectralMM")
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
├── test/
│   └── runtests.jl
├── SpectralMM.ipynb
├── Poisson equation.ipynb
└── Project.toml
```

## License

SpectralMM.jl is released under the GNU General Public License version 3 (GPL-3.0). See the `LICENSE` file for details.
