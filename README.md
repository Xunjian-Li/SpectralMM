# SpectralIRLS.jl

A package-level implementation of iterative quadratic statistical estimation with:

- Spectral-MM with Lanczos restart, low-rank spectral majorization, PCG/PSD inner solves,
  and adaptive first-order spectral correction.
- Direct IRLS-Cholesky with weighted-Gram factorization reuse.
- Matrix-free IRLS Krylov solvers: CG, CGLS, CRLS, LSQR, and LSMR.
- GLMs plus asymmetric and robust regression families.

## Families

GLM:
`BernoulliLogit`, `BernoulliProbit`, `BinomialLogit`, `PoissonLog`,
`NegativeBinomialLog`, `GaussianIdentity`, `GaussianLog`, `GammaLog`,
`GammaInverse`, `TweedieLog`.

Asymmetric:
`SmoothQuantile(tau, epsilon)`, `Expectile(tau)`.

Robust:
`PseudoHuber(delta)`, `StudentT(nu, sigma)`.

`StudentT` deliberately uses the positive scale-mixture/MM weight

```math
w_i = \frac{\nu+1}{\nu\sigma^2+r_i^2},
```

not the exact Hessian, whose curvature becomes negative in the tails.

## Install locally

```julia
using Pkg
Pkg.develop(path="/path/to/SpectralIRLS")
Pkg.instantiate()
using SpectralIRLS
```

## Example

```julia
using SpectralIRLS, Random, LinearAlgebra

m,p=2000,500
X=generate_correlated_design(m,p;rho=0.5,seed=1)
beta=randn(p)/sqrt(p)
family=StudentT(4.0,1.0)
y,_,_=simulate_y_from_beta(X,beta,family;seed=3124,beta0=0.5)
Xfit=hcat(ones(m),X)

spectral=SpectralOptions{Float64}(k=5,krylovdim=20,ridge=1e-6,correction_tol=5e-2)
inner=InnerOptions{Float64}(solver=:pcg,maxiter=100,eta_max=0.6)
outer=OuterOptions{Float64}(maxiter=100,relgtol=1e-8,verbose=true)

fit=spectral_mm(Xfit,y;family=family,spectral=spectral,inner=inner,outer=outer)
```

## Tests

```julia
using Pkg
Pkg.test("SpectralIRLS")
```

## Benchmark

See `benchmark/benchmark.jl`. Benchmark-only dependencies (`BenchmarkTools`,
`DataFrames`, `GLM`) are intentionally kept out of the core package.
