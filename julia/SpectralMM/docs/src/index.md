# SpectralMM

SpectralMM fits generalized linear and robust regression models with spectral
majorization and iterative linear solvers. The Julia frontend defaults to pure
Julia. R and Python use a shared C++ implementation; Julia can select it with
`backend=:cpp`.

Start with [Installation](installation.md) and [Julia examples](examples.md).
The [modeling interfaces](api.md) describe matrix and formula inputs in all three
languages. See [solvers and diagnostics](solvers.md) for numerical controls and
[post-fit inference](inference.md) for uncertainty estimates and their limitations.

The supported statistical accessors include `coef`, `fitted`, `residuals`,
`nobs`, `predict`, `vcov`, `stderror`, `confint`, `coeftable`, `deviance`, and
`loglikelihood`, where applicable. `modelmatrix` is not implemented for
`StatisticalResult`; retain the input design if it is needed after fitting.
