# Statistical modeling API

SpectralMM uses the modeling idioms of each language. Formula and matrix inputs
share the same numerical fitting path. The API layer constructs the design,
retains metadata and reports statistical results; it does not replace the
existing MM/IRLS, Lanczos, PCG or Krylov algorithms.

This interface is available from the repository's `main` branch, including the
R/Python loss constructors below. Registry publication is a separate release step.
See [the validation report](API_Refactor_Validation.md)
for the implementation scope, test coverage and known limitations.

## Julia

```julia
using SpectralMM, GLM, Distributions, StatsModels, DataFrames
X = [sin(i*j) for i in 1:100, j in 1:3]
y = Float64.([isodd(i) for i in 1:100])
dat = DataFrame(y=y, x1=X[:,1], x2=X[:,2], x3=X[:,3])
m = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); rank=3, solver=:pcg)
f = SpectralMM.glm(@formula(y ~ x1 + x2 + x3), dat,
                  Bernoulli(), LogitLink(); rank=3, solver=:pcg)
coef(f); fitted(f); residuals(f); nobs(f)
vcov(f); stderror(f); coeftable(f); confint(f)
deviance(f); loglikelihood(f)
predict(f, dat)              # response mean
predict(f, dat; type=:link)  # linear predictor
diagnostics(f)
```

Julia defaults to `backend=:julia`; either input route supports `backend=:cpp`.
The formula retains the applied StatsModels schema. New table predictions use
that schema, including categorical contrasts. Matrix prediction uses the same
feature ordering as fitting. As before, matrix fitting adds an intercept unless
`intercept=false` is supplied. This is SpectralMM's established matrix convention;
a complete GLM.jl-style design should explicitly use `intercept=false`.

Advanced settings use `SpectralMMControl(maxiter=500, krylovdim=12)` (dimension
must be appropriate for the number of parameters). Existing direct keywords
remain accepted. Backend-specific options retain their existing meanings:
Julia's nonspectral Krylov tolerances are `krylov_reltol`/`krylov_abstol`, while
C++ uses `krylov_rtol`/`krylov_atol`. Unsupported solver/backend combinations
raise errors; they are not silently translated into different algorithms.

## R

```r
library(SpectralMM)
X <- outer(1:100, 1:3, function(i,j) sin(i*j))
y <- as.numeric(1:100 %% 2 == 1)
dat <- data.frame(y=y, x1=X[,1], x2=X[,2], x3=X[,3])
m <- spectralmm_glm(X, y, family=binomial("logit"), rank=3, solver="pcg")
f <- spectralmm_glm(y ~ x1 + x2 + x3, data=dat,
                    family=binomial("logit"), rank=3, solver="pcg")
coef(f); fitted(f); residuals(f); nobs(f)
vcov(f); confint(f); summary(f)
deviance(f); logLik(f)
predict(f, dat, type="response")
predict(f, dat, type="link")
spectralmm_diagnostics(f)
```

`predict()` defaults to **link** for `spectralmm_glm`, as in R's GLM interface.
Legacy `spectralmm_fit` retains its response prediction default. Omitting
`newdata` returns cached training predictions. Formula models retain terms,
factor levels, contrasts and coefficient names. New factor levels are rejected.

Use `spectralmm_control(maxiter=500, krylov_rtol=1e-8)` for advanced settings.
Matrix column names are preserved. Formula intercepts are controlled by `1` or
`0` in the formula, not an additional intercept argument.

## Python

```python
import numpy as np
import spectralmm
X = np.sin(np.arange(1, 101)[:, None] * np.arange(1, 4))
y = (np.arange(1, 101) % 2 == 1).astype(float)
dat = dict(y=y, x1=X[:, 0], x2=X[:, 1], x3=X[:, 2])
m = spectralmm.glm(X, y, family="binomial", link="logit", rank=3, solver="pcg")
f = spectralmm.glm("y ~ x1 + x2 + x3", data=dat,
                   family="binomial", link="logit", rank=3, solver="pcg")
f.params; f.bse; f.pvalues; f.fittedvalues; f.resid_response; f.nobs
f.cov_params(); f.conf_int(alpha=.05)
f.deviance; f.llf
print(f.summary())
f.predict(dat)                  # conditional mean (default)
f.predict(dat, which="linear")  # linear predictor
f.diagnostics
```

Patsy parses formulas and retains `DesignInfo`, including stateful transforms
and categorical encoding. A dictionary of arrays works; pandas is not required.
With a formula fit, `predict(newdata)` applies the stored design specification;
`transform=False` accepts an already encoded feature matrix. Matrix fitting
adds an intercept unless `fit_intercept=False`; formulas control their intercept.
`param_names` preserves formula or DataFrame column names and otherwise uses
stable `x1`, `x2`, ... names. `params` always contains the full coefficient vector.

Advanced settings use `spectralmm.Control(maxiter=500, krylov_rtol=1e-8)`.
`coef`, `coef_`, `intercept_`, `info`, `trace` and existing inference accessors
remain available for compatibility. `coef_` excludes an automatically fitted
intercept; prefer `params` in new statistical code.

## Family and link

| Statistical family | Supported links | Legacy core identifiers |
|---|---|---|
| Gaussian | identity, log | gaussian, gaussian_log |
| Binomial (binary response) | logit, probit | bernoulli, probit |
| Poisson | log | poisson |
| Gamma | log, inverse | gamma, gamma_inverse |
| Negative binomial (fixed shape) | log | negative_binomial |

R accepts native family objects. For fixed-shape negative binomial, use
`family="negative_binomial"`, `family_options=list(theta=4)`; Python uses the
same family name and a dictionary. Julia accepts `NegativeBinomial(4, .5)` with
`LogLink()`; its shape is fixed and the distribution's probability does not
supply the fitted mean. Only listed links are supported.

The new binomial GLM interface supports numeric 0/1 responses. Grouped responses,
proportions and two-column successes/failures are not implemented by this v0.1
GLM interface. Legacy generic `fit` with `family=binomial` and `trials` remains
available with **success-count** and expected-count semantics. The legacy
`probit` identifier remains an alias, not a new statistical distribution.

## Non-GLM models

The recommended R/Python non-GLM API uses lightweight loss specifications:

```r
fit <- spectralmm_fit(y ~ ., data=dat, family=pseudo_huber(delta=1))
# Also: expectile(q=.25), smooth_quantile(q=.25, epsilon=.1), student_t(nu=4)
```

```python
fit = spectralmm.fit(X, y, family=spectralmm.PseudoHuber(delta=1.0))
# Also: Expectile(q=.25), SmoothQuantile(q=.25, epsilon=.1), StudentT(nu=4)
# Each class is exported from spectralmm; formula strings work with data= too.
```

These objects only normalize to existing family names and options. Public `q`
maps to internal `tau`; `epsilon` maps to `smoothing`. They work with the same
`rank`, `solver`, control and inference arguments as string specifications.
For example, `rank=5` requires at least six total coefficients.
The old string plus `family_options` API remains supported without deprecation.
Do not supply `family_options` with an object, including an empty mapping/list
or `None`/`NULL`; that raises an error. StudentT uses the existing unit residual
scale; custom `sigma` remains available through the string API. The new
PseudoHuber constructor defaults to `delta=1`; the old string path retains its
existing default `delta=1.345` when omitted. No Julia API or solver default changes.

Use the generic `fit` / `spectralmm_fit` interface for SmoothQuantile, Expectile,
PseudoHuber and StudentT residual models. Formula input is supported there too,
using `family=:pseudo_huber` (Julia) or `family="pseudo_huber"` (R/Python).
Existing Julia positional-family calls such as `fit(X,y,PseudoHuber())` remain
available. These result summaries describe regression losses, not GLM families.
Generic Tweedie fitting also remains available. A normalized Tweedie likelihood
and deviance are not implemented in the new result layer.

## Starts, unsupported specifications and missing data

`start` is the preferred initial coefficient vector, in the displayed coefficient
order, including any intercept. `beta0` remains an alias; supplying both raises
an error. Formula input does not accept a separate intercept flag. R/Python
formulas reject missing observations rather than silently dropping rows; Julia
uses StatsModels validation and does not introduce an implicit row-drop policy.

The numerical core does not implement observation weights or offsets. Non-null
`weights`/`offset` (also Python `freq_weights`/`var_weights`) and formula offsets
are rejected. There is no response rescaling, hidden offset coefficient or
ambiguous weighting convention. These features require a separately reviewed
mathematical extension, outside this API refactor.

## Statistical output versus numerical diagnostics

`verbose` prints runtime iterations. Summary displays family/link, observations,
coefficients, optional inference and fit statistics, followed by a short solver
section. Diagnostics report objective, gradients, solver, rank, iterations,
backend and numerical settings separately. C++ stop codes have explicit labels.
Older Julia solver results do not retain every early-stop label; when the exact
reason is unavailable it is reported as such, rather than inferred incorrectly.

Deviance is unscaled and excludes ridge penalties. Objective retains the exact
solver objective, including its normalization and penalty. Likelihood includes
distribution normalizing constants and excludes the penalty. Gaussian uses
RSS/n for the likelihood scale unless an explicit dispersion is supplied.
Gamma uses the supplied or Pearson n-p dispersion: its reported likelihood is
a plug-in density, not the profiled Gamma maximum likelihood used by some GLM
packages. Discrete normalized likelihoods require integer observations. Residual
losses do not acquire a fictional deviance or likelihood.

Covariance and standard errors retain the existing inference policy: automatic
only for at most 50 total coefficients, subject to the original validity checks.
Neither summaries nor diagnostics trigger covariance computation. Missing
inference is explicitly reported, and covariance access raises an informative
error. Cached predictions/residuals require O(n) storage; formula design
construction may be dense, while sparse matrix fitting stays sparse.

References: [GLM.jl](https://juliastats.org/GLM.jl/stable/),
[StatsModels formulas](https://juliastats.org/StatsModels.jl/stable/formula/),
[R glm](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html),
[statsmodels GLMResults](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html).
