# Optional post-fit inference

The Python, R, original Julia and native Julia interfaces now support optional
coefficient standard errors, coefficient-wise Wald tests, two-sided p-values,
confidence intervals and a full covariance matrix.

## Automatic cutoff

`inference="auto"` (Python/R) or `inference=:auto` (Julia) is the default.
Inference is attempted when **p <= inference_max_p**, with a default of **50**.
Here p counts every fitted coefficient, including the intercept and expanded
categorical/interaction columns. Above the limit, inference is skipped **before
allocating the covariance matrix**. Set `inference=True/TRUE/true` to bypass the
size cutoff, or `False/FALSE/false` to disable the entire post-fit stage.
`inference_max_p=25` is an example of a more conservative local setting.

This is a SpectralMM engineering default, **not a statistical validity threshold
or an established universal cutoff**. The documented R `summary.glm`, statsmodels
GLM and GLM.jl interfaces do not specify this common parameter-count cutoff.
Dense covariance needs 8 p² bytes per matrix (20 kB at p=50, 200 MB at p=5000),
with additional matrices for factorization and sandwich calculations. Dense work
is roughly O(n p² + p³), so n and matrix structure matter as well as p.
The optimizer remains matrix-free; the optional inference stage forms full
p-by-p statistical matrices and does not use the spectral preconditioner as
an approximate covariance. Sparse features remain sparse; full coefficient
covariances are generally dense.

## Usage

```python
model = spectralmm.fit(X, y, family="bernoulli", inference="auto", inference_max_p=50)
print(model.summary())
model.bse                       # standard errors, intercept first
model.pvalues                   # t/z reference p-values
model.cov_params()
model.conf_int()                # level chosen at fit time
model.inference["wald_chisq"]    # coefficient-wise H0: beta_j = 0
model.inference["wald_p_value"]  # asymptotic chi-square(1) p-values
```

Python imports SciPy only when inference is eligible; without SciPy the fit is
retained with an unavailable-inference reason. Install SciPy for inference.

```r
model <- spectralmm_fit(X, y, family="bernoulli",
                        inference="auto", inference_max_p=50)
summary(model)
vcov(model)
confint(model, level=.95)
model$inference$wald_chisq
```

```julia
using SpectralMM, GLM, Distributions, StatsAPI
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink();
                       inference=:auto, inference_max_p=50)
stderror(model)
vcov(model)
confint(model; level=.95)
coeftable(model)
model.inference.wald_chisq

# Explicit C++ backend:
model = SpectralMM.fit(X, y; backend=:cpp, family=:bernoulli, inference=true)
stderror(model)
```

The native Julia wrapper now uses Distributions.jl for reference distributions,
in addition to Julia standard libraries. The repository project already declares
this dependency. The statistical result's `coeftable` returns a `GLM.CoefTable`.
R/Python/Julia summary display methods limit the printed table to 20 rows;
this display limit is separate from the inference-computation cutoff.
All coefficients and inference arrays remain available programmatically.

## Statistical definitions

Let A be the full design, including any intercept. With independent observations,
GLM model-based covariance is `phi * inverse(A' W A)`, using the full expected
information weights `(d mu / d eta)^2 / V(mu)`, without the optimizer's weight
floor. `phi=1` for Bernoulli, Probit, Binomial, Poisson and negative binomial with
fixed theta. Gaussian (identity/log), Gamma (log/inverse) and Tweedie default to
the Pearson estimate `sum((y-mu)^2/V(mu))/(n-p)`. A positive `dispersion` option
fixes phi explicitly. Binomial trials are retained in the variance and weights;
residual degrees of freedom count independent rows, not individual trials.

`cov_type="auto"` selects `"model"` for GLMs and `"sandwich"` for the four
residual models (Julia uses symbols). Sandwich covariance is
`n/(n-p) * H^(-1) * B * H^(-1)`, where B is the sum of per-row score outer
products and H is the sum of score derivatives. This is the HC1 small-sample
factor; it is not a cluster/HAC correction. The optional GLM sandwich uses
observed score derivatives rather than expected-information weights.
Student-t uses its actual score derivative, which can be negative per row,
**not** its positive MM curvature. SmoothQuantile inference concerns the
smoothed objective at the fixed smoothing parameter, not an unsmoothed quantile
regression estimator. Family tuning/shape/scale parameters are treated as fixed.
No uncertainty from estimating theta, nu, sigma or other tuning parameters is
included. Model-based covariance is not offered for the four residual models.
A dispersion override is not applicable to sandwich covariance.

The reported t/z statistic is `beta_j/SE_j`; the confidence interval is the
estimate plus/minus the corresponding reference quantile times SE. By default,
model-based inference with estimated dispersion uses Student t with n-p degrees
of freedom; otherwise the reference is standard normal. This follows R's
`summary.glm` convention. `use_t=False/FALSE/false` requests asymptotic z inference,
which matches the default GLM convention in statsmodels/GLM.jl more closely;
`use_t=True/TRUE/true` forces a t reference. `level=.95` controls fitted intervals.
`wald_chisq` is the squared statistic with an asymptotic chi-square(1) p-value in
`wald_p_value`. When a t reference is selected, this asymptotic Wald p-value is
distinct from the t-based `p_value` displayed in the coefficient table.

## Availability and reproducibility

The inference object always reports `status` and `reason`: `ok`, `disabled`,
`skipped` (size), or `unavailable` (numerical/statistical conditions). Failed
inference does not change fitted coefficients or solver diagnostics. Inference
requires the final gradient to meet the configured fitting tolerance, n>p,
no ridge penalty, and a positive-definite, adequately conditioned information
matrix. It is withheld for detected binary separation, singular designs, zero
estimated dispersion and non-finite curvature. Separation screening detects
separation exhibited by the fitted predictor; it is not an exhaustive separation
LP test. Requiring n>p and full rank does not by itself guarantee reliable
finite-sample inference. No pseudoinverse or optimizer regularization is used to
manufacture ordinary Wald results for unidentified coefficients.

For step-stopped fits, inspect convergence and, if appropriate, refit with
`accept_stalled=False` and `accept_negligible=False` in Python (R logical values).
Original Julia exposes `accept_stalled=false`; native Julia exposes both flags.
Forcing `inference=true` bypasses only the size cutoff, not the validity checks.
Ridge-penalized inference would require a separately defined approximation or
bias correction and is intentionally not reported as ordinary Wald inference.

All native benchmark scripts set `inference=False/FALSE`; original Julia
benchmarks call the low-level optimizer, which has no automatic inference.
Archived timing results remain unchanged and do not measure this new stage.



## References

- [R summary.glm](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.glm.html): scale, standard errors and t/z convention.
- [statsmodels GLMResults](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html): covariance and inferential accessors.
- [GLM.jl examples](https://juliastats.org/GLM.jl/latest/examples/): inference and its default z/t differences from R.
- [sandwich: bread and meat](https://sandwich.r-forge.r-project.org/reference/sandwich.html): estimating-function covariance construction.

