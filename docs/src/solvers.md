# Solvers and diagnostics

## Choosing a solver

The snippets below assume `X` and `y` have already been constructed, as in
the [examples](examples.md), and the relevant package has been imported.

All high-level fitting functions accept `solver`. Python and R use strings;
Julia uses symbols (for example, `solver=:cho`).

| Value | Algorithm | Forms a full curvature matrix? |
|---|---|---|
| `pcg` or `spectral` | Spectral-preconditioned conjugate gradient (default) | No |
| `mm` | Spectral MM: preconditioned gradient updates with a quadratic line search | No |
| `cg` | Conjugate gradient; Jacobi scaling by default | No |
| `cho` or `cholesky` | Direct Cholesky solve of each weighted least-squares subproblem | Yes |
| `cgls`, `crls`, `lsqr`, `lsmr` | Alternative Krylov solvers | No |

Julia `glm` and positional-family `fit` dispatch to Julia solvers. The unified
`SpectralMM.fit` defaults to Julia; select `backend=:cpp` to use C++. Python and
R use C++. Both backends support all 14 families. Formula input is supported in all three languages, and with both Julia backends.
The low-level Julia `spectral_mm` function is specifically for `:pcg` and `:mm`.

```python
model = spectralmm.fit(X, y, family="bernoulli", solver="cho")
model = spectralmm.fit(X, y, family="bernoulli", solver="mm")
model = spectralmm.fit(X, y, family="bernoulli", solver="cg", preconditioner="none")
```

```r
model <- spectralmm_fit(X, y, family="bernoulli", solver="cho")
model <- spectralmm_fit(X, y, family="bernoulli", solver="mm")
```

```julia
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); solver=:cg)
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); solver=:cho)
model = SpectralMM.fit(X, y, PseudoHuber(1.345); solver=:mm)
```

Cholesky requires quadratic parameter storage and cubic factorization work.
Both implementations retry a failed factorization with small diagonal jitter;
this stabilizes the solve and does not change the reported objective or authorize
inference for an unidentified model. MM and PCG share the spectral approximation,
workspace reuse and outer safeguards. CG defaults to Jacobi scaling; request
`preconditioner="none"` (Julia `:none`) for unpreconditioned CG.
Krylov tolerance keywords are `krylov_rtol`/`krylov_atol` in the native wrappers,
and `krylov_reltol`/`krylov_abstol` in original Julia. Set inner tolerances tightly
enough for the requested outer gradient accuracy.

Intercept, ridge handling, prediction and optional inference use the same model
interface for every solver. The inference cutoff remains 50 total coefficients.


---

## Iteration logs

All high-level interfaces support `verbose`, disabled by default. This includes
all eight solver choices, original Julia and the C++-backed Julia wrapper.

```python
model = spectralmm.fit(X, y, solver="pcg", verbose=True, trace=True)
rows = model.trace
```

```r
model <- spectralmm_fit(X, y, solver="cho", verbose=TRUE, trace=TRUE)
rows <- model$trace
```

```julia
model = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); solver=:cg, verbose=true)
native = SpectralMM.fit(X, y; backend=:cpp, family=:bernoulli, solver=:mm,
                             verbose=true, trace=true)
rows = native.trace
```

The common table is:

```text
Iter        LogLik    GradNorm     RelGrad  Inner  Stepsize  Spectrum
```

`LogLik` is the summed normalized log likelihood, including distribution constants
and excluding ridge penalties. It increases as likelihood improves. Gaussian uses
RSS/n for variance unless `dispersion` is supplied. Gamma uses the Pearson estimate
with n-p degrees of freedom unless `dispersion` is supplied; its displayed likelihood
is a plug-in likelihood, not a profile maximum over dispersion. These scales are
recomputed at each displayed point. Student-t uses its fixed `nu` and `sigma`.
For Pseudo-Huber, expectile, smoothed quantile, Tweedie, or fractional count data,
the column is `Objective`: summed model loss plus ridge penalty, without division
by n. Undefined likelihoods at individual points are displayed as `-`.

- Row 0 is the initial point. Later rows report accepted parameter updates.
- `GradNorm` and `RelGrad` refer to the **optimization objective**, including ridge,
  not to the displayed normalized likelihood. `RelGrad` uses the initial scale
  1 + initial gradient norm. The stopping rules and optimization objective are unchanged.
- `Inner` counts inner iterations (one factorization for CHO).
- `Stepsize` is the outer line-search multiplier, not the norm of the coefficient update.
- `Spectrum` reports spectral operations; nonspectral solvers display `-`.
- Inner and eigen residuals remain in internal diagnostics but are omitted from the table.
- The footer distinguishes gradient convergence from step-based termination and limits.

Python, R and native Julia additionally support `trace` independently of `verbose`.
`trace=true` retains structured rows; `verbose=true` prints them. Native logs and
original Julia's non-spectral logs are buffered until fitting finishes. Original
Julia spectral logs are printed during fitting. Retained native trace rows include
per-step diagnostics and cumulative inner-iteration/restart/correction counts.
Detailed diagnostics may add matrix products when logging or tracing is enabled;
keep both off for performance comparisons. Original Julia's low-level solver
functions retain their existing `verbose=true` default; high-level fitting defaults
to quiet operation. Original Julia models retain their existing history arrays.

