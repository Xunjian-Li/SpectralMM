# Statistical API refactor: implementation and validation

Publication remains paused. Changes are on `codex/statistical-api`, not `main`.

## Scope and public interfaces

The numerical implementation remains shared. Formula parsing constructs the
same feature matrix consumed by matrix fitting, with intercept handling and
prediction metadata retained in the result.

| Language | Preferred GLM interface | Generic loss interface | Numerical diagnostics |
|---|---|---|---|
| Julia | `SpectralMM.glm(X, y, family, link)` or `glm(formula, data, family, link)` | `SpectralMM.fit(...)` | `diagnostics(model)` |
| R | `spectralmm_glm(X, y, family=binomial())` or `spectralmm_glm(formula, data=...)` | `spectralmm_fit(...)` | `spectralmm_diagnostics(model)` |
| Python | `spectralmm.glm(X, y, family="binomial", link="logit")` or `glm(formula, data=...)` | `spectralmm.fit(...)` | `model.diagnostics` |

Julia retains its default pure Julia backend and optional `backend=:cpp`.
R and Python use the C++ core. Results expose coefficients, response and link
predictions, fitted means, response residuals, optional covariance and inference,
observation counts, and supported deviance/likelihood statistics. Summaries show
statistical results first and a short optimization section afterward.

`start` is the preferred name; `beta0` remains supported. Existing generic fits,
legacy family identifiers such as `bernoulli` and `probit`, Python `coef_` and
raw numerical result access remain available. Family/link separation is the
preferred GLM interface. Advanced numerical overrides use language-specific
control objects; direct options remain available.

## Main files

- `src/model_api.jl`, `src/glm.jl`, `src/backend_api.jl`, `src/SpectralMM.jl`:
  Julia result metadata, StatsAPI methods and routing to existing solvers.
- `R/model.R`, `R/spectralmm.R`, `NAMESPACE`, `man/*.Rd`:
  R formulas, standard S3 model methods, controls and function documentation.
- `bindings/python/spectralmm/model.py`, `__init__.py`, `pyproject.toml`:
  Python statistical results and Patsy formula support.
- `test/model_api.jl`, `test/R/model-api.R`, `test/python/test_model_api.py`,
  `tools/check_api_consistency.py`: deterministic API regression checks.
- `tools/build_packages.py`, `.github/workflows/package-checks.yml`:
  independent package contents, installed-package tests and cross-language CI.
- `README.md`, `docs/Statistical_API.md`, installation/package guides and
  existing LaTeX guides: preferred matrix/formula usage and compatibility notes.

## Validation

Local macOS validation:

- Julia: all 512 assertions passed, including the existing 322 and 190 new
  statistical API assertions, exercised with both Julia and C++ backends.
- R: source-package check completed with zero errors and zero warnings;
  three local environment notes concerned pandoc, system HTML tidy and
  an Apple toolchain temporary file. Formula/matrix tests and comparisons
  with R's `glm` passed.
- Python: a wheel built from the source distribution installed successfully
  in a separate environment; all four installed-package tests passed.
- Cross-language comparison: six GLMs, each with coefficients, linear
  predictors, fitted means, objective, deviance, normalized likelihood and
  convergence status. Each comparison covers 1,248 scalar quantities.
  Maximum absolute differences against Python were 4.55e-13 (R) and
  5.69e-14 (each Julia backend).
- Formula checks cover equivalent matrices, explicit intercept columns,
  no-intercept formulas, categorical encoding, new-data prediction and
  rejection of unknown categories. Large-p checks preserve automatic
  inference skipping above 50 total coefficients.

The cross-language test uses absolute tolerance 1e-6 and relative tolerance
1e-7 to accommodate the original stopping tolerances and platform-dependent
floating-point arithmetic. These thresholds are fixed across models.

The separate Python environment uses NumPy 2.0.2 with macOS Accelerate and
emitted matrix-product floating-point warnings reproducible without calling
SpectralMM. Outputs were finite and matched an independent contraction within
3.34e-16; tests passed. No warning filter or numerical workaround was added.

Platform CI runs Julia, R and Python on Ubuntu, macOS and Windows, followed by
an aggregate comparison of exported Linux results. Its result must be checked
before treating this development branch as ready for integration.

## Explicit limitations

- Observation weights and offsets are rejected: neither is implemented in
  the existing numerical core. Adding them would require a mathematical
  extension, outside this refactor.
- New binomial GLMs accept numeric binary responses. Grouped success counts
  and trials remain in the legacy generic interface. R fixed-shape negative
  binomial uses the documented name and `family_options`, not MASS family objects.
- Only response residuals are currently exposed. Residual loss models and
  generic Tweedie fitting do not claim a normalized likelihood or deviance.
- Gamma likelihood uses the specified or Pearson dispersion; it is a plug-in
  density, not a profiled maximum-likelihood estimate of Gamma dispersion.
- Some old Julia solver results do not retain their precise early-exit label.
  Diagnostics explicitly state when that exact reason is unavailable.
- Formula construction can be dense. Existing sparse matrix fitting remains
  sparse. Summary does not trigger covariance computation.
- Equivalent explicit-intercept matrix comparisons assume the same penalty
  specification. With ridge and `intercept=false`, an explicit constant column
  is an ordinary feature and follows the existing feature-penalty semantics.

## Numerical preservation

The canonical C++ core, family losses/gradients, MM/IRLS algorithms, spectral
majorization, Lanczos, PCG/CG, Hessian-vector products, stopping tolerances and
BLAS/Eigen selection were not changed by this refactor. The Linux LP64 BLAS /
libblastrampoline fix remains intact. Julia routing changes call the existing
implementations; the optional Krylov dimension is passed through explicitly
while preserving its previous default.

No registry submission, release tag or package publication was performed.
