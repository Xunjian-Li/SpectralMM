# Validation summary

The native family extension was validated on macOS arm64 with Eigen 3.4.0,
Julia 1.12.6, R/Rcpp and NumPy/SciPy. Production uses the default Eigen backend.
The implementation retains the validated HQ/gradient/workspace reuse and Julia
projected-correction reuse. No timing superiority over Julia is assumed.

- Julia test suite after projection reuse: 182 assertions passed.
- Before/after Julia projection audit: all 13 result fields matched in 36 fits
  across nine model types, dense/sparse designs and MM/PCG inner solvers.
- Historical intermediate audit scripts, snapshots and logs were removed.
  Development test scripts were also removed in the final cleanup.
- Final performance data, including failures and per-sample timings, are in
  `../benchmark/native/results/family-performance/`.

## Family extension — 2026-09-29

The Release C++ library now supports all 14 family types in `src/families.jl`.
Python and R expose their model parameters using `family_options`. The original
C ABI struct layouts and entry points are retained; `smm_fit_family` is additive.

Validation on synthetic n=200, p=8 designs, ridge=0.2 and floor/rho=1e-6:

- CTest: 1/1 passed.
- Python unittest suite: 19/19 passed, with no skipped tests in the final run.
- Family comparison: 14 families × 2 storage formats × 6 solvers = 168 fits.
  All converged under the configured stopping rules. Maximum coefficient absolute
  difference from Julia Spectral-PCG references: 6.59930881941051e-7.
  Objective values were checked to rtol/atol 2e-8; initial objective and gradient
  norms were independently checked against Julia to rtol/atol 2e-12.
- R comparison: all 168 fits matched Python, maximum coefficient absolute
  difference 4.440892e-16 using identical binary inputs. An initial CSV-based
  comparison had a 2.93e-7 Probit difference; sharing exact input bits eliminated
  this difference without loosening the final cross-wrapper tolerance (1e-11).
- Existing R Gaussian/Bernoulli and stopping tests passed.
- Existing Julia native wrapper tests: 22 parity and 7 stopping checks passed.
- Parameter/domain rejection and prediction behavior are covered, including
  scalar/vector binomial trials, nonpositive scales, invalid probabilities and
  infeasible inverse-Gamma initial predictors.

These checks validate model semantics and wrapper agreement, not bitwise agreement
with Julia's entire optimization trajectory or runtime rankings on article-sized
data. Student-t retains the Julia positive MM surrogate; Probit retains its erf
approximation. The standalone Julia native wrapper was not expanded in this change.

The following results describe completed checks; their development scripts and fixtures have been removed. NumPy on this host emits
floating-point warnings in some matmul calls; the tests additionally check finite
predictions and numerical agreement, and all assertions passed.

## Automatic intercept — 2026-09-29

Python, R, native Julia and original Julia high-level matrix interfaces now fit
an intercept by default. The C++ core implements the intercept through matrix
products without forming an augmented design; original Julia preserves sparse
storage when augmenting. Ridge excludes automatic/formula intercepts by default.
Legacy explicit-design benchmark calls now explicitly disable automatic intercept.
Existing archived timings were not regenerated.

Validation:

- Release CTest passed, including the additive intercept API and invalid flags.
- Python: 24 tests passed, no skips. Independent ridge least-squares and logistic
  references, all six solvers, dense/CSC, intercept-only fits, no-intercept fits,
  explicit-design equivalence across 14 families, and input/prediction validation.
- Julia: 267 assertions passed in the original Julia suite, including 85 new intercept
  assertions for PCG/MM, formulas with/without/only intercept, sparse storage,
  predictions, and all five alternative Krylov methods plus Cholesky.
- Family fixtures exercise both penalized and unpenalized leading intercepts.
  Unpenalized native/Julia maximum coefficient error: 6.267326031483833e-7
  over 14 families x 2 storage formats x 6 solvers.
- R/Python: 336 fits matched, maximum coefficient error 4.440892e-16.
  R independent ridge/intercept-only tests and legacy stopping checks passed.
- Native Julia: 15 new intercept checks, 22 parity and 7 stopping checks passed.
- The 5 new Python intercept tests also passed with Eigen's inner-loop heap
  allocation guard enabled, exercising all six native solvers.

The penalty mask is applied consistently to objectives, gradients, Jacobi scaling,
Lanczos/correction products and augmented least-squares operators. Julia's
least-squares Krylov correction RHS now also includes the current coefficients'
penalty contribution, matching the penalized gradient used by CG.

## Optional post-fit inference — 2026-09-29

Added the separate `smm_infer` C ABI and optional high-level inference in Python,
R, native Julia and original Julia. Automatic cutoff: 50 parameters including
the intercept; explicit true bypasses only this size gate. Benchmark wrappers
explicitly disable inference and archived timings were not overwritten.

- Independent GLM comparisons: Gaussian identity/log, Bernoulli logit/probit,
  Poisson, Gamma log/inverse, negative binomial (fixed theta), Tweedie and Binomial
  with vector trials; standard errors, intervals and p-values compared to
  statsmodels. R GLM checks include an explicit unit dispersion for fixed-theta
  negative binomial, matching the model definition rather than estimating an
  additional overdispersion parameter.
- Four residual-family sandwich matrices checked against analytic score/derivative
  formulas, including true Student-t curvature; all six native solvers checked.
- GLM sandwich matrices also checked against statsmodels observed Hessians and
  per-observation scores for nine families, with the HC1 correction. Probit uses
  a small absolute tolerance for the original Julia normal-CDF approximation.
- R/Python covariance comparisons across all 14 families and dense/CSC input:
  maximum absolute covariance difference 9.714451e-17.
- Original Julia/C++ covariance comparisons across all 14 families: 36 assertions
  passed, including native Julia accessors.
- Original Julia full suite: 305 assertions passed (267 existing/intercept + 38
  new inference assertions). Model coefficients are identical with inference on
  and off in the explicit invariance checks.
- Python suite: 28 tests passed/skipped, with one pre-existing fixture-dependent
  family fitting oracle skipped because its regenerated data was absent. The
  new inference fixtures and all four inference test groups ran successfully.
- Release CTest passed, including closed-form intercept-only covariance, metadata
  and invalid inference arguments.

Checks cover auto/forced/off modes, actual p=50/51 cutoff accounting, fixed
dispersion, t/z references, singular designs, insufficient convergence and
separated binary data. Ordinary unpenalized Wald inference is intentionally
unavailable for ridge fits. Full statistical assumptions are in `../docs/README.md`. These tests establish numerical agreement,
not a finite-sample coverage guarantee or a new performance ranking.

## Unified solver selection and English documentation — 2026-09-30

High-level original Julia, native Julia, Python and R now accept PCG/spectral,
CG, Cholesky/cho, MM, CGLS, CRLS, LSQR and LSMR. Original Julia dispatches to its
existing implementations. The C++ core adds direct Cholesky and spectral MM;
C ABI solver IDs 5 and 6 extend the existing options structure without changing
its layout. Cholesky forms the full weighted curvature matrix. MM uses the
spectral inverse with quadratic line-search gradient updates, without conjugate
directions. Existing archived performance tables were not rerun.

Temporary validation scripts were run outside the repository:

- Clean Release C++ build succeeded.
- Python: 168 fits across 14 families, dense/CSC input, PCG/CG/Cholesky/MM and
  aliases agreed with Cholesky coefficients within 2e-6 absolute/relative tolerance.
  Strict comparisons used Krylov rtol=1e-8 and atol=1e-12. Additional checks covered
  independent Gaussian least squares, inference, intercept-only and explicit-design
  fits, and unavailable inference for a singular Cholesky fit.
- R: all ten solver names/aliases passed Gaussian coefficient and inference checks
  for dense and sparse input against independent least squares.
- Julia: 160 high-level/native-wrapper checks passed, including sparse input,
  prediction, inference and original-Julia robust fits. Another 24 checks passed
  for logistic formula/matrix agreement, native agreement and intercept exclusion.
- Project text, including decoded JSON/notebooks, contains no Chinese text.
  Report generation now produces English; raw timing CSVs were preserved.
- Both English LaTeX guides compiled; the quick-start PDF was visually checked.

These are completed validation records, not new benchmark timings. Development
validation scripts remain outside the cleaned project.

## Unified iteration output — 2026-09-30

All eight solver methods now use the same table columns across Python, R,
original Julia and native Julia. Initial coefficients appear at row 0; PCG/MM
show the initial spectrum, while non-spectral methods show dashes. Accepted-state
loss and relative gradient are aligned, including returned endpoints after early
termination. Original Julia CG/CHO history arrays now retain the final endpoint
without discarding the initial gradient scale. Verbose output remains optional.

The new `smm_fit_logged` entry point provides detailed structured rows without
changing any existing C ABI layout. Python/R/native Julia expose optional `trace`
and `verbose`; original Julia retains its history arrays and verbose control.
Additional diagnostic products are only evaluated when detailed native tracing
or verbose original non-spectral logging is requested. Histories are finalized
at returned coefficients on non-spectral Julia exits.

Temporary checks outside the repository passed:

- Python: 32 checks across eight solvers, dense/CSC inputs, normal stopping and
  iteration limits, plus an initially converged fit. Coefficients were identical
  with logging on and off; trace endpoints matched final diagnostics exactly.
- R: all eight solvers on dense/CSC inputs; table/footer gradient agreement and
  shared-input Python trace comparisons for PCG/CG/CHO/MM.
- Julia: 304 assertions across original/native interfaces, dense/sparse inputs,
  all eight solvers, iteration limits, logging invariance and final endpoint
  gradients independently recomputed from the returned coefficients.
- Another 24 native early-stop trace checks passed, and a direct legacy C ABI
  call verified the original trace layout and coefficient agreement.
- Release C++ build and updated English LaTeX manual compilation passed.

Archived performance measurements were not rerun or overwritten. Temporary test
scripts and generated inputs remain outside the cleaned repository.

## Julia library-handle cache

A warmed before/after comparison identified repeated shared-library loading in
the standalone Julia native wrapper. The wrapper now retains validated handles.
Sixteen dense/CSC fits across all eight solvers were compared with the previous
wrapper: coefficient vectors, diagnostics, covariance matrices and full traces
were unchanged (NaN diagnostic placeholders compared with isequal). Repeated
calls used one cached handle. The temporary validation script was not retained.
See `benchmark/native/results/julia-wrapper-timing.md` for timings and limitations.

## Small-data examples

All fourteen families were run on the common 100-by-3 example data in Python, R
and original Julia. Python and R reached gradient convergence for every model;
original Julia Probit reached the 500-iteration limit at relative gradient
1.5064e-7 and correctly withheld inference. The other thirteen Julia fits reached
gradient convergence. Maximum coefficient differences against Python were
8.9e-16 for R and 9.7e-8 for Julia. The three low-level Julia GLM examples do not
attach inference. Saved outputs and English documentation are in `examples/`;
the standalone LaTeX example document compiled without overflow warnings.

## Wrapper initialization and allocation follow-up

Python and R each passed 224 before/after fit comparisons (14 families, eight
solvers, dense/CSC inputs), including coefficient, diagnostic, inference and
trace equality and input immutability. Julia passed 32 comparisons covering
dense/CSC/Float32/view inputs across all eight solvers. Constant-column warnings,
R integer conversion and Python library-cache/error-recovery checks passed.
The final serial timing and Julia allocation measurements are retained in
`benchmark/native/results/wrapper-optimizations.md`. Julia allocation decreased
substantially, but its timing benefit was not consistent; no universal speedup
is claimed. Temporary verification scripts remain outside the repository.

## Apple Accelerate BLAS build

The local Release library was rebuilt with SPECTRALMM_USE_BLAS=ON and
BLA_VENDOR=Apple. Dependency and undefined-symbol inspection confirms Accelerate
and BLAS matrix-operation references. The 224 Eigen-only/BLAS comparisons across
all fourteen families and eight solvers passed coefficient, loss and covariance
tolerances with matching convergence/inference status. Maximum coefficient
difference was 1.4685e-8; four Probit configurations changed iteration counts.
Fresh R and Julia processes each passed sixteen dense/CSC calls across all eight
solvers using the new binary; R inference and both final traces were checked.
Both English LaTeX guides compiled after updating their build instructions.
Timings and exact scope are retained in
`benchmark/native/results/blas-comparison.md` and its CSV companion.

## Standard R package

The repository root now supports automatic R installation through DESCRIPTION,
NAMESPACE, generated Rcpp registration and src/Makevars. A single canonical core
in src/native is compiled by both R and standalone CMake. A clean source tarball
installed without the standalone library; R CMD check reported Status: OK,
including compiled-code checks, examples and the PDF reference manual.
Installed-package tests passed 224 family/solver/dense/CSC combinations plus
inference, intercept, registration and input-preservation checks. Maximum
reference coefficient difference was 2.65506e-8. Standalone CMake rebuilt with
Accelerate after relocation, and Python and legacy R smoke checks passed.
See docs/R_Package.md for usage and platform scope. Temporary validation scripts
and compiler logs remain outside the repository.

## Automatic Python and Julia packages

Python now installs through PEP 517 and bundles its shared library in the
`spectralmm` package. Julia exposes the C++ interface through `SpectralMM.fit`,
with package-managed build tools and a source-keyed scratch cache. The Julia
interface now covers all fourteen C++ families, including family parameters.
Both interfaces passed 224 family/solver/dense/CSC reference comparisons on the
100-by-3 examples, with a maximum coefficient difference of 2.65506e-8. Standard
errors, convergence, prediction and final trace diagnostics were checked.

A wheel built from the Python source archive passed the same checks after an
offline installation outside the repository. A clean Julia checkout without a
precompiled core built automatically into a separate depot; a subsequent Julia
process reused the cache without rebuilding. The original Julia solver API
remains available and passed a smoke check. Validation was on macOS arm64,
Python 3.9 and Julia 1.12.6; no cross-platform or performance claim is implied.
See docs/Automatic_Installation.md for installation and artifact details.

Additional Julia checks passed invalid family/parameter/trials/beta0 rejection,
vector binomial trials and prediction overrides, disabled inference, Float32
conversion and equivalence with an explicitly supplied intercept column.

## Explicit Julia backend selection

The unified matrix API now defaults to `backend=:julia`; `backend=:cpp` selects
the shared C++ implementation. All fourteen families are supported by both,
including high-level Julia inference and response-scale prediction for negative
binomial, Tweedie and binomial models. Existing glm and positional-family calls
retain Julia execution. The legacy native shim explicitly selects C++.

A 448-fit comparison (14 families, 8 solvers, dense/CSC inputs, 2 backends)
checked coefficient and prediction agreement. Maximum coefficient difference
was 1.16157e-7. Standard errors agreed wherever Julia reported inference as
available. Eighteen Julia configurations withheld inference at the strict
requested gradient threshold: all sixteen Probit configurations, plus sparse
CHO negative-binomial and Student-t. This is retained numerical behavior, not
an assertion of identical convergence or inference availability across backends.
The 100-by-3 data, zero starts (gamma-inverse intercept 2), rank 3, maximum 500
iterations, gtol=1e-6 and relgtol=1e-8 match the comparison settings. No solver
algorithm was changed to conceal these differences.

The package retains a small public-interface regression suite in test/runtests.jl.
C++ compilation is deferred to first explicit use, so the default Julia path
can fit even with an invalid SPECTRALMM_LIBRARY override. Documentation describes
the keyword-interface default change and C++-specific advanced options.

The final retained public backend regression suite passed all 98 assertions.
It checks default-versus-explicit Julia selection without access to a native
library, legacy glm compatibility, dense/CSC coefficient and prediction
agreement, inference availability guards, backend display, Float32 conversion,
invalid backend/family parameters and binomial prediction trial lengths.
Both updated English LaTeX guides compiled successfully. Refreshed source
archives are retained under output/Julia, output/R and output/python; the Julia
and Python source archives were checked against the final backend implementation
and regression suite. These are local distributions, not registry publications.

## Numerical correctness fixes (2026-10-01)

Supersedes the unresolved-inference observations in the previous section.
Probit now uses consistent log-CDF objectives and scores, including stable tails;
Julia line searches allow machine-roundoff objective differences; and a
converged endpoint reached on the final allowed step is reported as converged.
The 448 backend comparisons now have no unavailable-inference exceptions at the
unchanged thresholds. All 322 retained regression assertions, 224 Python fits
and 224 installed R fits passed. Detailed causes, independent reference checks,
limitations and before/after results are in
benchmark/native/results/numerical-correctness-audit.md. Publication packaging
and performance optimization remain separate work; old output archives are
explicitly marked as pre-fix snapshots.
