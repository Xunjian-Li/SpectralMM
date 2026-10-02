# C++ core prototype

This is an opt-in prototype alongside the original Julia implementation. It
implements all 14 family types currently defined in Julia, with float64 dense and CSC
matrices, a matrix-free WLS Hessian, reorthogonalized Lanczos, spectral reuse,
retained-subspace Rayleigh–Ritz correction, spectral-preconditioned PCG,
Nesterov extrapolation, and a safeguarded outer iteration. All iterations run
inside C++; no Julia runtime is needed by Python or R.

## Build

Normal Python and Julia users follow [automatic installation](../docs/Automatic_Installation.md).
The manual commands below are for standalone development and benchmarking.

Requirements: C++17 compiler, CMake >= 3.18, Eigen 3.4 headers. Eigen is an
external build dependency, not vendored or automatically downloaded. Set
`EIGEN3_INCLUDE_DIR` to the directory containing `Eigen/Core`:

```sh
cmake -S cpp -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0 \
  -DSPECTRALMM_USE_BLAS=ON
cmake --build build/native -j2
```

The current local build links Apple Accelerate BLAS and used `/tmp/eigen-3.4.0`.
On macOS, add `-DBLA_VENDOR=Apple` to explicitly select Accelerate. Use
`-DSPECTRALMM_USE_BLAS=OFF` for an Eigen-only comparison. BLAS affects supported
dense operations; sparse matrix products still use Eigen. Restart Python/Julia
and restart or reload the native bridge in R after rebuilding a loaded library. For a lasting setup, install
Eigen or point CMake to a persistent checkout. Existing local binaries can be
used immediately. On macOS the output is `build/native/libspectralmm.dylib`;
on Linux it is `libspectralmm.so`. Set `SPECTRALMM_LIBRARY` to an absolute path
to use a different build. macOS arm64 is tested; Linux/Windows packaging and
binary distribution are not yet validated.

## Python

Requires NumPy; SciPy is needed for sparse matrices and statistical inference.
The initial wrapper uses standard-library `ctypes`, with an explicit versioned
C ABI, rather than requiring pybind11 at build time.

```python
import sys
sys.path.insert(0, "bindings/python")  # from the repository root
import numpy as np
from spectralmm import fit

X = np.array([-2., -1., 0., 1., 2., 3.]).reshape(-1, 1)
y = np.array([0., 0., 1., 0., 1., 1.])
model = fit(X, y, family="bernoulli", rank=1, ridge=0.1)
print(model.coef)
print(model.info)
print(model.predict(X))
```

## R

Install the [standard R package](../docs/R_Package.md); installation automatically
builds this same core using Rcpp, RcppEigen and R's configured BLAS. A C++17
toolchain is needed for source installation; `Matrix` supplies sparse inputs.

```r
library(SpectralMM)
X <- matrix(c(-2, -1, 0, 1, 2, 3), ncol=1)
y <- c(0, 0, 1, 0, 1, 1)
model <- spectralmm_fit(X, y, "bernoulli", rank=1, ridge=0.1)
coef(model)
model$info
predict(model, X)
```

Sparse input currently accepts `dgCMatrix`. Installed package users only need
`library(SpectralMM)`; no manual loader is involved.

## Julia

The Julia package builds and caches the shared core automatically. Use
`SpectralMM.fit(X, y; backend=:cpp, family=:...)` for the C++ interface. Existing
`SpectralMM.glm` calls still use the original Julia solver.

```julia
using SpectralMM
X = reshape([-2., -1., 0., 1., 2., 3.], :, 1)
y = [0., 0., 1., 0., 1., 1.]
model = SpectralMM.fit(X, y; backend=:cpp, family=:bernoulli, rank=1, ridge=0.1)
model.coef
model.info
predict(model, X)
```

`SparseMatrixCSC` is supported. Indices are converted from Julia's one-based
layout to the C ABI's zero-based int64 layout.

## Shared contract

- The family names and parameters below apply to Python, R and the Julia C++
  interface (Julia uses symbols and `family_options`). Bernoulli responses must
  be exactly 0 or 1. There are no observation weights, offsets or formula API yet.
- An intercept is added automatically: Python `fit_intercept=True`, R
  `intercept=TRUE`, Julia `intercept=true`. Pass features without a constant column.
  Disable the option for an existing complete design matrix. Prediction takes the
  same feature columns as training. A redundant constant column triggers a warning.
- The added intercept is implicit in C++ matrix products, preserving sparse storage
  and avoiding a full augmented-matrix copy. Zero feature columns fit an intercept-only model.
- Python `model.coef` includes the intercept first, `model.coef_` contains slopes,
  and `model.intercept_` is a scalar. R `coef()` names it `(Intercept)`;
  native Julia `model.coef` includes it first.
- `beta0`: optional full parameter vector, including the intercept first. Otherwise
  the intercept uses the response mean/link; models without one start at zero when feasible.
- `rank=0` means automatic: `p-1` for `p<20`, otherwise 10. Here `p` is the total parameter count, including the intercept. For `p=1`,
  automatic rank is zero and the scalar spectrum is used.
- `ridge=0`, `floor=1e-6`, `gtol=1e-7`, `relgtol=1e-8`, `eta_max=0.5`,
  `correction_tol=0.01`, `resid_tol=0.5`, `maxiter=200`, `inner_maxiter=100`,
  `nesterov=true`. Ridge excludes the automatic intercept by default. Set
  `penalize_intercept=True` (Python; language-appropriate booleans in R/Julia) to
  penalize it. When automatic intercept is disabled, every supplied column is penalized.
- `krylovdim=0`: automatic `min(p, max(3*(rank+1)+20, rank+11))`; otherwise use
  an integer in `[rank+1,p]`. Up to three Lanczos attempts grow this dimension.
- Loss is the **sum**, not mean, of the Julia family objective (without response-only
  likelihood constants), plus the ridge penalty on the selected coefficients.
- `floor` is Julia's `rho`, default `1e-6`, also explicit in the benchmarks.
- Gradient convergence requires `||gradient|| <= gtol` or
  `||gradient|| / (1 + ||initial_gradient||) <= relgtol`.
- Wrappers additionally accept Julia-style negligible and stalled steps by default:
  `||step|| <= 1e-14*(1+||beta||)`, or a line-searched step below
  `sqrt(eps)*(1+||beta||)` while the base relative gradient is at most `1e-4`.
  Set `accept_negligible=False, accept_stalled=False` in Python (logical values
  in R/Julia) to require gradient convergence. Tolerances are exposed as
  `negligible_step_tol`, `step_reltol`, and `stalled_relgtol`.
- Check `info.gradient_converged` separately from `info.converged`; step-based
  acceptance is not evidence that the requested gradient tolerance was reached.
- Always inspect `info.converged`. Termination codes: 0 gradient convergence,
  1 iteration limit, 2 line search stalled, 3 inner solver breakdown,
  4 negligible step, 5 stalled step. `termination_reason` gives a readable name. Invalid
  inputs or non-finite computations raise a language-level exception.
- Diagnostics describe the returned coefficients. `eigresidual` describes the
  last accepted cached spectral approximation; it is NaN if no spectrum was
  needed. `iterations` counts accepted outer steps.

The ABI is documented in `../src/native/include/spectralmm/c_api.h`. It borrows input buffers
only for a synchronous call and writes to caller-owned output buffers. It does
not throw exceptions across the language boundary. Dense input is column-major;
CSC indices are sorted, unique, zero-based int64. Wrappers can allocate to
normalize types/layouts/indices. Sparse inputs are never converted to dense in
the core. Low-rank bases and small projected matrices are dense.

## Validation history

Completed numerical checks are recorded in [VALIDATION.md](VALIDATION.md).
Development test scripts and generated fixtures were removed during cleanup.

## Intentional prototype differences and next steps

This is not a complete port of the Julia package:

- Python/R expose all 14 family types and offer CG, CGLS, CRLS, LSQR
  LSMR, spectral MM and Cholesky through a shared native outer solver; see
  [the benchmark guide](../benchmark/native/README.md). The Julia C++ interface supports the same fourteen families and standard accessors. Formula integration remains in original Julia.
- Correction directly performs Rayleigh–Ritz in the retained subspace. Julia
  first applies a square QR rotation within that same subspace. The resulting
  Ritz subspace is mathematically equivalent, but floating-point paths differ.
- Native Lanczos uses a different RNG and handles early invariant-subspace
  breakdown explicitly. Iteration counts need not match Julia.
- The line search permits a roundoff allowance of `8*eps*max(1,abs(loss))`.
  Wrappers now use the negligible/stalled rules above. The spectral path retries
  failed inner solves with no progress and failed line searches; legacy C entry
  points retain their previous strict behavior. The additive `smm_fit_extended`
  API and Python/R/native-Julia `trace` expose accepted-state histories.
  All wrappers support `verbose` for the common iteration table; see
  [iteration logs](../docs/README.md#iteration-logs).
- Bernoulli uses the stable canonical gradient `X'*(mu-y)+ridge*beta`. Extremely
  saturated predictors can differ from Julia's derivative/variance clipping.
- Spectral PCG and CG/CGLS/CRLS/LSQR/LSMR use per-fit reusable workspaces;
  their inner solves have no Eigen heap allocations in the tested dense/CSC
  cases. Outer gradient, weight, trial, and prediction buffers are reused.
  Spectral restart/correction buffers are retained, and correction reuses
  `H*V` via `H*(V*R)=(H*V)*R`. Small eigensolver and matrix-packing allocations
  remain; the whole fit is not allocation-free. Canonical float64 Python CSC
  inputs are borrowed. No speedup over Julia is claimed.
- For a debug allocation check, configure with
  `-DSPECTRALMM_CHECK_PCG_ALLOCATIONS=ON` and run the fitting workload against that
  library using `SPECTRALMM_LIBRARY`. Despite its historical option name, it
  checks the iterative inner solvers. This Eigen global guard is for serial
  diagnostic runs only; leave it disabled in production.
- Wheels, an installable R package, Julia artifacts and CI across operating
  systems are follow-up work. The current loader paths target a local checkout.

## Implementation notes

Lanczos caches Hessian-vector products (`HQ`) for Ritz residuals and keeps the
full Hessian implicit. Workspaces and unchanged-point gradients are reused.
The CMake option defaults to Eigen-only operation; the build command above and
the current local binary enable BLAS. Profiling remains an opt-in diagnostic.
The retained validation summary is in [VALIDATION.md](VALIDATION.md).

## Usage and benchmark guide

The [LaTeX quick-start guide](../docs/usage_and_benchmark.tex) covers building,
Python/R examples, all model parameters, and reproducible benchmark commands.
See [the final performance report](../benchmark/native/results/family-performance/report.md)
for measured results. Temporary optimization experiments and their snapshots
have been removed; the final source and recorded validation results are retained.

## Python/R model families

Both wrappers accept `family_options`: a Python dict or R named list. Unknown or
inapplicable parameters raise an error. All families support dense/CSC inputs and
`spectral`/`pcg`, `mm`, `cho`/`cholesky`, `cg`, `cgls`, `crls`, `lsqr`, `lsmr`.
See [solver selection](../docs/README.md#choosing-a-solver) for definitions and examples.
The C ABI accepts solver IDs 5 (Cholesky) and 6 (spectral MM) through the existing
`smm_krylov_options` structure; a null pointer still selects spectral PCG.
Rebuild the library before using the new solver IDs.

| Family | Julia model | Parameters (defaults) | Response |
|---|---|---|---|
| `gaussian` | GaussianIdentity | none | finite real |
| `bernoulli` | BernoulliLogit | none | 0 or 1 |
| `probit` | BernoulliProbit | none | 0 or 1 |
| `poisson` | PoissonLog | none | nonnegative |
| `gamma` | GammaLog | none | positive |
| `negative_binomial` | NegativeBinomialLog | `theta=1` | nonnegative |
| `smooth_quantile` | SmoothQuantile | `tau=0.5, smoothing=0.1` | finite real |
| `expectile` | Expectile | `tau=0.5` | finite real |
| `pseudo_huber` | PseudoHuber | `delta=1.345` | finite real |
| `student_t` | StudentT | `nu=4, sigma=1` | finite real |
| `gaussian_log` | GaussianLog | none | positive |
| `gamma_inverse` | GammaInverse | none | positive |
| `tweedie` | TweedieLog | `power=1.5` | nonnegative |
| `binomial` | BinomialLogit | `trials=1` (scalar or length n) | 0 <= y <= trials |

`tau` must be in (0,1); scales/shape/trial counts must be positive and finite;
`power` must be in [1,2]. `smoothing` corresponds to Julia's `epsilon`.
Student-t uses Julia's positive MM surrogate weights, not its exact Hessian.
Probit matches Julia's erf approximation. Inverse-Gamma trial predictors must be
positive; supply a feasible `beta0` when there is no constant column.
Predictions return response means for GLMs and fitted locations/quantiles/expectiles
for residual models. Binomial predictions return counts; pass `trials=` to
`predict` for new observations with different trial counts.

```python
from spectralmm import fit
m = fit(X, y, family="negative_binomial", family_options={"theta": 4.}, solver="spectral")
q = fit(X, y, family="smooth_quantile", family_options={"tau": .25, "smoothing": .1})
values = q.predict(X_new)
```

```r
library(SpectralMM)
m <- spectralmm_fit(X, y, family="student_t",
                    family_options=list(nu=4, sigma=1), solver="cg")
values <- predict(m, X_new)
```

The additive `smm_fit_intercept` C interface accepts intercept flags and model parameters without changing
the layouts or behavior of the original C ABI structures. Rebuild the shared library
before using these updated Python/R wrappers; older binaries lack the new symbol.

## Optional statistical inference

Inference defaults to `auto`: compute standard errors, covariance, coefficient-wise
Wald statistics, p-values and confidence intervals when the total parameter count
(including the intercept) is at most 50. `inference_max_p` changes this cutoff.
Set `inference=True` (R/Julia booleans as appropriate) to bypass the size limit,
or `inference=False` to disable post-fit calculations. This is a project default,
not a universal threshold used by GLM packages.

Use Python `model.summary()`, `model.bse`, `model.pvalues`, `model.cov_params()`;
R `summary(model)`, `vcov(model)`, `confint(model)`; original Julia
`stderror(model)`, `vcov(model)`, `coeftable(model)` (StatsAPI).
All interfaces provide `model.inference` (R: `model$inference`) with status/reason.

GLMs default to expected-information covariance; residual models default to HC1
sandwich covariance with true score derivatives. Ordinary Wald inference is not
reported for ridge-penalized, insufficiently converged, singular or detected
separated fits. The C++ function `smm_infer` is an additive ABI entry point;
rebuild the library before using updated wrappers. Inference forms dense p-by-p
matrices independently of the matrix-free fitting stage.

See [the inference guide](../docs/README.md) for statistical assumptions,
dispersion/t-versus-z conventions, validation commands and references.

## Detailed trace ABI

`smm_fit_logged` is an additive entry point with the same fitting arguments as
`smm_fit_intercept` and a `smm_trace_detail` output buffer of capacity at least
`maxiter+1`. Existing functions and the legacy trace layout are unchanged.
Detailed rows include per-step inner iterations, WLS residual, spectral residual,
step size and spectrum event, plus accepted-state loss/gradient and cumulative
counts. NaN marks inapplicable numeric values; `inner=-1` marks row 0.
Rebuild the native library before using the updated wrappers. Printing is optional
and occurs in the wrapper, so the C++ core does not write to stdout.

## Julia wrapper timing

The native Julia wrapper retains validated shared-library handles by absolute path
for the process lifetime. Repeated fits do not unload and reload the library.
Restart Julia after replacing a library at the same path. A different
`SPECTRALMM_LIBRARY` path selects a separately cached library.
For timing comparisons, warm both implementations, disable printing and trace
capture, choose the same inference setting, initial coefficients and tolerances,
and explicitly align the native `krylovdim` with the original high-level Julia
value `min(p, max(12, rank + 1))`. Here `p` includes the intercept. The native
default Lanczos dimension is different. Compare iteration counts and final
gradients as well as elapsed time.

## Wrapper allocation and initialization

Python retains each `ctypes.CDLL` object and its configured C function signatures
by absolute library path. Julia retains the library handle, fitting/inference
function pointers and immutable default options. Each fit still constructs its
own mutable options and output buffers. Restart the language session after
replacing a native library at the same path. Changing `SPECTRALMM_LIBRARY` to a
different path selects another cached library.

For repeated dense fits, prepare Python inputs with
`X = numpy.asfortranarray(X, dtype=numpy.float64)` and a contiguous float64 response
once outside the timed loop. Compatible arrays are borrowed; row-major, strided
or differently typed inputs are converted when needed. Julia borrows
`Matrix{Float64}` features and `Vector{Float64}` response/initial coefficients;
other input types are converted. Julia CSC values can also be borrowed, while
one-based sparse indices still require zero-based conversion for the C API.
Inputs are read-only during the native call; changing them concurrently with a
fit is unsupported. The wrappers do not retain input arrays as cached datasets.
R already passes double matrices to the bridge without a conversion copy; its
wrapper now avoids redundant `storage.mode` assignments for these matrices.

Python's dense constant-column warning uses the first two observations to
reject nonconstant candidates before scanning a complete column. Every surviving
candidate is fully checked, so the warning's meaning is unchanged. Printing,
trace capture and post-fit inference retain their existing opt-in/automatic
behavior; none were disabled to obtain the wrapper speed improvements.

The shared C++ sources now live in `../src/native/`, so the standard R package
and this standalone CMake build compile a single canonical core. R benchmarks load the installed package with `library(SpectralMM)`; the old
development loader has been removed.
