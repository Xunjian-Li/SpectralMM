# Numerical correctness audit (2026-10-01)

Publication-directory work and performance optimization were paused for this
audit. The fixes concern objective derivatives, rounding-scale line-search
comparisons and final-step convergence reporting. Gradient thresholds and
inference validity requirements were not relaxed.

## Findings and fixes

1. **Probit objective/gradient mismatch.** The objective used a polynomial erf
   approximation, while its gradient used the normal density, which is not the
   exact derivative of that approximation. On the saved 100-by-3 data, a
   256-bit central-difference check gave maximum gradient discrepancy
   7.53554e-6. With a consistent normal log-CDF and score, the discrepancy is
   1.77636e-15. Julia uses the distribution library's log-CDF; C++ uses erfc and
   a normal-tail expansion where direct evaluation would underflow. Stable
   score, working response, Fisher information and sandwich curvature are
   computed consistently. R, Python and Julia prediction functions also use
   their accurate normal CDFs. Misclassified extreme tails no longer acquire a
   clipped constant objective and an incompatible derivative.
2. **Rounding-scale line-search rejection.** Julia required exact floating-point
   objective decrease, while C++ already allowed a rounding-scale increase.
   Near stationarity, a useful step could be rejected and reduced to a tiny
   step, leaving a gradient above the inference threshold. Julia spectral, CHO
   and Krylov line searches now allow 8 * eps(T) * max(1, abs(reference loss));
   the CHO/Krylov Armijo requirement is retained within that allowance.
3. **Final allowed step.** Julia could return `converged=false` when the last
   accepted step met the gradient threshold but no next iteration was allowed.
   All Julia solver result paths now recognize a gradient-converged returned
   endpoint, matching the native final check.

Step-based stopping remains a separate allowed termination condition. A
successful tiny/stalled-step stop does not by itself authorize Wald inference;
the gradient and inference checks still apply. Backend iteration histories are
not required to be identical.

## Reproduced cases

All rows use rank 3, zero beta0, maxiter 500, gtol 1e-6, relgtol 1e-8,
accept_stalled=false, and the existing examples/data.csv. Family parameters are
negative-binomial theta=4 and Student-t nu=4, sigma=0.5. These are accuracy
checks, not new timing benchmarks.

| Family / solver / storage | Iterations before → after | Relative gradient before → after | Inference before → after |
|---|---:|---:|---|
| probit / pcg / dense | 500 → 6 | 1.506e-07 → 5.845e-09 | unavailable → ok |
| probit / pcg / CSC | 500 → 6 | 1.335e-07 → 5.845e-09 | unavailable → ok |
| probit / cho / dense | 7 → 6 | 1.514e-07 → 5.838e-09 | unavailable → ok |
| probit / cho / CSC | 9 → 6 | 1.329e-07 → 5.838e-09 | unavailable → ok |
| negative_binomial / cho / dense | 9 → 8 | 3.047e-08 → 1.900e-08 | ok → ok |
| negative_binomial / cho / CSC | 9 → 8 | 5.652e-08 → 4.690e-09 | unavailable → ok |
| student_t / cho / dense | 13 → 13 | 6.179e-09 → 6.179e-09 | ok → ok |
| student_t / cho / CSC | 13 → 13 | 2.069e-08 → 6.179e-09 | unavailable → ok |

## Validation

- 448 Julia/C++ fits: 14 families, 8 solvers, dense and CSC matrices, both
  backends. Every case passed the comparison; all previous eighteen unavailable
  Julia-inference configurations now report valid inference at the unchanged
  thresholds. Maximum backend coefficient difference: 4.41181e-8.
- 322 retained Julia regression assertions: backend contract, 256-bit objective
  derivative checks for all fourteen families, native initial-loss/gradient
  comparisons, Probit tails from -100 to 100, final-step convergence, accurate
  independent GLM.jl Probit coefficients/standard errors and inference guards.
  GLM.jl's own deviance stopping tolerances are tightened for the reference
  calculation; they are not equivalent to SpectralMM's gradient tolerances.
- 224 Python and 224 freshly installed R fits against saved reference results;
  maximum coefficient difference 7.00820e-8. Both cover fourteen families,
  eight solvers, dense/CSC inputs and standard errors. The small reference
  differences include the intentional correction of the Probit objective.
- Thirty native Probit tail objective/gradient checks against SciPy's
  log_ndtr, with maximum absolute loss discrepancy 1.42109e-14.
- R Probit coefficients and standard errors independently agree with stats::glm
  using tight deviance tolerances, with inference explicitly required available.

Run the retained regression suite with `julia --project=. test/runtests.jl`.
It requires examples/data.csv as the small regression fixture. C++ is rebuilt
on first use when its source-keyed Julia cache changes. Validation was performed
on macOS arm64; this is not proof for every dataset or platform. Earlier timing
reports and example-output files remain historical snapshots. Release artifacts
under output/ predate these fixes and must be regenerated before distribution.
