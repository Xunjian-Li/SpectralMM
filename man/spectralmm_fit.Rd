\name{spectralmm_fit}
\alias{spectralmm_fit}
\title{Fit a Model Using the SpectralMM Native Core}
\description{Fits generalized linear, asymmetric or robust regression models using a shared C++ core, with optional post-fit inference.}
\usage{
spectralmm_fit(X, y, family = .spectralmm_families, beta0 = NULL,
  solver = c("spectral", "cg", "cgls", "crls", "lsqr", "lsmr", "cho", "mm",
             "pcg", "cholesky"),
  preconditioner = c("jacobi", "none"), krylov_rtol = 1e-4,
  krylov_atol = 1e-8, family_options = list(),
  intercept = TRUE, penalize_intercept = FALSE,
  inference = "auto", inference_max_p = 50L, cov_type = "auto",
  level = .95, use_t = NULL, dispersion = NULL,
  verbose = FALSE, trace = FALSE, ...)
}
\arguments{
\item{X}{A real numeric matrix or Matrix dgCMatrix, with feature columns and no added intercept column.}
\item{y}{A real numeric response vector with one value per row.}
\item{family}{Family name: gaussian, bernoulli, probit, poisson, gamma, negative_binomial, gaussian_log, gamma_inverse, tweedie, binomial, smooth_quantile, expectile, pseudo_huber, or student_t. The default is gaussian.}
\item{beta0}{Optional initial coefficients, including the intercept first when fitted.}
\item{solver}{Spectral PCG (spectral or pcg), spectral mm, a Krylov solver, or Cholesky (cho or cholesky). Cholesky forms a full curvature matrix.}
\item{preconditioner}{Jacobi or no preconditioner for nonspectral Krylov solvers.}
\item{krylov_rtol, krylov_atol}{Relative and absolute nonspectral inner tolerances.}
\item{family_options}{Named list of fixed parameters: theta (negative binomial, default 1); power (Tweedie, 1.5); trials (binomial, 1, scalar or vector); tau (quantile/expectile, 0.5); smoothing (smooth quantile, 0.1); delta (pseudo-Huber, 1.345); nu and sigma (Student-t, 4 and 1).}
\item{intercept}{Add an intercept automatically. Default TRUE.}
\item{penalize_intercept}{Include the automatically added intercept in the ridge penalty. Default FALSE.}
\item{inference}{"auto", TRUE or FALSE. Automatic inference is limited to inference_max_p parameters, including the intercept. TRUE bypasses only the size limit.}
\item{inference_max_p}{Automatic inference size cutoff, default 50.}
\item{cov_type}{"auto", "model", or "sandwich", where supported by the family.}
\item{level}{Confidence level between zero and one.}
\item{use_t}{Use a Student-t reference. NULL follows the model dispersion convention.}
\item{dispersion}{Optional positive fixed dispersion; NULL uses the model convention.}
\item{verbose}{Print the iteration table and termination summary.}
\item{trace}{Retain the accepted-iterate table.}
\item{...}{Native options: rank, maxiter, inner_maxiter, krylovdim, ridge, floor, gtol, relgtol, eta_max, correction_tol, resid_tol, nesterov, accept_negligible, accept_stalled, negligible_step_tol, step_reltol, stalled_relgtol. Unknown options are rejected.}
}
\details{
The native library is compiled during package installation. No CMake invocation, manual library loading, or Julia runtime is needed by R users. Dense operations use the BLAS configured for the R installation.

The matrix interface does not accept formulas, case weights or offsets. Bernoulli responses are zero/one; grouped binomial responses are success counts with trials supplied separately. Gamma responses are positive. Family parameters are fixed, not jointly estimated.

Inference requires gradient convergence, an unpenalized fit, more observations than coefficients, and identifiable information. It can be unavailable for separation, singularity or other numerical conditions. A step-based stop alone does not guarantee inference availability. Inspect \code{fit$info$gradient_converged} and \code{fit$inference$status}.
}
\value{An object of class \code{spectralmm_native}, with coefficient vector \code{coef}, diagnostics \code{info}, \code{family}, \code{family_options}, \code{intercept}, \code{n_features}, \code{inference}, and optionally \code{trace}. Inference includes status, covariance, standard errors, reference statistics, p-values, confidence intervals and Wald chi-square results when available.}
\examples{
set.seed(1)
X <- matrix(rnorm(300), 100, 3)
y <- rbinom(100, 1, plogis(0.2 + X \%*\% c(0.4, -0.3, 0.2)))
fit <- spectralmm_fit(X, y, family = "bernoulli")
coef(fit)
summary(fit)
head(predict(fit, X))
}
\seealso{\code{\link{spectralmm_methods}}}
