\name{spectralmm_fit}
\alias{spectralmm_fit}
\alias{spectralmm_glm}
\alias{spectralmm_control}
\alias{spectralmm_diagnostics}
\title{Fit Statistical Models with SpectralMM}
\description{Matrix and formula interfaces sharing the same numerical core. Use the GLM interface for conventional family/link specifications and the generic interface for residual losses or legacy family identifiers.}
\usage{
spectralmm_glm(X, y = NULL, data = NULL, family = stats::gaussian(),
  link = NULL, start = NULL, rank = NULL, solver = "pcg",
  control = spectralmm_control(), ...)
spectralmm_fit(X, y = NULL, family = "gaussian", data = NULL,
  start = NULL, beta0 = NULL, intercept = NULL, rank = NULL,
  solver = "pcg", control = spectralmm_control(),
  weights = NULL, offset = NULL, ...)
spectralmm_control(...)
spectralmm_diagnostics(object)
}
\arguments{
\item{X}{A numeric feature matrix, a sparse dgCMatrix, or a formula.}
\item{y}{Numeric response vector for matrix input.}
\item{data}{A data frame for formula input. Missing values are rejected.}
\item{family}{For GLMs, an R family object (gaussian, binomial, poisson or Gamma), or a supported family name. Negative binomial with fixed theta uses the name \code{"negative_binomial"}. For generic fitting, a constructor such as \code{pseudo_huber(delta=1)}, or a legacy core family name or residual loss: smooth_quantile, expectile, pseudo_huber or student_t.}
\item{link}{Link name for a named family; specify it within an R family object instead when using one. Supported pairs are Gaussian identity/log, binomial logit/probit, Poisson log, Gamma log/inverse and negative binomial log.}
\item{start,beta0}{Initial coefficients including an intercept when present. beta0 is a compatibility alias; do not supply both.}
\item{intercept}{NULL adds an intercept for matrix input. FALSE means a complete design matrix. Formula input controls its own intercept and rejects this argument.}
\item{rank}{Retained spectral rank; NULL retains the existing numerical default.}
\item{solver}{pcg, mm, cg, cho, cgls, crls, lsqr or lsmr. Legacy aliases spectral and cholesky remain accepted.}
\item{control}{Advanced numerical settings from spectralmm_control. Omitted settings retain existing defaults. Duplicate direct and control options are rejected.}
\item{weights,offset}{Not implemented by the numerical core. Non-NULL values are rejected, including formula offsets.}
\item{object}{A fitted SpectralMM model.}
\item{...}{Numerical settings (maxiter, inner_maxiter, krylovdim, ridge, floor, gtol, relgtol, eta_max, correction_tol, resid_tol, nesterov, preconditioner, krylov_rtol, krylov_atol, accept_negligible, accept_stalled, negligible_step_tol, step_reltol, stalled_relgtol); or fit arguments family_options, inference, inference_max_p, cov_type, level, use_t, dispersion, verbose, trace, penalize_intercept. Control accepts only numerical settings.}
}
\details{
The GLM binomial interface currently accepts binary zero/one responses, not proportions or two-column success/failure responses. Generic binomial fitting retains success counts with family_options=list(trials=...). Probit is a binomial link; the old probit core identifier remains a generic compatibility alias. Fixed negative binomial shape uses family_options=list(theta=...).

Formula fits preserve terms, factor levels, contrasts and coefficient names for new-data prediction. Both input routes call the same solver. Matrix input defaults to automatically adding one intercept. Starting vectors follow the resulting coefficient order.

Inference defaults to automatic only for at most 50 coefficients including the intercept. It also requires an unpenalized gradient-converged identifiable model with n greater than p. Setting inference=TRUE bypasses only the size cutoff. No covariance matrix is allocated merely for summary or diagnostics.

Deviance is unscaled. Gaussian log likelihood uses the maximum-likelihood scale RSS/n unless dispersion is supplied. Gamma log likelihood is evaluated at the supplied dispersion or Pearson dispersion with n-p denominator; this is not a profiled Gamma likelihood. The covariance dispersion policy is unchanged. Shape parameters remain fixed. Normalized likelihood and deviance are not offered for residual losses or Tweedie in this interface.
}
\value{A spectralmm object retaining the spectralmm_native compatibility class. It contains coefficients, family/link, formula metadata, cached linear predictors, fitted values, response residuals, inference and separate numerical diagnostics. The control constructor returns a validated options list; the diagnostics accessor returns a list.}
\examples{
set.seed(1)
dat <- data.frame(x1=rnorm(100), x2=rnorm(100), y=rbinom(100,1,.5))
m <- spectralmm_glm(as.matrix(dat[c("x1","x2")]), dat$y,
                    family=binomial("logit"))
f <- spectralmm_glm(y ~ x1 + x2, data=dat, family=binomial("logit"))
coef(f)
summary(f)
head(predict(f, dat, type="response"))
spectralmm_diagnostics(f)$termination_reason
}
\seealso{\code{\link{spectralmm_methods}}}
