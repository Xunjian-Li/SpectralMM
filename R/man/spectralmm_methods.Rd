\name{spectralmm_methods}
\alias{spectralmm_methods}
\alias{coef.spectralmm_native}
\alias{predict.spectralmm_native}
\alias{vcov.spectralmm_native}
\alias{confint.spectralmm_native}
\alias{summary.spectralmm_native}
\alias{print.spectralmm_native}
\alias{print.summary.spectralmm_native}
\alias{predict.spectralmm}
\alias{fitted.spectralmm}
\alias{residuals.spectralmm}
\alias{nobs.spectralmm}
\alias{deviance.spectralmm}
\alias{logLik.spectralmm}
\alias{summary.spectralmm}
\alias{print.spectralmm}
\alias{print.summary.spectralmm}
\title{Extract Results from a SpectralMM Fit}
\description{Coefficient, prediction, covariance, confidence interval and summary methods for fitted SpectralMM models.}
\usage{
\method{predict}{spectralmm}(object, newdata = NULL, type = NULL,
  trials = NULL, offset = NULL, ...)
\method{fitted}{spectralmm}(object, ...)
\method{residuals}{spectralmm}(object, type = "response", ...)
\method{nobs}{spectralmm}(object, ...)
\method{deviance}{spectralmm}(object, ...)
\method{logLik}{spectralmm}(object, ...)
\method{summary}{spectralmm}(object, ...)
\method{print}{spectralmm}(x, ...)
\method{print}{summary.spectralmm}(x, digits = 6, max_rows = 20L, ...)

\method{coef}{spectralmm_native}(object, ...)
\method{predict}{spectralmm_native}(object, newdata, trials = NULL, ...)
\method{vcov}{spectralmm_native}(object, ...)
\method{confint}{spectralmm_native}(object, parm, level = .95, ...)
\method{summary}{spectralmm_native}(object, ...)
\method{print}{spectralmm_native}(x, ...)
\method{print}{summary.spectralmm_native}(x, digits = 5, max_rows = 20L, ...)
}
\arguments{
\item{object}{A fitted model.}
\item{x}{A fitted model or summary, as appropriate.}
\item{type}{Prediction: link or response; NULL defaults to link for spectralmm_glm and response for legacy spectralmm_fit. Residuals: only response is supported.}
\item{newdata}{NULL for training predictions, a data frame for formula fits, or a numeric or sparse matrix with the same feature columns and ordering as training data, without an automatically fitted intercept column.}
\item{trials}{Grouped binomial prediction trial counts, a positive scalar or vector. Defaults to the fitting value. Supply new counts for new rows when needed.}
\item{offset}{Not implemented; a non-NULL value raises an error.}
\item{parm}{Optional coefficient indices or names.}
\item{level}{Confidence level between zero and one.}
\item{digits}{Number of significant digits to print.}
\item{max_rows}{Maximum coefficient rows to print. Full results remain in the summary.}
\item{...}{Further arguments; the fitted-model print method forwards them to the summary print method.}
}
\value{Coefficients (intercept first), response predictions, a covariance matrix, a confidence-interval matrix, or a summary list, depending on the method. Binomial predictions are expected counts. Non-GLM predictions are fitted locations: conditional expectiles for expectile loss, smoothed quantiles for smoothed quantile loss, and robust locations for Pseudo-Huber and Student-t. These are point predictions, not prediction intervals. Print methods return their input invisibly. Covariance and confidence interval methods signal an error when inference is unavailable.}
\details{Coefficient tables use a t or z reference. Separate \code{wald_chisq} and \code{wald_p_value} components use an asymptotic chi-square reference; their p-values can differ from t-reference p-values.}
\seealso{\code{\link{spectralmm_fit}}}
