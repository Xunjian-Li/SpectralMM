\name{spectralmm_family}
\alias{pseudo_huber}
\alias{expectile}
\alias{smooth_quantile}
\alias{student_t}
\title{Non-GLM Loss Specifications}
\description{Lightweight frontend specifications for the existing non-GLM fitting implementation. Use these constructors with spectralmm_fit, not spectralmm_glm.}
\usage{
pseudo_huber(delta = 1)
expectile(q = 0.5)
smooth_quantile(q = 0.5, epsilon = 0.1)
student_t(nu = 4)
}
\arguments{
\item{delta}{Positive finite pseudo-Huber scale.}
\item{q}{Finite asymmetry level strictly between zero and one.}
\item{epsilon}{Positive finite smoothing parameter.}
\item{nu}{Positive finite Student-t degrees of freedom. The residual scale retains its existing default of one.}
}
\value{A lightweight object of class spectralmm_family, containing a core family name and its options.}
\details{Matrix and formula fits normalize these objects to the existing string and family_options representation. The public q maps to the internal tau; epsilon maps to smoothing. Supplying family_options together with an object is an error, even for NULL or an empty list. Existing string specifications remain supported unchanged, including custom Student-t sigma. All interfaces use delta=1 by default, including the legacy string path. To reproduce the earlier development default, explicitly set delta=1.345.}
\examples{
set.seed(1)
dat <- data.frame(y=rnorm(100), x=rnorm(100))
m <- spectralmm_fit(y ~ x, data=dat, family=pseudo_huber(delta=1))
coef(m)
}
\seealso{\code{\link{spectralmm_fit}}}
