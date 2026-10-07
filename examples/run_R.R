# Run from the repository root after examples/run_python.py.
library(SpectralMM)
data <- read.csv('examples/data.csv',check.names=FALSE)
X <- as.matrix(data[c('x1','x2','x3')])
parameters <- list(gaussian=list(),bernoulli=list(),probit=list(),poisson=list(),gamma=list(),
 negative_binomial=list(theta=4),gaussian_log=list(),gamma_inverse=list(),tweedie=list(power=1.5),
 binomial=list(trials=4),smooth_quantile=list(q=.25,smoothing=.1),expectile=list(q=.25),
 pseudo_huber=list(delta=1),student_t=list(nu=4,sigma=.5))
rows <- list()
log <- file('examples/results/R.txt',open='wt');sink(log)
for (family in names(parameters)) {
 cat('\n===',family,': 100 observations, 3 features, 4 coefficients ===\n')
 beta0 <- if(family=='gamma_inverse') c(2,0,0,0) else rep(0,4)
 m <- spectralmm_fit(X,data[[family]],family,family_options=parameters[[family]],beta0=beta0,
  solver='pcg',rank=3,maxiter=500,gtol=1e-6,relgtol=1e-8,accept_stalled=FALSE,verbose=TRUE)
 print(summary(m));cat('First five fitted values:',head(predict(m,X),5),'\n')
 inf <- m$inference;ok <- inf$status=='ok'
 value <- function(key) if(ok) inf[[key]] else rep(NA_real_,4)
 rows[[family]] <- data.frame(family=family,term=names(coef(m)),estimate=unname(coef(m)),
  std_error=value('std_error'),statistic=value('statistic'),p_value=value('p_value'),
  lower=if(ok) inf$conf_int[,1] else NA_real_,upper=if(ok) inf$conf_int[,2] else NA_real_,
  wald_chisq=value('wald_chisq'),wald_p_value=value('wald_p_value'),loss=m$info$loss,
  relgradnorm=m$info$relgradnorm,iterations=m$info$iterations,converged=m$info$gradient_converged,
  inference_status=inf$status,row.names=NULL)
}
sink();close(log)
write.csv(do.call(rbind,rows),'examples/results/R.csv',row.names=FALSE,na='')
cat('R completed all 14 small-data examples.\n')
