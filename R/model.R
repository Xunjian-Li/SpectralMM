# Statistical API. All fits delegate to the existing numerical binding.
spectralmm_control <- function(...) {
  options <- list(...)
  allowed <- c("maxiter","inner_maxiter","krylovdim","ridge","floor","gtol","relgtol",
    "eta_max","correction_tol","resid_tol","nesterov","preconditioner","krylov_rtol",
    "krylov_atol","accept_negligible","accept_stalled","negligible_step_tol","step_reltol","stalled_relgtol")
  if (length(options) && (is.null(names(options)) || any(!nzchar(names(options))) ||
      anyDuplicated(names(options)) || any(!names(options) %in% allowed))) stop("unknown or duplicate control option")
  structure(options,class="spectralmm_control")
}

.smm_spec <- function(family, link=NULL) {
  if (inherits(family,"family")) {
    if (!is.null(link)) stop("specify the link in the R family object")
    link <- family$link; family <- tolower(family$family)
  }
  if (!is.character(family) || length(family)!=1L) stop("family must be a family object or a supported name")
  aliases <- list(bernoulli=c("binomial","logit"),probit=c("binomial","probit"),
                  gaussian_log=c("gaussian","log"),gamma_inverse=c("gamma","inverse"))
  if (family %in% names(aliases)) {
    a <- aliases[[family]]
    if (!is.null(link) && link!=a[2L]) stop("legacy family alias conflicts with link")
    family<-a[1L]; link<-a[2L]
  }
  defaults<-c(gaussian="identity",binomial="logit",poisson="log",gamma="log",negative_binomial="log")
  if (is.null(link)) link<-unname(defaults[family])
  map<-c("gaussian/identity"="gaussian","gaussian/log"="gaussian_log",
    "binomial/logit"="bernoulli","binomial/probit"="probit","poisson/log"="poisson",
    "gamma/log"="gamma","gamma/inverse"="gamma_inverse","negative_binomial/log"="negative_binomial")
  internal<-unname(map[paste(family,link,sep="/")])
  if (is.na(internal)) stop("unsupported GLM family/link; use spectralmm_fit for residual losses")
  obj<-switch(family,gaussian=stats::gaussian(link),binomial=stats::binomial(link),
    poisson=stats::poisson(link),gamma=stats::Gamma(link),list(family=family,link=link))
  list(family=family,link=link,internal=internal,object=obj)
}

.smm_fit_statistics <- function(core,y,mu,eta,p,options,dispersion) {
  dev<-ll<-NA_real_
  if (core %in% c("gaussian","gaussian_log")) {
    dev<-sum((y-mu)^2); scale<-if(is.null(dispersion)) dev/length(y) else dispersion
    ll<-if(scale==0) Inf else sum(stats::dnorm(y,mu,sqrt(scale),log=TRUE))
  } else if (core %in% c("bernoulli","probit","binomial")) {
    n<-if(is.null(options$trials)) 1 else options$trials
    lp<-if(core=="probit") stats::pnorm(eta,log.p=TRUE) else -pmax(-eta,0)-log1p(exp(-abs(eta)))
    lq<-if(core=="probit") stats::pnorm(-eta,log.p=TRUE) else -pmax(eta,0)-log1p(exp(-abs(eta)))
    xl<-function(a,b) ifelse(a==0,0,a*log(b))
    kernel<-sum(y*lp+(n-y)*lq)
    dev<-2*(sum(xl(y,y/n)+xl(n-y,(n-y)/n))-kernel)
    ll<-kernel+sum(lgamma(n+1)-lgamma(y+1)-lgamma(n-y+1))
  } else if (core=="poisson") {
    dev<-2*sum(ifelse(y==0,0,y*log(y/mu))-y+mu); ll<-sum(stats::dpois(y,mu,log=TRUE))
  } else if (core %in% c("gamma","gamma_inverse")) {
    dev<-2*sum((y-mu)/mu-log(y/mu))
    scale<-if(is.null(dispersion)) if(length(y)>p) sum(((y-mu)/mu)^2)/(length(y)-p) else NA_real_ else dispersion
    if (is.finite(scale) && scale>0) ll<-sum(stats::dgamma(y,shape=1/scale,scale=mu*scale,log=TRUE))
  } else if (core=="negative_binomial") {
    theta<-if(is.null(options$theta)) 1 else options$theta
    dev<-2*sum(ifelse(y==0,0,y*log(y/mu))-(y+theta)*log((y+theta)/(mu+theta)))
    ll<-sum(stats::dnbinom(y,size=theta,mu=mu,log=TRUE))
  }
  if(core %in% c("bernoulli","probit","binomial","poisson","negative_binomial") && any(y!=floor(y))) ll<-NA_real_
  if(core=="binomial" && any(n!=floor(n))) ll<-NA_real_
  list(deviance=dev,loglikelihood=ll)
}

spectralmm_glm <- function(X,y=NULL,data=NULL,family=stats::gaussian(),link=NULL,
                           start=NULL,rank=NULL,solver="pcg",control=spectralmm_control(),...) {
  spec<-.smm_spec(family,link)
  dots<-list(...)
  if (spec$family=="binomial" && !is.null(dots$family_options$trials))
    stop("spectralmm_glm currently supports binary responses only; grouped counts remain available through spectralmm_fit")
  fit<-spectralmm_fit(X,y,data=data,family=spec$internal,start=start,rank=rank,solver=solver,control=control,...)
  fit$family<-spec$object; fit$link<-spec$link
  fit$prediction_default<-"link"
  fit$call<-match.call()
  fit
}

spectralmm_fit <- function(X,y=NULL,family="gaussian",data=NULL,start=NULL,beta0=NULL,
                          intercept=NULL,rank=NULL,solver="pcg",control=spectralmm_control(),
                          weights=NULL,offset=NULL,...) {
  if (!is.null(weights) || !is.null(offset)) stop("observation weights and offset are not implemented by the numerical core")
  if (!is.null(start) && !is.null(beta0)) stop("use start or beta0, not both")
  if (!inherits(control,"spectralmm_control")) stop("control must be created by spectralmm_control")
  dots<-list(...)
  if (anyDuplicated(names(dots)) || any(names(dots) %in% names(control))) stop("duplicate direct/control options")
  opts<-c(unclass(control),dots)
  terms<-xlevels<-contrasts<-formula<-NULL
  if (inherits(X,"formula")) {
    if (!is.null(y) || is.null(data)) stop("formula input requires data= and no separate y")
    if (!is.null(intercept)) stop("control the intercept in the formula")
    formula<-X
    mf<-stats::model.frame(X,data=data,na.action=stats::na.fail)
    terms<-attr(mf,"terms")
    if (length(attr(terms,"offset"))) stop("formula offsets are not implemented by the numerical core")
    y<-stats::model.response(mf)
    if (!is.numeric(y) || !is.null(dim(y))) stop("formula response must be one numeric column")
    X<-stats::model.matrix(terms,mf)
    contrasts<-attr(X,"contrasts")
    xlevels<-lapply(mf,function(v) if(is.factor(v)) levels(v) else if(is.character(v)) levels(factor(v)) else NULL)
    xlevels<-xlevels[!vapply(xlevels,is.null,logical(1))]
    intercept<-isTRUE(attr(terms,"intercept")==1L)
    if(intercept) X<-X[,colnames(X)!="(Intercept)",drop=FALSE]
  } else {
    if (!is.null(data)) stop("data= is only used with a formula")
    if (is.null(intercept)) intercept<-TRUE
  }
  if(!is.null(rank)) opts$rank<-rank
  raw<-do.call(.spectralmm_fit_core,c(list(X=X,y=y,family=family,
    beta0=if(is.null(start)) beta0 else start,intercept=intercept,solver=solver),opts))
  eta<-as.vector(X %*% if(intercept) raw$coef[-1L] else raw$coef)+if(intercept) raw$coef[1L] else 0
  mu<-predict.spectralmm_native(raw,X)
  raw$core_family<-family
  raw$family<-switch(family,probit=stats::binomial("probit"),bernoulli=stats::binomial("logit"),
    gaussian=stats::gaussian(),gaussian_log=stats::gaussian("log"),poisson=stats::poisson(),
    gamma=stats::Gamma("log"),gamma_inverse=stats::Gamma("inverse"),list(family=family,link=NULL))
  if(family %in% c("negative_binomial","binomial")) raw$family$link<-if(family=="binomial") "logit" else "log"
  raw$link<-raw$family$link
  raw$linear.predictors<-eta; raw$fitted.values<-mu; raw$residuals<-y-mu
  raw$nobs<-length(y); raw$nparams<-length(raw$coef); raw$objective<-raw$info$loss
  raw$terms<-terms; raw$formula<-formula; raw$xlevels<-xlevels; raw$contrasts<-contrasts
  stat<-.smm_fit_statistics(family,y,mu,eta,raw$nparams,raw$family_options,opts$dispersion)
  raw$deviance<-stat$deviance; raw$loglikelihood<-stat$loglikelihood
  default<-list(maxiter=200L,inner_maxiter=100L,krylovdim=0L,ridge=0,floor=1e-6,
    gtol=1e-7,relgtol=1e-8,eta_max=.5,correction_tol=.01,resid_tol=.5,preconditioner="jacobi",
    krylov_rtol=1e-4,krylov_atol=1e-8)
  settings<-utils::modifyList(default,opts)
  r<-if(is.null(rank) || rank==0) if(raw$nparams<20) raw$nparams-1L else 10L else rank
  raw$diagnostics<-c(raw$info,list(solver=if(solver=="spectral") "pcg" else solver,rank=r,
    outer_iterations=raw$info$iterations,gradient_norm=raw$info$gradnorm,objective=raw$objective,
    krylov_dimension=if(settings$krylovdim==0) min(raw$nparams,max(3*(r+1)+20,r+11)) else settings$krylovdim,preconditioner=settings$preconditioner,control=settings,backend="cpp"))
  raw$diagnostics$termination_reason<-c("gradient tolerance reached","maximum iterations reached",
    "line search failed","inner breakdown","negligible step","stalled near tolerance")[raw$info$termination+1L]
  raw$estimated_scale<-family %in% c("gaussian","gaussian_log","gamma","gamma_inverse") && is.null(opts$dispersion)
  raw$call<-match.call()
  class(raw)<-c("spectralmm","spectralmm_native")
  raw
}

spectralmm_diagnostics <- function(object) {
  if(!inherits(object,"spectralmm")) stop("object must be a SpectralMM fitted result")
  object$diagnostics
}
fitted.spectralmm <- function(object,...) object$fitted.values
residuals.spectralmm <- function(object,type="response",...) {
  if(type!="response") stop("only response residuals are implemented")
  object$residuals
}
nobs.spectralmm <- function(object,...) object$nobs
deviance.spectralmm <- function(object,...) {
  if(is.na(object$deviance)) stop("deviance is not implemented for this loss")
  object$deviance
}
logLik.spectralmm <- function(object,...) {
  if(is.na(object$loglikelihood)) stop("normalized likelihood is not available for this model/scale")
  structure(object$loglikelihood,class="logLik",df=object$nparams+as.integer(object$estimated_scale),nobs=object$nobs)
}
predict.spectralmm <- function(object,newdata=NULL,type=NULL,trials=NULL,...) {
  if(is.null(type)) type<-if(is.null(object$prediction_default)) "response" else object$prediction_default
  type<-match.arg(type,c("link","response"))
  if(is.null(newdata)) return(if(type=="link") object$linear.predictors else object$fitted.values)
  if(!is.null(object$terms)) {
    tt<-stats::delete.response(object$terms)
    mf<-stats::model.frame(tt,newdata,na.action=stats::na.fail,xlev=object$xlevels)
    newdata<-stats::model.matrix(tt,mf,contrasts.arg=object$contrasts)
    if(object$intercept) newdata<-newdata[,colnames(newdata)!="(Intercept)",drop=FALSE]
  }
  if(is.null(dim(newdata)) || ncol(newdata)!=object$n_features) stop("prediction feature count differs from training")
  if(type=="link") return(as.vector(newdata %*% if(object$intercept) object$coef[-1L] else object$coef)+if(object$intercept) object$coef[1L] else 0)
  object$family<-object$core_family
  predict.spectralmm_native(object,newdata,trials=trials)
}
summary.spectralmm <- function(object,...) {
  inf<-object$inference
  tab<-matrix(coef(object),ncol=1L,dimnames=list(names(coef(object)),"Estimate"))
  if(inf$status=="ok") {
    tab<-cbind(tab,`Std. Error`=inf$std_error,statistic=inf$statistic,p=inf$p_value)
    colnames(tab)[3:4]<-c(paste(inf$statistic_type,"value"),paste0("Pr(>|",inf$statistic_type,"|)"))
  }
  structure(list(family=object$family,observations=object$nobs,parameters=object$nparams,
    coefficients=tab,inference=inf,deviance=object$deviance,loglikelihood=object$loglikelihood,
    diagnostics=object$diagnostics),class="summary.spectralmm")
}
print.summary.spectralmm <- function(x,digits=5,max_rows=20L,...) {
  cat(if(is.null(x$family$link)) "SpectralMM Regression Model\n" else "SpectralMM Generalized Linear Model\n")
  cat("Family:",x$family$family," Link:",if(is.null(x$family$link)) "not applicable" else x$family$link,"\n")
  cat("Observations:",x$observations," Parameters:",x$parameters,"\nCoefficients:\n")
  print(x$coefficients[seq_len(min(nrow(x$coefficients),max_rows)),,drop=FALSE],digits=digits)
  if(nrow(x$coefficients)>max_rows) cat("Additional coefficients are available in coef(model).\n")
  if(x$inference$status!="ok") cat("Inference",x$inference$status,":",x$inference$reason,"\n")
  if(!is.na(x$deviance)) cat("Deviance:",x$deviance," Log-Likelihood:",x$loglikelihood,"\n")
  d<-x$diagnostics
  cat("SpectralMM optimization:\n  Solver:",toupper(d$solver)," Rank:",d$rank,
      " Converged:",d$converged," Iterations:",d$outer_iterations,"\n")
  invisible(x)
}
print.spectralmm <- function(x,...) { print(summary(x),...); invisible(x) }
