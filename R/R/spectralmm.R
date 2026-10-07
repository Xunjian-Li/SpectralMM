.spectralmm_families <- c("gaussian", "bernoulli", "probit", "poisson", "gamma", "negative_binomial",
  "smooth_quantile", "expectile", "pseudo_huber", "student_t", "gaussian_log", "gamma_inverse", "tweedie", "binomial")
.spectralmm_fit_core <- function(X, y, family = .spectralmm_families, beta0 = NULL,
                           solver = c("spectral", "cg", "cgls", "crls", "lsqr", "lsmr", "cho", "mm", "pcg", "cholesky"),
                           preconditioner = c("jacobi", "none"), krylov_rtol = 1e-4,
                           krylov_atol = 1e-8, family_options = list(),
                           intercept = TRUE, penalize_intercept = FALSE,
                           inference = "auto", inference_max_p = 50L, cov_type = "auto",
                           level = .95, use_t = NULL, dispersion = NULL, verbose = FALSE, trace = FALSE, ...) {
  if (!is.logical(verbose) || length(verbose)!=1L || is.na(verbose) ||
      !is.logical(trace) || length(trace)!=1L || is.na(trace)) stop("verbose and trace must be TRUE or FALSE")
  if (!(identical(inference,"auto") || identical(inference,TRUE) || identical(inference,FALSE)))
    stop("inference must be 'auto', TRUE or FALSE")
  if (!is.numeric(inference_max_p) || length(inference_max_p)!=1L || !is.finite(inference_max_p) ||
      inference_max_p<1 || inference_max_p> .Machine$integer.max || inference_max_p!=floor(inference_max_p))
    stop("inference_max_p must be a positive integer")
  cov_type <- match.arg(cov_type,c("auto","model","sandwich"))
  if (!is.numeric(level) || length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
    stop("level must be in (0,1)")
  if (!is.null(use_t) && (!is.logical(use_t) || length(use_t)!=1L || is.na(use_t)))
    stop("use_t must be NULL, TRUE or FALSE")
  if (!is.null(dispersion) && (!is.numeric(dispersion) || length(dispersion)!=1L ||
      !is.finite(dispersion) || dispersion<=0)) stop("dispersion must be positive and finite")
  if (!is.logical(intercept) || length(intercept)!=1L || is.na(intercept) ||
      !is.logical(penalize_intercept) || length(penalize_intercept)!=1L || is.na(penalize_intercept))
    stop("intercept flags must be TRUE or FALSE")
  family <- match.arg(family, .spectralmm_families)
  if (!is.list(family_options)) stop("family_options must be a named list")
  if (length(family_options) && (is.null(names(family_options)) ||
      anyNA(names(family_options)) || any(!nzchar(names(family_options))) || anyDuplicated(names(family_options))))
    stop("family_options must have unique parameter names")
  for (key in names(family_options)) {
    value <- family_options[[key]]
    if (!is.numeric(value) || is.complex(value) || !is.null(dim(value)) ||
        !length(value) || (key != "trials" && length(value) != 1L))
      stop("family parameters must be real scalars (trials may be a vector)")
  }
  solver <- match.arg(solver)
  if (solver == "pcg") solver <- "spectral"
  if (solver == "cholesky") solver <- "cho"
  preconditioner <- match.arg(preconditioner)
  if (is.matrix(X)) {
    if (!is.numeric(X) || is.complex(X)) stop("X must be real numeric")
    if (typeof(X) != "double") storage.mode(X) <- "double"
  } else if (!inherits(X, "dgCMatrix")) stop("X must be a numeric matrix or dgCMatrix")
  if (!is.numeric(y) || is.complex(y) || !is.null(dim(y))) stop("y must be a real vector")
  if (!is.null(beta0) && (!is.numeric(beta0) || is.complex(beta0) || !is.null(dim(beta0))))
    stop("beta0 must be a real vector")
  if (intercept && nrow(X)>0L) {
    constant <- if (inherits(X, "dgCMatrix")) any(vapply(seq_len(ncol(X)), function(j) {
      a <- X@p[j]+1L; b <- X@p[j+1L]
      b-a+1L == nrow(X) && X@x[a] != 0 && all(X@x[a:b] == X@x[a])
    }, logical(1))) else any(vapply(seq_len(ncol(X)), function(j)
      isTRUE(X[1L,j] != 0 && all(X[,j] == X[1L,j])), logical(1)))
    if (isTRUE(constant)) warning("X contains a nonzero constant column; remove it or use intercept=FALSE to avoid a redundant intercept")
  }
  if ("q" %in% names(family_options)) {
    if ("tau" %in% names(family_options)) stop("supply only q, not both q and tau")
    family_options$tau<-family_options$q; family_options$q<-NULL
  }
  result <- smm_fit_bridge(X, as.double(y), match(family, .spectralmm_families)-1L,
                    list(...), family_options, if (is.null(beta0)) NULL else as.double(beta0),
                    match(solver,c("spectral","cg","cgls","crls","lsqr","lsmr","cho","mm"))-2L,
                    preconditioner=="jacobi", krylov_rtol, krylov_atol, intercept, penalize_intercept,
                    if (identical(inference,FALSE)) 0L else if (identical(inference,TRUE)) 2L else 1L,
                    as.integer(inference_max_p), match(cov_type,c("auto","model","sandwich"))-1L,
                    if (is.null(dispersion)) NaN else dispersion, verbose || trace)
  if (verbose) {
    options <- list(...)
    p <- ncol(X)+as.integer(intercept)
    rank <- options$rank
    if (is.null(rank) || rank==0) rank <- if (p<20) p-1 else 10
    .spectralmm_print_trace(result$trace,result$info,solver,rank,family,family_options,nrow(X),p,intercept,
      if(is.null(options$gtol)) 1e-7 else options$gtol,if(is.null(options$relgtol)) 1e-8 else options$relgtol)
  }
  if (!trace) result$trace <- NULL
  result$intercept <- intercept
  result$n_features <- ncol(X)
  labels <- colnames(X)
  if (is.null(labels)) labels <- if (ncol(X)) paste0("x", seq_len(ncol(X))) else character()
  names(result$coef) <- c(if (intercept) "(Intercept)", labels)
  result$family <- family
  if("tau" %in% names(family_options)) { family_options$q<-family_options$tau; family_options$tau<-NULL }
  result$family_options <- family_options
  if (result$inference$status=="ok") {
    inf <- result$inference
    dimnames(inf$covariance) <- list(names(result$coef),names(result$coef))
    inf$std_error <- sqrt(diag(inf$covariance))
    inf$statistic <- result$coef/inf$std_error
    student <- if (is.null(use_t)) inf$estimated_dispersion else use_t
    inf$statistic_type <- if (student) "t" else "z"
    inf$p_value <- if (student) 2*pt(-abs(inf$statistic),df=inf$df_resid) else 2*pnorm(-abs(inf$statistic))
    inf$wald_chisq <- inf$statistic^2
    inf$wald_p_value <- pchisq(inf$wald_chisq,df=1,lower.tail=FALSE)
    critical <- if (student) qt((1-level)/2,df=inf$df_resid,lower.tail=FALSE) else qnorm((1-level)/2,lower.tail=FALSE)
    inf$conf_int <- cbind(lower=result$coef-critical*inf$std_error,upper=result$coef+critical*inf$std_error)
    inf$level <- level
    result$inference <- inf
  } else if (identical(inference,TRUE)) warning(paste("Inference unavailable:",result$inference$reason))
  class(result) <- "spectralmm_native"
  result
}
coef.spectralmm_native <- function(object, ...) object$coef
predict.spectralmm_native <- function(object, newdata, trials = NULL, ...) {
  if (is.null(dim(newdata)) || length(dim(newdata)) != 2L || ncol(newdata)!=object$n_features)
    stop("newdata must have the same feature columns as training X")
  beta <- if (object$intercept) object$coef[-1L] else object$coef
  eta <- as.vector(newdata %*% beta) + if (object$intercept) object$coef[1L] else 0
  if (object$family %in% c("gaussian", "smooth_quantile", "expectile", "pseudo_huber", "student_t")) return(eta)
  if (object$family == "probit") {
    return(stats::pnorm(eta))
  }
  if (object$family == "gamma_inverse") return(1/pmax(eta,sqrt(.Machine$double.eps)))
  if (object$family == "bernoulli") return(plogis(eta))
  if (object$family == "binomial") {
    if (is.null(trials)) trials <- object$family_options$trials
    if (is.null(trials)) trials <- 1
    if (!is.numeric(trials) || is.complex(trials) || !is.null(dim(trials)) ||
        !(length(trials) %in% c(1L,length(eta))) || any(!is.finite(trials) | trials<=0))
      stop("prediction trials must be positive scalar or length n")
    return(trials*plogis(eta))
  }
  exp(pmin(eta,700))
}

.inference_required <- function(object) {
  if (object$inference$status!="ok") stop(paste("Inference unavailable:",object$inference$reason))
  object$inference
}
vcov.spectralmm_native <- function(object, ...) .inference_required(object)$covariance
confint.spectralmm_native <- function(object, parm, level=.95, ...) {
  if (!is.finite(level) || level<=0 || level>=1) stop("level must be in (0,1)")
  inf <- .inference_required(object)
  critical <- if (inf$statistic_type=="t") qt((1-level)/2,df=inf$df_resid,lower.tail=FALSE) else qnorm((1-level)/2,lower.tail=FALSE)
  intervals <- cbind(lower=coef(object)-critical*inf$std_error,upper=coef(object)+critical*inf$std_error)
  if (missing(parm)) intervals else intervals[parm,,drop=FALSE]
}
summary.spectralmm_native <- function(object, ...) {
  inf <- object$inference
  result <- list(family=object$family, inference=inf, coefficients=NULL)
  if (inf$status=="ok") result$coefficients <- cbind(Estimate=coef(object),`Std. Error`=inf$std_error,
    Statistic=inf$statistic,`Pr(>|statistic|)`=inf$p_value,inf$conf_int)
  if(inf$status=="ok") colnames(result$coefficients)[3:4]<-c(paste(inf$statistic_type,"statistic"),"p-value")
  class(result) <- "summary.spectralmm_native"
  result
}
print.summary.spectralmm_native <- function(x, digits=5, max_rows=20L, ...) {
  cat("SpectralMM (",x$family,")\n",sep="")
  if (x$inference$status!="ok") cat("Inference",x$inference$status,":",x$inference$reason,"\n")
  else {
    cat("Covariance:",x$inference$cov_type," Reference:",x$inference$statistic_type,"\n")
    print(x$coefficients[seq_len(min(nrow(x$coefficients),max_rows)),,drop=FALSE],digits=digits)
    if (nrow(x$coefficients)>max_rows) cat("Further rows are available in summary(model)$coefficients.\n")
  }
  invisible(x)
}
print.spectralmm_native <- function(x, ...) { print(summary(x), ...); invisible(x) }

.spectralmm_print_trace <- function(rows, info, solver, rank, family, options, n, p, intercept, gtol, relgtol) {
  label <- if (solver=="spectral") "PCG" else toupper(solver)
  metric <- if (!is.na(utils::tail(rows$loglikelihood,1))) "LogLik" else "Objective"
  values <- if (metric=="LogLik") rows$loglikelihood else rows$loss
  .smm_print_header(family,options,n,p,intercept,solver,rank); cat("\n")
  cat(sprintf("%4s  %12s  %10s  %10s  %5s  %8s  %s\n",
      "Iter",metric,"GradNorm","RelGrad","Inner","Stepsize","Spectrum"))
  for (i in seq_len(nrow(rows))) {
    r <- rows[i,]
    cat(sprintf("%4d  %12s  %10.3e  %10.3e  %5s  %8s  %s\n",
      as.integer(r$iteration),if (is.na(values[i])) "-" else sprintf("%.4e",values[i]),r$gradnorm,r$relgradnorm,
      if (r$inner<0) "-" else as.character(r$inner),
      if (is.finite(r$step)) sprintf("%.2f",r$step) else "-",r$spectrum))
  }
  .smm_print_terminal(info,metric,utils::tail(values,1),gtol,relgtol)
  cat(if (metric=="LogLik") "LogLik includes distribution constants and excludes ridge; gradients refer to the optimization objective.\n" else "Objective is the summed model loss plus ridge penalty.\n")
}
