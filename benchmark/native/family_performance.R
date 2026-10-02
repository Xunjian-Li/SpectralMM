# Invoked by run.py so Python and R share exact binary input fixtures.
args <- commandArgs(trailingOnly=TRUE)
out <- normalizePath(args[[1]]); root <- normalizePath(args[[2]])
library(Matrix)
library(jsonlite)
library(SpectralMM)
config <- fromJSON(file.path(out,"config.json"),simplifyVector=FALSE)
capture.output(sessionInfo(),file=file.path(out,"environment_R.txt"))
cat("\nLAPACK (La_library): ",La_library(),"\n",file=file.path(out,"environment_R.txt"),append=TRUE)
cat("Thread environment: ",paste(Sys.getenv(c("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")),collapse=","),"\n",file=file.path(out,"environment_R.txt"),append=TRUE)
read_num <- function(path,n) readBin(path,what="double",n=n,size=8,endian="little")
read_int <- function(path,n) readBin(path,what="integer",n=n,size=4,endian="little")
params <- list(negative_binomial=list(theta=4),smooth_quantile=list(tau=.25,smoothing=.1),
  expectile=list(tau=.25),pseudo_huber=list(delta=1),student_t=list(nu=4,sigma=1))
family_params <- function(family) { p <- params[[family]]; if(is.null(p)) list() else p }
score <- function(X,y,b,family,scale) {
  i <- spectralmm_fit(inference=FALSE, intercept=FALSE, X,y,family,family_options=family_params(family),beta0=b,gtol=1e100)$info
  c(loss=i$loss,gradnorm=i$gradnorm,relgrad=i$gradnorm/scale)
}
methods <- c("Spectral-MM","CG","CGLS","CRLS","LSQR","LSMR","GLM")
solvers <- setNames(c("spectral","cg","cgls","crls","lsqr","lsmr"),methods[1:6])
rows <- list(); samples <- list()
for (case in config$cases) {
  base <- file.path(out,"data",case$id); n <- case$n; p <- case$p_total
  y <- read_num(paste0(base,"_y.bin"),n)
  if (case$storage=="dense") X <- matrix(read_num(paste0(base,"_X.bin"),n*p),nrow=n) else
    X <- new("dgCMatrix",Dim=as.integer(c(n,p)),
             x=read_num(paste0(base,"_values.bin"),case$nnz),
             i=read_int(paste0(base,"_indices.bin"),case$nnz),
             p=read_int(paste0(base,"_indptr.bin"),p+1))
  b0 <- read_num(paste0(base,"_beta0.bin"),p)
  scale <- 1+score(X,y,b0,case$family,1)[["gradnorm"]]
  set.seed(case$seed+17)
  coefs <- list(); case_rows <- list()
  for (method in sample(if(case$family %in% c("gaussian","bernoulli","probit","poisson","gamma","negative_binomial")) methods else methods[1:6])) {
    notes <- character()
    row <- list(language="R",family=case$family,storage=case$storage,n=n,p=case$p,p_total=p,method=method,
                median_ms=NA_real_,min_ms=NA_real_,max_ms=NA_real_,samples=0,batch=0,converged=FALSE,quality_pass=FALSE,
                iterations=NA_integer_,inner_iterations=NA_integer_,loss=NA_real_,gradnorm=NA_real_,relgrad=NA_real_,
                coef_norm=NA_real_,relative_coef_error=NA_real_,
                glm_input=if(method=="GLM" && case$storage=="sparse") "dense conversion included" else case$storage,
                warnings="",error="")
    run <- if(method=="GLM") function() {
      XX <- if(case$storage=="sparse") as.matrix(X) else X
      fam <- switch(case$family,gaussian=gaussian(),bernoulli=binomial(),probit=binomial(link="probit"),
        poisson=poisson(),gamma=Gamma(link="log"),negative_binomial=MASS::negative.binomial(theta=4,link="log"))
      m <- glm.fit(XX,y,start=b0,family=fam,control=glm.control(epsilon=config$glm_tolerance,maxit=config$maxiter))
      list(coef=coef(m),converged=m$converged,outer=m$iter,inner=NA_integer_)
    } else function() {
      m <- spectralmm_fit(inference=FALSE, intercept=FALSE, X,y,case$family,family_options=family_params(case$family),beta0=b0,solver=solvers[[method]],rank=config$rank,
                floor=if(is.null(config$floor)) 1e-6 else config$floor,
                krylovdim=config$krylovdim,maxiter=config$maxiter,inner_maxiter=config$inner_maxiter,
                gtol=config$gtol,relgtol=config$relgtol,eta_max=config$eta_max,
                correction_tol=config$correction_tol,resid_tol=config$resid_tol,
                krylov_rtol=config$krylov_rtol,krylov_atol=config$krylov_atol,ridge=0,nesterov=TRUE)
      list(coef=coef(m),converged=m$info$converged,outer=m$info$iterations,inner=m$info$inner_iterations)
    }
    row <- tryCatch({
      start <- proc.time()[["elapsed"]]
      result <- withCallingHandlers(run(),warning=function(w) { notes <<- c(notes,conditionMessage(w)); invokeRestart("muffleWarning") })
      warm <- proc.time()[["elapsed"]]-start
      batch <- max(1,min(config$max_batch,ceiling(config$min_sample_seconds/max(warm,.001))))
      while(warm<config$min_sample_seconds) {
        start <- proc.time()[["elapsed"]]
        for(k in seq_len(batch)) result <- suppressWarnings(run())
        duration <- proc.time()[["elapsed"]]-start
        if(duration>=config$min_sample_seconds || batch>=config$max_batch) break
        batch <- min(config$max_batch,max(batch+1,ceiling(batch*config$min_sample_seconds/max(duration,.001))))
      }
      times <- numeric(config$samples)
      for (j in seq_len(config$samples)) {
        start <- proc.time()[["elapsed"]]
        for (k in seq_len(batch)) result <- suppressWarnings(run())
        times[j] <- (proc.time()[["elapsed"]]-start)*1000/batch
        samples[[length(samples)+1]] <- data.frame(language="R",case=case$id,method=method,sample=j,batch=batch,time_ms=times[j])
      }
      b <- result$coef; metrics <- score(X,y,b,case$family,scale)
      row$median_ms <- median(times); row$min_ms <- min(times); row$max_ms <- max(times)
      row$samples <- length(times); row$batch <- batch; row$converged <- result$converged
      row$quality_pass <- all(is.finite(b)) && (metrics[["gradnorm"]]<=config$gtol || metrics[["relgrad"]]<=config$relgtol)
      if(any(grepl("0 or 1|converge",notes,ignore.case=TRUE))) row$quality_pass <- FALSE
      row$iterations <- result$outer; row$inner_iterations <- result$inner
      row$loss <- metrics[["loss"]]; row$gradnorm <- metrics[["gradnorm"]]; row$relgrad <- metrics[["relgrad"]]
      row$coef_norm <- sqrt(sum(b*b)); row$warnings <- paste(unique(notes),collapse=" | ")
      coefs[[method]] <- b
      row
    },error=function(e) { row$error <- conditionMessage(e); row })
    case_rows[[length(case_rows)+1]] <- row
    cat(sprintf("R %s %s: %.3f ms, quality=%s\n",case$id,method,row$median_ms,row$quality_pass)); flush.console()
  }
  ref <- read_num(paste0(base,"_julia_coef.bin"),p)
  for(j in seq_along(case_rows)) {
    method <- case_rows[[j]]$method
    if(!is.null(coefs[[method]])) case_rows[[j]]$relative_coef_error <- sqrt(sum((coefs[[method]]-ref)^2))/max(1,sqrt(sum(ref^2)))
  }
  rows <- c(rows,case_rows)
  write.csv(do.call(rbind,lapply(rows,as.data.frame)),file.path(out,"R.csv"),row.names=FALSE)
  write.csv(do.call(rbind,samples),file.path(out,"R_samples.csv"),row.names=FALSE)
}
