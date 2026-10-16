# Invoked by family_performance.py with shared binary input fixtures.
args <- commandArgs(trailingOnly=TRUE)
out <- normalizePath(args[[1]]); root <- normalizePath(args[[2]])
library(Matrix)
library(jsonlite)
library(SpectralMM)
config <- fromJSON(file.path(out,"config.json"),simplifyVector=FALSE)
capture.output(sessionInfo(),file=file.path(out,"environment_R.txt"))
cat("\nLAPACK (La_library): ",La_library(),"\n",file=file.path(out,"environment_R.txt"),append=TRUE)
cat("Thread environment: ",paste(Sys.getenv(c("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")),collapse=","),"\n",file=file.path(out,"environment_R.txt"),append=TRUE)
source(file.path(root,"benchmark/native/common.R"))
score <- function(X,y,b,family,scale) {
  i <- spectralmm_fit(inference=FALSE, intercept=FALSE, X,y,family,family_options=family_params(family),beta0=b,gtol=1e100)$info
  c(loss=i$loss,gradnorm=i$gradnorm,relgrad=i$gradnorm/scale)
}
rows <- list(); samples <- list()
for (case in config$cases) {
  n <- case$n; p <- case$p_total
  input <- load_case(out,case); X <- input$X; y <- input$y; b0 <- input$b0
  scale <- 1+score(X,y,b0,case$family,1)[["gradnorm"]]
  set.seed(case$seed+17)
  coefs <- list(); case_rows <- list()
  for (method in sample(if(case$family %in% c("gaussian","bernoulli","probit","poisson","gamma","negative_binomial")) methods else methods[1:7])) {
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
      m <- native_fit(X,y,case$family,b0,method,config)
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
  ref <- read_num(file.path(out,"data",paste0(case$id,"_julia_coef.bin")),p)
  for(j in seq_along(case_rows)) {
    method <- case_rows[[j]]$method
    if(!is.null(coefs[[method]])) case_rows[[j]]$relative_coef_error <- sqrt(sum((coefs[[method]]-ref)^2))/max(1,sqrt(sum(ref^2)))
  }
  rows <- c(rows,case_rows)
  write.csv(do.call(rbind,lapply(rows,as.data.frame)),file.path(out,"R.csv"),row.names=FALSE)
  write.csv(do.call(rbind,samples),file.path(out,"R_samples.csv"),row.names=FALSE)
}
