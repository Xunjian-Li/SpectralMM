args <- commandArgs(trailingOnly=TRUE); out <- args[[1]]; root <- args[[2]]
library(Matrix); library(jsonlite)
library(SpectralMM)
config <- fromJSON(file.path(out,"config.json"),simplifyVector=FALSE)
params <- list(negative_binomial=list(theta=4),smooth_quantile=list(tau=.25,smoothing=.1),expectile=list(tau=.25),pseudo_huber=list(delta=1),student_t=list(nu=4,sigma=1))
read_num <- function(path,n) readBin(path,"double",n=n,endian="little")
read_int <- function(path,n) readBin(path,"integer",n=n,endian="little")
solvers <- c("Spectral-MM (PCG)"="pcg","Spectral-MM (MM)"="mm",CG="cg",CGLS="cgls",CRLS="crls",LSQR="lsqr",LSMR="lsmr")
rows <- list(); samples <- list()
for(case in config$cases) {
 base <- file.path(out,"data",case$id); n <- case$n; p <- case$p_total
 y <- read_num(paste0(base,"_y.bin"),n); b0 <- read_num(paste0(base,"_beta0.bin"),p)
 if(case$storage=="dense") X <- matrix(read_num(paste0(base,"_X.bin"),n*p),nrow=n) else
  X <- new("dgCMatrix",Dim=as.integer(c(n,p)),x=read_num(paste0(base,"_values.bin"),case$nnz),i=read_int(paste0(base,"_indices.bin"),case$nnz),p=read_int(paste0(base,"_indptr.bin"),p+1))
 fp <- params[[case$family]]; if(is.null(fp)) fp <- list()
 run <- function(method) spectralmm_fit(inference=FALSE, intercept=FALSE, X,y,case$family,family_options=fp,beta0=b0,solver=solvers[[method]],
   rank=config$rank,krylovdim=config$krylovdim,floor=config$floor,ridge=0,maxiter=config$maxiter,inner_maxiter=config$inner_maxiter,eta_max=config$eta_max,
   correction_tol=config$correction_tol,resid_tol=config$resid_tol,gtol=config$gtol,relgtol=config$relgtol,nesterov=TRUE,
   krylov_rtol=config$krylov_rtol,krylov_atol=config$krylov_atol,
   accept_negligible=config$accept_negligible,accept_stalled=config$accept_stalled)
 batches <- list(); results <- list(); times <- setNames(lapply(solvers,function(x) numeric()),names(solvers))
 for(method in names(solvers)) {
  start <- proc.time()[["elapsed"]]; run(method); warm <- proc.time()[["elapsed"]]-start
  batches[[method]] <- max(1,min(config$max_batch,ceiling(config$min_sample_seconds/max(.001,warm))))
 }
 set.seed(718)
 for(sample in seq_len(config$repeat_samples)) for(method in sample(names(solvers))) {
  start <- proc.time()[["elapsed"]]
  for(k in seq_len(batches[[method]])) results[[method]] <- run(method)
  ms <- 1000*(proc.time()[["elapsed"]]-start)/batches[[method]]
  times[[method]] <- c(times[[method]],ms)
  samples[[length(samples)+1]] <- data.frame(language="R",case=case$id,method=method,sample=sample,batch=batches[[method]],time_ms=ms)
 }
 for(method in names(solvers)) {
  i <- results[[method]]$info
  rows[[length(rows)+1]] <- data.frame(language="R",family=case$family,storage=case$storage,method=method,
   median_ms=median(times[[method]]),min_ms=min(times[[method]]),max_ms=max(times[[method]]),
   quality_pass=i$gradient_converged,converged=i$converged,termination=i$termination_reason,
   iterations=i$iterations,inner_iterations=i$inner_iterations,loss=i$loss,gradnorm=i$gradnorm,relgrad=i$relgradnorm)
 }
 write.csv(do.call(rbind,rows),file.path(out,"R_repeat.csv"),row.names=FALSE)
 write.csv(do.call(rbind,samples),file.path(out,"R_repeat_samples.csv"),row.names=FALSE)
 cat(case$id,"completed\n"); flush.console()
}
