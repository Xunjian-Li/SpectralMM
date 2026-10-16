# Shared input loading and native fitting for initial and repeated measurements.
params <- list(negative_binomial=list(theta=4),smooth_quantile=list(tau=.25,smoothing=.1),
               expectile=list(tau=.25),pseudo_huber=list(delta=1),student_t=list(nu=4,sigma=1))
family_params <- function(family) { p <- params[[family]]; if(is.null(p)) list() else p }
methods <- c("Spectral-MM (PCG)","Spectral-MM (MM)","CG","CGLS","CRLS","LSQR","LSMR","GLM")
solvers <- setNames(c("pcg","mm","cg","cgls","crls","lsqr","lsmr"),methods[1:7])
read_num <- function(path,n) readBin(path,"double",n=n,size=8,endian="little")
read_int <- function(path,n) readBin(path,"integer",n=n,size=4,endian="little")
load_case <- function(out,case) {
  base <- file.path(out,"data",case$id); n <- case$n; p <- case$p_total
  y <- read_num(paste0(base,"_y.bin"),n)
  b0 <- read_num(paste0(base,"_beta0.bin"),p)
  if(case$storage=="dense") X <- matrix(read_num(paste0(base,"_X.bin"),n*p),nrow=n) else
    X <- new("dgCMatrix",Dim=as.integer(c(n,p)),
             x=read_num(paste0(base,"_values.bin"),case$nnz),
             i=read_int(paste0(base,"_indices.bin"),case$nnz),
             p=read_int(paste0(base,"_indptr.bin"),p+1))
  list(X=X,y=y,b0=b0)
}
native_fit <- function(X,y,family,b0,method,config) {
  keys <- c("rank","krylovdim","floor","maxiter","inner_maxiter","eta_max",
            "correction_tol","resid_tol","gtol","relgtol","krylov_rtol","krylov_atol",
            "accept_negligible","accept_stalled")
  if(is.null(config$floor)) config$floor <- 1e-6
  do.call(spectralmm_fit,c(list(X=X,y=y,family=family,beta0=b0,
     solver=solvers[[method]],family_options=family_params(family),
     inference=FALSE,intercept=FALSE,ridge=0,nesterov=TRUE),config[keys]))
}
