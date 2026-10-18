args <- commandArgs(trailingOnly=TRUE); out <- args[[1]]; root <- args[[2]]
library(Matrix); library(jsonlite)
library(SpectralMM)
config <- fromJSON(file.path(out,"config.json"),simplifyVector=FALSE)
source(file.path(root,"benchmark/python_r/common.R"))
rows <- list(); samples <- list()
for(case in config$cases) {
 input <- load_case(out,case); X <- input$X; y <- input$y; b0 <- input$b0
 run <- function(method) native_fit(X,y,case$family,b0,method,config)
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
