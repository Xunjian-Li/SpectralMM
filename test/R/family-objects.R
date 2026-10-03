library(SpectralMM)
set.seed(31)
X<-matrix(rnorm(100*6),100,6); colnames(X)<-paste0('x',1:6)
y<-.3+as.vector(X%*%(1:6/10))+rnorm(100)*.2
dat<-data.frame(y=y,X)
cases<-list(
 list(pseudo_huber(),'pseudo_huber',list(delta=1)),
 list(expectile(.25),'expectile',list(tau=.25)),
 list(smooth_quantile(.25,.1),'smooth_quantile',list(tau=.25,smoothing=.1)),
 list(student_t(4),'student_t',list(nu=4)))
for(case in cases) {
 old<-spectralmm_fit(X,y,family=case[[2]],family_options=case[[3]],rank=5,solver='pcg')
 new<-spectralmm_fit(X,y,family=case[[1]],rank=5,solver='pcg')
 form<-spectralmm_fit(y~.,data=dat,family=case[[1]],rank=5,solver='pcg')
 for(result in list(new,form)) {
  stopifnot(max(abs(coef(old)-coef(result)))<1e-12,
   max(abs(fitted(old)-fitted(result)))<1e-12,identical(old$objective,result$objective),
   result$diagnostics$rank==5,result$diagnostics$solver=='pcg',is.null(result$link))
 }
 stopifnot(max(abs(predict(form,dat)-predict(new,X)))<1e-12)
 for(options in list(list(),NULL,list(delta=2))) {
  err<-try(spectralmm_fit(X,y,family=case[[1]],family_options=options),silent=TRUE)
  stopifnot(inherits(err,'try-error'),grepl('family_options',err))
 }
 stopifnot(inherits(try(spectralmm_glm(X,y,family=case[[1]]),silent=TRUE),'try-error'))
}
for(constructor in list(pseudo_huber,expectile,smooth_quantile,student_t)) {
 for(value in list(0,-1,NA_real_,NaN,Inf,TRUE,'1',c(1,2),NULL))
  stopifnot(inherits(try(constructor(value),silent=TRUE),'try-error'))
}
for(constructor in list(expectile,smooth_quantile))
 stopifnot(inherits(try(constructor(1),silent=TRUE),'try-error'))
for(value in c(0,-1,NaN,Inf))
 stopifnot(inherits(try(smooth_quantile(epsilon=value),silent=TRUE),'try-error'))
stopifnot(expectile()$family_options$tau==.5,smooth_quantile()$family_options$smoothing==.1,
 student_t()$family_options$nu==4)
cat('All four non-GLM family objects: legacy/formula equivalence, rank=5 and validation passed\n')
