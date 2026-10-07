library(SpectralMM)
d<-read.csv(if(file.exists('data.csv')) 'data.csv' else if(file.exists('tests/data.csv')) 'tests/data.csv' else 'R/tests/data.csv')
X<-as.matrix(d[c('x1','x2','x3')])
for(name in setdiff(names(d),c('x1','x2','x3'))) {
 opts<-if(name=='binomial') list(trials=5) else list()
 for(dispersion in list(NULL,.7)) {
  quiet<-spectralmm_fit(X,d[[name]],family=name,family_options=opts,inference=FALSE,ridge=.1,dispersion=dispersion,maxiter=4)
  txt<-capture.output(m<-spectralmm_fit(X,d[[name]],family=name,family_options=opts,inference=FALSE,ridge=.1,dispersion=dispersion,maxiter=4,verbose=TRUE,trace=TRUE))
  stopifnot(identical(coef(m),coef(quiet)),any(grepl('GradNorm',txt)),any(grepl('Stepsize',txt)),!any(grepl('InnerRes|EigRes',txt)))
  if(name %in% c('pseudo_huber','expectile','smooth_quantile','tweedie')) stopifnot(any(grepl('Objective',txt)))
  else stopifnot(abs(tail(m$trace$loglikelihood,1)-as.numeric(logLik(m)))<1e-8)
 }
}
set.seed(91); new<-matrix(rnorm(21),7,3); colnames(new)<-colnames(X)
for(family in list(pseudo_huber(),expectile(.3),smooth_quantile(.3,.2),student_t(4))) {
 for(intercept in c(TRUE,FALSE)) {
  m<-spectralmm_fit(X,d$gaussian,family=family,intercept=intercept,inference=FALSE)
  expected<-if(intercept) drop(new%*%coef(m)[-1]+coef(m)[1]) else drop(new%*%coef(m))
  stopifnot(max(abs(predict(m,newdata=new)-expected))<1e-10)
  dat<-data.frame(y=d$gaussian,X)
  fm<-spectralmm_fit(if(intercept) y~x1+x2+x3 else y~0+x1+x2+x3,data=dat,family=family,inference=FALSE)
  stopifnot(max(abs(predict(fm,newdata=as.data.frame(new),type='response')-expected))<1e-8)
 }
}

for(family in c("expectile","smooth_quantile")) {
 a<-spectralmm_fit(X,d$gaussian,family=family,family_options=list(q=.3))
 b<-spectralmm_fit(X,d$gaussian,family=family,family_options=list(tau=.3))
 stopifnot(identical(coef(a),coef(b)),"q" %in% names(a$family_options),!"tau" %in% names(a$family_options))
 stopifnot(inherits(try(spectralmm_fit(X,d$gaussian,family=family,family_options=list(q=.3,tau=.3)),silent=TRUE),"try-error"))
 txt<-capture.output(print(a)); stopifnot(sum(grepl("Family:",txt))==2L,any(grepl("q=0.3",txt)),!any(grepl("tau",txt)))
}
for(name in c("gaussian","bernoulli")) {
 m<-spectralmm_fit(X,d[[name]],family=name)
 txt<-capture.output(print(m)); stat<-if(name=="gaussian") "t statistic" else "z statistic"
 stopifnot(any(grepl(stat,txt)),any(grepl("p-value",txt)),startsWith(txt[1],"SpectralMM Regression Model"))
}
