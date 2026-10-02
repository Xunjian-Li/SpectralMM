library(SpectralMM)
fixture<-if(file.exists('data.csv')) 'data.csv' else 'examples/data.csv'
d<-read.csv(fixture); X<-as.matrix(d[c('x1','x2','x3')])
cases<-c('gaussian','bernoulli','probit','poisson','gamma','negative_binomial')
families<-list(gaussian(),binomial('logit'),binomial('probit'),poisson(),Gamma('log'),'negative_binomial')
exports<-list()
for(k in seq_along(cases)) {
  name<-cases[k]; y<-d[[name]]; dd<-d; dd$y<-y
  options<-if(name=='negative_binomial') list(theta=4) else list()
  m<-spectralmm_glm(X,y,family=families[[k]],family_options=options,start=rep(0,4))
  f<-spectralmm_glm(y~x1+x2+x3,data=dd,family=families[[k]],family_options=options,start=rep(0,4))
  a<-spectralmm_glm(cbind(1,X),y,intercept=FALSE,family=families[[k]],family_options=options,start=rep(0,4))
  stopifnot(max(abs(coef(m)-coef(f)))<1e-10,max(abs(coef(m)-coef(a)))<1e-10,
    max(abs(fitted(m)-fitted(f)))<1e-10,abs(m$objective-f$objective)<1e-9,
    max(abs(predict(f,dd)-predict(m)))<1e-10,
    max(abs(predict(f,dd,type='response')-fitted(m)))<1e-10,
    identical(nobs(m),100L),m$info$gradient_converged,m$inference$status=='ok',
    max(abs(residuals(m)-(y-fitted(m))))<1e-14,all(dim(vcov(m))==4L),all(dim(confint(m))==c(4L,2L)))
  if(k<=5L) {
    ref<-stats::glm(y~x1+x2+x3,data=dd,family=families[[k]],
      control=stats::glm.control(epsilon=1e-12,maxit=100))
    stopifnot(max(abs(coef(m)-coef(ref)))<1e-6,abs(deviance(m)-deviance(ref))<1e-7)
  }
  vals<-list(coef=coef(m),eta=predict(m),mu=fitted(m),objective=m$objective,
    deviance=deviance(m),loglikelihood=as.numeric(logLik(m)),converged=as.numeric(m$info$gradient_converged))
  for(q in names(vals)) exports[[length(exports)+1L]]<-data.frame(family=name,quantity=q,index=seq_along(vals[[q]])-1L,value=vals[[q]])
}
if(nzchar(Sys.getenv('SPECTRALMM_API_EXPORT'))) write.csv(do.call(rbind,exports),Sys.getenv('SPECTRALMM_API_EXPORT'),row.names=FALSE)
d<-data.frame(x=sin(seq_len(100)-1),g=factor(rep(c('a','b','c','a'),25)))
d$y<-.3+.4*d$x+.2*(d$g=='b')-.1*(d$g=='c')+.1*cos((seq_len(100)-1)*.7)
for(formula in list(y~x+g,y~0+x+g)) {
  f<-spectralmm_glm(formula,data=d)
  A<-model.matrix(formula,d)
  m<-spectralmm_glm(A,d$y,intercept=FALSE)
  stopifnot(max(abs(coef(f)-coef(m)))<1e-10)
  nd<-data.frame(x=c(.2,.5),g=factor(c('c','b'),levels=levels(d$g)))
  manual<-as.vector(model.matrix(delete.response(terms(formula)),nd)%*%coef(f))
  stopifnot(max(abs(predict(f,nd,type='response')-manual))<1e-10)
  bad<-try(predict(f,data.frame(x=1,g='unseen')),silent=TRUE); stopifnot(inherits(bad,'try-error'))
}
set.seed(9); big<-matrix(rnorm(80*51),80,51); y<-sin(seq_len(80))
m<-spectralmm_glm(big,y,control=spectralmm_control(maxiter=1))
stopifnot(m$inference$status=='skipped',is.null(m$inference$covariance),nrow(summary(m)$coefficients)==52L)
stopifnot(inherits(try(spectralmm_glm(big,y,weights=rep(1,80)),silent=TRUE),'try-error'))
stopifnot(inherits(try(spectralmm_glm(y~x+offset(x),data=d),silent=TRUE),'try-error'))
stopifnot(inherits(try(predict(m,offset=rep(1,80)),silent=TRUE),'try-error'))
stopifnot(inherits(try(spectralmm_glm(big,y,control=spectralmm_control(maxiter=1),maxiter=2),silent=TRUE),'try-error'))
f<-spectralmm_fit(y~x,data=d,family='pseudo_huber')
stopifnot(is.null(f$link),length(fitted(f))==100L)
cat('Statistical API: six GLMs, formula/matrix, categories, prediction and inference gates passed\n')
