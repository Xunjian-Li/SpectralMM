# Lightweight non-GLM specifications, normalized before calling the binding.
.smm_validate_parameter <- function(name,value,probability=FALSE) {
  if(!is.numeric(value) || is.complex(value) || length(value)!=1L ||
     !is.null(dim(value)) || !is.finite(value) || value<=0 ||
     (probability && value>=1)) {
    constraint<-if(probability) "strictly between 0 and 1" else "positive"
    stop(name," must be a finite real scalar, ",constraint,call.=FALSE)
  }
}
.smm_family <- function(name,options) {
  structure(list(family=name,family_options=options),class="spectralmm_family")
}
pseudo_huber <- function(delta=1) {
  .smm_validate_parameter("delta",delta)
  .smm_family("pseudo_huber",list(delta=delta))
}
expectile <- function(q=.5) {
  .smm_validate_parameter("q",q,TRUE)
  .smm_family("expectile",list(q=q))
}
smooth_quantile <- function(q=.5,epsilon=.1) {
  .smm_validate_parameter("q",q,TRUE)
  .smm_validate_parameter("epsilon",epsilon)
  .smm_family("smooth_quantile",list(q=q,smoothing=epsilon))
}
student_t <- function(nu=4) {
  .smm_validate_parameter("nu",nu)
  .smm_family("student_t",list(nu=nu))
}
.smm_normalize_family <- function(family,options) {
  if(inherits(family,"spectralmm_family")) {
    if("family_options" %in% names(options))
      stop("do not supply family_options with a family object; use constructor parameters",call.=FALSE)
    options$family_options<-family$family_options
    family<-family$family
  }
  list(family=family,options=options)
}
