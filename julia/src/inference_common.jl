# Shared reporting policy for original Julia and the native Julia wrapper.
function _check_inference_options(inference, max_p, cov_type, level, use_t, dispersion)
    (inference === :auto || inference isa Bool) || throw(ArgumentError("inference must be :auto, true or false"))
    (max_p isa Integer && !(max_p isa Bool) && max_p>0) || throw(ArgumentError("inference_max_p must be a positive integer"))
    cov_type in (:auto,:model,:sandwich) || throw(ArgumentError("cov_type must be :auto, :model or :sandwich"))
    0<level<1 || throw(ArgumentError("level must be in (0,1)"))
    (use_t===nothing || use_t isa Bool) || throw(ArgumentError("use_t must be nothing or boolean"))
    (dispersion===nothing || (isfinite(dispersion) && dispersion>0)) || throw(ArgumentError("dispersion must be positive and finite"))
end
_inference_unavailable(reason) = (status="unavailable", reason=string(reason))
function _inference_gate(inference, max_p, p, n, ridge, gradient_converged=true)
    inference === false && return (status="disabled",reason="inference=false")
    inference === :auto && p>max_p && return (status="skipped",reason="parameter count exceeds inference_max_p")
    ridge!=0 && return _inference_unavailable("ordinary Wald inference is unavailable for ridge-penalized fits")
    !gradient_converged && return _inference_unavailable("fit did not satisfy the gradient convergence criterion")
    n<=p && return _inference_unavailable("inference requires n > number of parameters")
    return nothing
end
function _finish_inference(result, inference)
    inference === true && result.status=="unavailable" && @warn "Inference unavailable: $(result.reason)"
    return result
end
function _wald_result(beta, covariance, cov_type, estimated_dispersion, df_resid, dispersion, rcond, level, use_t)
    student=use_t===nothing ? estimated_dispersion : use_t
    dist=student ? Distributions.TDist(df_resid) : Distributions.Normal()
    se=sqrt.(diag(covariance)); statistic=beta./se
    critical=Distributions.cquantile(dist,(1-level)/2)
    return (status="ok",reason="",covariance=covariance,std_error=se,statistic=statistic,
        statistic_type=student ? "t" : "z",wald_chisq=statistic.^2,
        wald_p_value=Distributions.ccdf.(Ref(Distributions.Chisq(1)),statistic.^2),
        p_value=2 .* Distributions.ccdf.(Ref(dist),abs.(statistic)),
        conf_int=hcat(beta.-critical.*se,beta.+critical.*se),level=level,
        df_resid=df_resid,dispersion=dispersion,rcond=rcond,cov_type=string(cov_type),
        estimated_dispersion=estimated_dispersion)
end
function _require_inference(model)
    model.inference.status=="ok" || throw(ArgumentError("Inference unavailable: $(model.inference.reason)"))
    model.inference
end
