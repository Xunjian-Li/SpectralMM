# ============================================================
# SpectralGLM
# ============================================================

StatsAPI.coef(m::SpectralGLM) = m.coef
StatsAPI.response(m::SpectralGLM) = m.y
StatsAPI.modelmatrix(m::SpectralGLM) = m.X
StatsAPI.nobs(m::SpectralGLM) = length(m.y)

StatsModels.coefnames(m::SpectralGLM) =
    m.formula === nothing ?
    (m.intercept ? vcat("(Intercept)", ["x$(j)" for j in 1:length(m.coef)-1]) : ["x$(j)" for j in eachindex(m.coef)]) :
    StatsModels.coefnames(m.formula.rhs)

function StatsAPI.predict(m::SpectralGLM, Xnew::AbstractMatrix)
    expected = length(m.coef) - m.intercept
    size(Xnew,2) == expected || throw(DimensionMismatch("prediction feature count differs from training"))
    η = m.intercept ? Xnew * view(m.coef, 2:length(m.coef)) .+ m.coef[1] : Xnew * m.coef
    return GLM.linkinv.(Ref(m.link), η)
end

function StatsAPI.predict(m::SpectralGLM, newdata)
    m.formula === nothing &&
        throw(ArgumentError(
            "table prediction requires a model fitted with the formula interface"
        ))

    Xnew = StatsModels.modelcols(m.formula.rhs, newdata)
    return StatsAPI.predict(m, Xnew)
end

StatsAPI.predict(m::SpectralGLM) =
    GLM.linkinv.(Ref(m.link), m.X * m.coef)

StatsAPI.fitted(m::SpectralGLM) =
    StatsAPI.predict(m)


# ============================================================
# SpectralModel
# ============================================================

StatsAPI.coef(m::SpectralModel) = m.coef
StatsAPI.response(m::SpectralModel) = m.y
StatsAPI.modelmatrix(m::SpectralModel) = m.X
StatsAPI.nobs(m::SpectralModel) = length(m.y)

StatsModels.coefnames(m::SpectralModel) =
    m.intercept ? vcat("(Intercept)", ["x$(j)" for j in 1:length(m.coef)-1]) : ["x$(j)" for j in eachindex(m.coef)]

function _response_prediction(f::IRLSFamily, eta; trials=nothing)
    trials === nothing || f isa BinomialLogit || throw(ArgumentError("trials only applies to binomial models"))
    if f isa BinomialLogit
        t=isnothing(trials) ? f.n : trials
        t isa Real || t isa AbstractVector{<:Real} || throw(ArgumentError("trials must be a real scalar or vector"))
        t isa AbstractVector && length(t)==1 && (t=only(t))
        t isa Real || length(t)==length(eta) || throw(DimensionMismatch("trials must be scalar or match prediction rows"))
        all(v -> isfinite(v) && v>0,t isa Real ? (t,) : t) || throw(ArgumentError("trials must be finite and positive"))
        f=BinomialLogit(t)
    end
    return f isa GLMFamily ? [mean_eta(f,e,i) for (i,e) in enumerate(eta)] : eta
end

function StatsAPI.predict(m::SpectralModel, Xnew::AbstractMatrix; trials=nothing)
    size(Xnew,2) == length(m.coef)-m.intercept || throw(DimensionMismatch("prediction feature count differs from training"))
    eta = m.intercept ? Xnew * view(m.coef,2:length(m.coef)) .+ m.coef[1] : Xnew*m.coef
    return _response_prediction(m.family,eta;trials)
end

StatsAPI.predict(m::SpectralModel) =
    _response_prediction(m.family,m.X * m.coef)

StatsAPI.fitted(m::SpectralModel) =
    StatsAPI.predict(m)

StatsAPI.vcov(m::Union{SpectralGLM,SpectralModel}) = _require_inference(m).covariance
StatsAPI.stderror(m::Union{SpectralGLM,SpectralModel}) = _require_inference(m).std_error
function StatsAPI.confint(m::Union{SpectralGLM,SpectralModel};level=.95)
    0<level<1 || throw(ArgumentError("level must be in (0,1)"))
    r=_require_inference(m)
    dist=r.statistic_type=="t" ? Distributions.TDist(r.df_resid) : Distributions.Normal()
    q=Distributions.cquantile(dist,(1-level)/2)
    hcat(m.coef.-q.*r.std_error,m.coef.+q.*r.std_error)
end
function StatsAPI.coeftable(m::Union{SpectralGLM,SpectralModel})
    r=_require_inference(m)
    return (term=StatsModels.coefnames(m),estimate=m.coef,std_error=r.std_error,
        statistic=r.statistic,p_value=r.p_value,lower=r.conf_int[:,1],upper=r.conf_int[:,2])
end

function _show_inference(io::IO, m)
    r=m.inference
    if r.status!="ok"
        print(io,"\nInference ",r.status,": ",r.reason)
        return
    end
    println(io,"\nCovariance: ",r.cov_type,"; reference: ",r.statistic_type)
    @printf(io,"%-18s %12s %12s %12s %12s\n","Term","Estimate","Std. Error","Statistic","p-value")
    names=StatsModels.coefnames(m)
    for j in 1:min(20,length(m.coef))
        @printf(io,"%-18s %12.5g %12.5g %12.5g %12.5g\n",names[j],m.coef[j],r.std_error[j],r.statistic[j],r.p_value[j])
    end
    length(m.coef)>20 && print(io,"Further rows are available through coeftable(model).")
end
function Base.show(io::IO, ::MIME"text/plain", m::SpectralGLM)
    print(io,"SpectralGLM; Backend: Julia; Solver: ",uppercase(string(m.solver)),"; parameters=",length(m.coef),"; converged=",m.result.converged)
    _show_inference(io,m)
end
