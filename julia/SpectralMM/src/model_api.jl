# Statistical metadata and interfaces; no numerical algorithm is implemented here.
struct SpectralMMControl{T<:NamedTuple}
    options::T
end
function SpectralMMControl(;kwargs...)
    allowed=(:maxiter,:inner_maxiter,:krylovdim,:ridge,:floor,:gtol,:relgtol,:eta_max,
        :correction_tol,:resid_tol,:nesterov,:preconditioner,:krylov_rtol,:krylov_atol,
        :krylov_reltol,:krylov_abstol,:accept_negligible,:accept_stalled,
        :negligible_step_tol,:step_reltol,:stalled_relgtol,:line_search,:alpha_min,
        :backtrack_factor,:c_armijo,:chol_jitter,:max_chol_tries)
    all(k->k in allowed,keys(kwargs)) || throw(ArgumentError("unknown control option"))
    SpectralMMControl((;kwargs...))
end
struct StatisticalResult{M,F,L,T} <: StatsAPI.RegressionModel
    model::M
    family::F
    link::L
    formula::T
    coefficient_names::Vector{String}
    linear_predictor::Vector{Float64}
    fittedvalues::Vector{Float64}
    y::Vector{Float64}
    optimization::NamedTuple
    deviance_value::Union{Nothing,Float64}
    likelihood_value::Union{Nothing,Float64}
    intercept::Bool
end
function Base.getproperty(m::StatisticalResult,k::Symbol)
    k===:objective && return getfield(m,:optimization).objective
    k===:nparams && return length(coef(getfield(m,:model)))
    k in fieldnames(typeof(m)) && return getfield(m,k)
    getproperty(getfield(m,:model),k)
end
Base.propertynames(m::StatisticalResult,private::Bool=false) =
    (fieldnames(typeof(m))..., :objective, :nparams, propertynames(getfield(m,:model),private)...)
StatsAPI.coef(m::StatisticalResult)=coef(m.model)
StatsAPI.nobs(m::StatisticalResult)=length(m.y)
StatsAPI.response(m::StatisticalResult)=m.y
StatsAPI.fitted(m::StatisticalResult)=copy(m.fittedvalues)
StatsAPI.residuals(m::StatisticalResult)=m.y.-m.fittedvalues
StatsModels.coefnames(m::StatisticalResult)=m.coefficient_names
StatsAPI.vcov(m::StatisticalResult)=vcov(m.model)
StatsAPI.stderror(m::StatisticalResult)=stderror(m.model)
function StatsAPI.confint(m::StatisticalResult;level=.95)
    0<level<1 || throw(ArgumentError("level must lie in (0,1)"))
    r=_require_inference(m)
    dist=r.statistic_type=="t" ? Distributions.TDist(r.df_resid) : Normal()
    q=Distributions.cquantile(dist,(1-level)/2)
    hcat(coef(m).-q.*r.std_error,coef(m).+q.*r.std_error)
end
function StatsAPI.coeftable(m::StatisticalResult)
    r=_require_inference(m)
    GLM.CoefTable(hcat(coef(m),r.std_error,r.statistic,r.p_value),
        ["Estimate","Std. Error",r.statistic_type*" value","Pr(>|"*r.statistic_type*"|)"],
        m.coefficient_names,4,3)
end
function StatsAPI.deviance(m::StatisticalResult)
    m.deviance_value===nothing && throw(ArgumentError("deviance is not implemented for this loss"))
    m.deviance_value
end
function StatsAPI.loglikelihood(m::StatisticalResult)
    m.likelihood_value===nothing && throw(ArgumentError("normalized likelihood is not implemented for this loss"))
    m.likelihood_value
end
diagnostics(m::StatisticalResult)=m.optimization

const _PUBLIC_GLM = Dict(
    :gaussian=>(:gaussian,:identity),:gaussian_log=>(:gaussian,:log),
    :bernoulli=>(:binomial,:logit),:probit=>(:binomial,:probit),
    :binomial=>(:binomial,:logit),:poisson=>(:poisson,:log),
    :gamma=>(:gamma,:log),:gamma_inverse=>(:gamma,:inverse),
    :negative_binomial=>(:negative_binomial,:log))
_xlog(a,b)=iszero(a) ? 0.0 : a*log(b)
function _fit_statistics(name,y,mu,eta,p,options,dispersion)
    if name in (:gaussian,:gaussian_log)
        dev=sum(abs2,y.-mu); scale=isnothing(dispersion) ? dev/length(y) : dispersion
        ll=iszero(scale) ? Inf : sum(logpdf(Normal(m,sqrt(scale)),v) for (m,v) in zip(mu,y))
    elseif name in (:bernoulli,:probit,:binomial)
        trials=get(options,:trials,1.)
        nt=trials isa Real ? fill(trials,length(y)) : length(trials)==1 ? fill(only(trials),length(y)) : trials
        lp=name===:probit ? Distributions.logcdf.(Normal(),eta) : [-max(-e,0)-log1p(exp(-abs(e))) for e in eta]
        lq=name===:probit ? Distributions.logcdf.(Normal(),-eta) : [-max(e,0)-log1p(exp(-abs(e))) for e in eta]
        kernel=sum(y.*lp.+(nt.-y).*lq)
        dev=2*(sum(_xlog(v,v/n)+_xlog(n-v,(n-v)/n) for (n,v) in zip(nt,y))-kernel)
        # Count likelihoods require integer responses/trials; fractional quasi-data
        # have an objective but no normalized binomial probability mass function.
        ll=all(isinteger,y) && all(isinteger,nt) ? sum(logpdf(Binomial(Int(n),.5),Int(v))+n*log(2) for (n,v) in zip(nt,y))+kernel : NaN
    elseif name===:poisson
        dev=2*sum(_xlog(v,v/m)-v+m for (v,m) in zip(y,mu))
        ll=all(isinteger,y) ? sum(logpdf(Poisson(m),v) for (m,v) in zip(mu,y)) : NaN
    elseif name in (:gamma,:gamma_inverse)
        dev=2*sum((y.-mu)./mu.-log.(y./mu))
        scale=isnothing(dispersion) ? (length(y)>p ? sum(abs2,(y.-mu)./mu)/(length(y)-p) : NaN) : dispersion
        ll=isfinite(scale) && scale>0 ? sum(logpdf(Gamma(1/scale,m*scale),v) for (m,v) in zip(mu,y)) : NaN
    elseif name===:negative_binomial
        theta=get(options,:theta,1.)
        dev=2*sum(_xlog(v,v/m)-(v+theta)*log((v+theta)/(m+theta)) for (v,m) in zip(y,mu))
        ll=all(isinteger,y) ? sum(logpdf(NegativeBinomial(theta,theta/(theta+m)),v) for (m,v) in zip(mu,y)) : NaN
    else
        return nothing,nothing
    end
    Float64(dev),Float64(ll)
end
function _result_metadata(raw,X,y,name,options,intercept,formula,names,settings)
    beta=coef(raw)
    eta=Vector{Float64}(intercept ? X*view(beta,2:length(beta)).+beta[1] : X*beta)
    mu=raw isa _CppBackend.FittedModel ? _CppBackend.predict(raw,X) : predict(raw)
    p=length(beta)
    solver=get(settings,:solver,:pcg); solver=solver===:spectral ? :pcg : solver===:cholesky ? :cho : solver
    rank=get(settings,:rank,nothing); rank=isnothing(rank) ? _default_rank(p) : rank
    defaults=(maxiter=200,inner_maxiter=100,gtol=1e-7,relgtol=1e-8,floor=1e-6,
        krylovdim=raw isa _CppBackend.FittedModel ? min(p,max(3*(rank+1)+20,rank+11)) : min(p,max(12,rank+1)))
    cfg=merge(defaults,settings)
    if raw isa _CppBackend.FittedModel
        i=raw.info
        reasons=("gradient tolerance reached","maximum iterations reached","line search failed","inner breakdown","negligible step","stalled near tolerance")
        reason=reasons[i.termination+1]
        diag=(solver=solver,rank=rank,converged=i.converged,termination_reason=reason,
              outer_iterations=i.iterations,inner_iterations=i.inner_iterations,gradient_norm=i.gradnorm,
              relative_gradient=i.relgradnorm,objective=i.loss,backend=:cpp)
    else
        r=raw.result
        # The old result does not retain each early-exit label. Never invent one.
        grad=last(r.gradnorms)<=cfg.gtol || last(r.relgradnorms)<=cfg.relgtol
        reason=grad ? "gradient tolerance reached" : r.iters>=cfg.maxiter ? "maximum iterations reached" :
            r.converged ? "accepted step-based termination (exact reason not retained)" : "stopped before convergence (exact reason not retained)"
        inner=hasproperty(r,:inner_iters) ? sum(r.inner_iters) : nothing
        diag=(solver=solver,rank=rank,converged=r.converged,termination_reason=reason,
              outer_iterations=r.iters,inner_iterations=inner,gradient_norm=last(r.gradnorms),
              relative_gradient=last(r.relgradnorms),objective=last(r.losses),backend=:julia)
    end
    diag=merge(diag,(krylov_dimension=cfg.krylovdim,preconditioner=get(cfg,:preconditioner,nothing),control=cfg))
    d,ll=_fit_statistics(name,y,mu,eta,p,options,get(settings,:dispersion,nothing))
    family,link=get(_PUBLIC_GLM,name,(name,nothing))
    StatisticalResult(raw,family,link,formula,String.(names),eta,Vector{Float64}(mu),Vector{Float64}(y),diag,d,ll,intercept)
end
function _statistical_fit(X,y,name;data=nothing,backend=:julia,family_options=NamedTuple(),
                          intercept=nothing,start=nothing,beta0=nothing,control=SpectralMMControl(),
                          weights=nothing,offset=nothing,kwargs...)
    weights===nothing && offset===nothing || throw(ArgumentError("observation weights and offset are not implemented by the numerical core"))
    start===nothing || beta0===nothing || throw(ArgumentError("use start or beta0, not both"))
    control isa SpectralMMControl || throw(ArgumentError("control must be SpectralMMControl"))
    isempty(intersect(keys(control.options),keys(kwargs))) || throw(ArgumentError("duplicate direct/control options"))
    settings=merge(control.options,(;kwargs...))
    f=nothing
    if X isa StatsModels.FormulaTerm
        y===nothing && data!==nothing || throw(ArgumentError("formula requires data and no separate y"))
        intercept===nothing || throw(ArgumentError("the formula controls the intercept"))
        # StatsModels does not support GLM offsets here: do not encode one as a coefficient.
        occursin(r"\boffset\s*\(",string(X)) && throw(ArgumentError("formula offsets are not implemented"))
        f=StatsModels.apply_schema(X,StatsModels.schema(X,data),SpectralGLM)
        y,X=StatsModels.modelcols(f,data)
        names=String.(StatsModels.coefnames(f.rhs))
        intercept=StatsModels.hasintercept(f)
        if intercept
            first(names)=="(Intercept)" || throw(ArgumentError("unsupported formula intercept ordering"))
            X=X[:,2:end]
        end
    else
        data===nothing || throw(ArgumentError("data= is only valid for formulas"))
        intercept=isnothing(intercept) ? true : intercept
        names=vcat(intercept ? ["(Intercept)"] : String[],["x$j" for j in 1:size(X,2)])
    end
    raw=_fit_backend(X,y;family=name,backend,family_options,intercept,
                     beta0=isnothing(start) ? beta0 : start,settings...)
    _result_metadata(raw,X,y,Symbol(name),family_options,intercept,f,names,settings)
end
function fit(X::AbstractMatrix,y::AbstractVector;family=:gaussian,kwargs...)
    _statistical_fit(X,y,Symbol(family);kwargs...)
end
function fit(f::StatsModels.FormulaTerm,data;family=:gaussian,kwargs...)
    _statistical_fit(f,nothing,Symbol(family);data,kwargs...)
end
function _distribution_spec(family,link)
    if family isa NegativeBinomial && link isa GLM.LogLink
        return :negative_binomial,(theta=first(Distributions.params(family)),)
    elseif family isa Binomial
        first(Distributions.params(family))==1 || throw(ArgumentError("glm currently supports binary responses only"))
        family=Bernoulli()
    end
    internal=_glm_family(family,link)
    name=internal isa BernoulliLogit ? :bernoulli : internal isa BernoulliProbit ? :probit :
         internal isa GaussianIdentity ? :gaussian : internal isa GaussianLog ? :gaussian_log :
         internal isa PoissonLog ? :poisson : internal isa GammaLog ? :gamma : :gamma_inverse
    name,NamedTuple()
end
function glm(X::AbstractMatrix,y::AbstractVector,family::Distribution,link::GLM.Link;kwargs...)
    name,options=_distribution_spec(family,link)
    _statistical_fit(X,y,name;family_options=options,kwargs...)
end
function glm(f::StatsModels.FormulaTerm,data,family::Distribution,link::GLM.Link;kwargs...)
    name,options=_distribution_spec(family,link)
    _statistical_fit(f,nothing,name;data,family_options=options,kwargs...)
end
function fit(X::AbstractMatrix,y::AbstractVector,family::IRLSFamily;kwargs...)
    name=family isa SmoothQuantile ? :smooth_quantile : family isa Expectile ? :expectile :
        family isa PseudoHuber ? :pseudo_huber : family isa StudentT ? :student_t :
        family isa NegativeBinomialLog ? :negative_binomial : family isa BinomialLogit ? :binomial :
        family isa TweedieLog ? :tweedie : nothing
    name===nothing && return _fit_julia(X,y,family;kwargs...)
    options=name===:smooth_quantile ? (tau=family.tau,smoothing=family.epsilon) :
        name===:expectile ? (tau=family.tau,) : name===:pseudo_huber ? (delta=family.delta,) :
        name===:student_t ? (nu=family.nu,sigma=family.sigma,) :
        name===:negative_binomial ? (theta=family.theta,) : name===:binomial ? (trials=family.n,) : (power=family.p,)
    _statistical_fit(X,y,name;family_options=options,kwargs...)
end
function StatsAPI.predict(m::StatisticalResult,Xnew=nothing;type=:response,trials=nothing)
    type in (:response,:link) || throw(ArgumentError("type must be :response or :link"))
    Xnew===nothing && return copy(type===:response ? m.fittedvalues : m.linear_predictor)
    if !(Xnew isa AbstractMatrix)
        m.formula===nothing && throw(ArgumentError("table prediction requires a formula model"))
        Xnew=StatsModels.modelcols(m.formula.rhs,Xnew)
        m.intercept && (Xnew=Xnew[:,2:end])
    end
    beta=coef(m)
    size(Xnew,2)==length(beta)-m.intercept || throw(DimensionMismatch("prediction feature count differs from training"))
    if type===:link
        return m.intercept ? Xnew*view(beta,2:length(beta)).+beta[1] : Xnew*beta
    end
    if m.model isa SpectralGLM
        trials===nothing || throw(ArgumentError("trials only applies to grouped binomial models"))
        return predict(m.model,Xnew)
    end
    predict(m.model,Xnew;trials)
end
function Base.show(io::IO,::MIME"text/plain",m::StatisticalResult)
    println(io,m.link===nothing ? "SpectralMM Regression Model" : "SpectralMM Generalized Linear Model")
    println(io,"Family: ",m.family,"   Link: ",something(m.link,"not applicable"))
    println(io,"Observations: ",nobs(m),"   Parameters: ",length(coef(m)))
    println(io,"Coefficients:")
    _show_inference(io,m)
    if m.inference.status!="ok"
        println(io)
        for j in 1:min(20,length(coef(m))); @printf(io,"%-24s %12.6g\n",m.coefficient_names[j],coef(m)[j]); end
    end
    m.deviance_value===nothing || println(io,"\nDeviance: ",m.deviance_value,"   Log-Likelihood: ",m.likelihood_value)
    d=diagnostics(m)
    println(io,"\nSpectralMM optimization:")
    print(io,"  Backend: ",m.backend===:julia ? "Julia" : "C++","   Solver: ",uppercase(string(d.solver)),
          "   Rank: ",d.rank,"   Converged: ",d.converged,"   Iterations: ",d.outer_iterations)
end
Base.show(io::IO,m::StatisticalResult)=show(io,MIME"text/plain"(),m)
