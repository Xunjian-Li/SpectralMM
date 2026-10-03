# Unified matrix interface. Legacy positional-family and glm methods stay Julia-only.
function _named_family(name::Symbol, options, n::Int)
    name in _CppBackend.FAMILIES || throw(ArgumentError("unsupported family: $name"))
    params = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in pairs(options))
    allowed = get(_CppBackend.FAMILY_PARAMS, name, ())
    all(k -> k in allowed, keys(params)) || throw(ArgumentError("unknown parameter for $name"))
    for (key,value) in params
        key === :trials && continue
        value isa Real && isfinite(value) || throw(ArgumentError("$key must be a finite real number"))
    end
    if name === :negative_binomial
        theta=get(params,:theta,1.); theta>0 || throw(ArgumentError("theta must be positive"))
        return NegativeBinomialLog(theta), nothing, params
    elseif name === :tweedie
        power=get(params,:power,1.5); 1<power<2 || throw(ArgumentError("power must lie in (1,2)"))
        return TweedieLog(power), nothing, params
    elseif name === :binomial
        trials=get(params,:trials,1.)
        trials isa Real || trials isa AbstractVector{<:Real} || throw(ArgumentError("trials must be a real scalar or vector"))
        if trials isa AbstractVector
            length(trials) in (1,n) || throw(DimensionMismatch("trials must be scalar or length n"))
            trials=length(trials)==1 ? Float64(only(trials)) : Float64.(trials)
        end
        all(t -> isfinite(t) && t>0, trials isa Real ? (trials,) : trials) || throw(ArgumentError("trials must be finite and positive"))
        params[:trials]=trials
        return BinomialLogit(trials), nothing, params
    elseif name === :smooth_quantile
        return SmoothQuantile(get(params,:tau,.5),get(params,:smoothing,.1)), nothing, params
    elseif name === :expectile
        return Expectile(get(params,:tau,.5)), nothing, params
    elseif name === :pseudo_huber
        return PseudoHuber(get(params,:delta,1.0)), nothing, params
    elseif name === :student_t
        return StudentT(get(params,:nu,4.),get(params,:sigma,1.)), nothing, params
    end
    family,link = name === :gaussian ? (Normal(),GLM.IdentityLink()) :
        name === :bernoulli ? (Bernoulli(),GLM.LogitLink()) :
        name === :probit ? (Bernoulli(),GLM.ProbitLink()) :
        name === :poisson ? (Poisson(),GLM.LogLink()) :
        name === :gamma ? (Gamma(),GLM.LogLink()) :
        name === :gamma_inverse ? (Gamma(),GLM.InverseLink()) : (Normal(),GLM.LogLink())
    return family,link,params
end

"""
    fit(X, y; family=:gaussian, backend=:julia, family_options=NamedTuple(), kwargs...)

Fit a GLM or residual model using pure Julia (default) or the shared C++ core
(`backend=:cpp`). Both backends support fourteen named families and eight solvers.
Matrix input contains features; `intercept=true` adds the intercept. Family-specific
parameters belong in `family_options`. Standard accessors are `coef`, `predict`,
`stderror`, `vcov` and `confint`; `model.backend` identifies the implementation.
C++ is built and cached only when that backend is first requested.
Legacy `glm` and positional-family `fit` calls remain pure Julia.
"""
function _fit_backend(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
             family=:gaussian, backend=:julia, family_options=NamedTuple(),
             intercept::Bool=true, rank=nothing, maxiter=200, inner_maxiter=100,
             gtol=1e-7, relgtol=1e-8, floor=1e-6, solver=:pcg, kwargs...)
    backend in (:julia,:cpp) || throw(ArgumentError("backend must be :julia or :cpp"))
    name=Symbol(family)
    f,link,params=_named_family(name,family_options,size(X,1))
    p=size(X,2)+intercept
    selected_rank=isnothing(rank) ? _default_rank(p) : rank
    common=(;intercept,rank=selected_rank,maxiter,inner_maxiter,gtol,relgtol,floor,solver)
    if backend === :cpp
        return _CppBackend.fit(X,y;family=name,family_options=params,common...,kwargs...)
    end
    # Common matrix interface uses Float64, preserving CSC storage and avoiding
    # copies for already-compatible dense arrays.
    xx=X isa SparseMatrixCSC ? convert(SparseMatrixCSC{Float64,Int},X) :
        X isa Matrix{Float64} ? X : Matrix{Float64}(X)
    yy=y isa Vector{Float64} ? y : Vector{Float64}(y)
    return link === nothing ? _fit_julia(xx,yy,f;common...,kwargs...) :
        _glm_julia(xx,yy,f,link;common...,kwargs...)
end

# Backend is a read-only property without changing legacy model constructors.
function Base.getproperty(m::Union{SpectralGLM,SpectralModel}, name::Symbol)
    name === :backend && return :julia
    return getfield(m,name)
end
Base.propertynames(m::Union{SpectralGLM,SpectralModel}, private::Bool=false) =
    (fieldnames(typeof(m))..., :backend)
