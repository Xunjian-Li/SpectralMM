# ============================================================
# High-level model interface
# ============================================================

struct SpectralGLM{T,F,L,R,TX,TY,TF} <: StatsAPI.RegressionModel
    coef::Vector{T}
    family::F
    link::L
    result::R
    X::TX
    y::TY
    formula::TF
    solver::Symbol
    rank::Int
end

struct SpectralModel{T,F,R,TX,TY} <: StatsAPI.RegressionModel
    coef::Vector{T}
    family::F
    result::R
    X::TX
    y::TY
    solver::Symbol
    rank::Int
end


# ============================================================
# GLM family/link -> internal SpectralMM family
# ============================================================

_glm_family(::Bernoulli, ::GLM.LogitLink) = BernoulliLogit()
_glm_family(::Bernoulli, ::GLM.ProbitLink) = BernoulliProbit()
_glm_family(::Normal, ::GLM.IdentityLink) = GaussianIdentity()
_glm_family(::Normal, ::GLM.LogLink) = GaussianLog()
_glm_family(::Poisson, ::GLM.LogLink) = PoissonLog()
_glm_family(::Gamma, ::GLM.LogLink) = GammaLog()
_glm_family(::Gamma, ::GLM.InverseLink) = GammaInverse()


# ============================================================
# GLM initialization
# ============================================================

@inline function _clamp_mean01(y, ::Type{T}) where {T<:Real}
    δ = sqrt(eps(T))
    return clamp(T(mean(y)), δ, one(T)-δ)
end

_initial_eta(::BernoulliLogit, y, ::Type{T}) where {T<:Real} = begin
    μ = _clamp_mean01(y, T)
    log(μ / (one(T)-μ))
end

_initial_eta(::BernoulliProbit, y, ::Type{T}) where {T<:Real} = begin
    μ = _clamp_mean01(y, T)
    T(quantile(Normal(), μ))
end

_initial_eta(::GaussianIdentity, y, ::Type{T}) where {T<:Real} =
    T(mean(y))

_initial_eta(
    ::Union{GaussianLog,PoissonLog,GammaLog,NegativeBinomialLog,TweedieLog},
    y,
    ::Type{T},
) where {T<:Real} =
    log(max(T(mean(y)), sqrt(eps(T))))

_initial_eta(::GammaInverse, y, ::Type{T}) where {T<:Real} =
    inv(max(T(mean(y)), sqrt(eps(T))))


@inline valid_initial_eta(::IRLSFamily, η) =
    finite_all(η)

@inline valid_initial_eta(::GammaInverse, η) =
    finite_all(η) && finite_min(η) > zero(eltype(η))


function _constant_column(X::AbstractMatrix{T}) where {T<:Real}
    n, p = size(X)
    n == 0 && return nothing

    tol = sqrt(eps(T))

    @inbounds for j in 1:p
        c = X[1, j]
        isfinite(c) && abs(c) > tol || continue

        constant = true
        for i in 2:n
            if !isapprox(X[i, j], c; rtol=tol, atol=tol)
                constant = false
                break
            end
        end

        constant && return j, c
    end

    return nothing
end


function _default_beta0(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    family::GLMFamily,
) where {T<:Real}

    β = zeros(T, size(X, 2))
    η0 = _initial_eta(family, y, T)
    col = _constant_column(X)

    if col !== nothing
        j, c = col
        β[j] = η0 / c
        return β
    end

    # Without an intercept, retain the zero start whenever it is feasible.
    η = zeros(T, size(X, 1))
    valid_initial_eta(family, η) && return β

    throw(ArgumentError(
        "no automatic feasible initialization is available; " *
        "supply beta0 such that the initial linear predictor is admissible"
    ))
end


function _init_beta(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    family::GLMFamily,
    beta0,
) where {T<:Real}

    β = beta0 === nothing ?
        _default_beta0(X, y, family) :
        Vector{T}(beta0)

    length(β) == size(X, 2) ||
        throw(DimensionMismatch(
            "length(beta0) = $(length(β)) does not match size(X, 2) = $(size(X, 2))"
        ))

    all(isfinite, β) ||
        throw(ArgumentError("beta0 must contain only finite values"))

    η = X * β

    valid_initial_eta(family, η) ||
        throw(ArgumentError(
            "beta0 is not admissible for the specified family and link"
        ))

    return β
end


# ============================================================
# Common options
# ============================================================

"""
    _default_rank(p)

Choose the default retained spectral rank.

For `p < 20`, use `rank = p - 1`, so that the full spectrum is
available. Otherwise use `rank = 10`.
"""
function _default_rank(p::Int)
    p >= 2 ||
        throw(ArgumentError(
            "Spectral-MM requires at least two predictor columns"
        ))

    return p < 20 ? p - 1 : 10
end


function _check_options(
    p,
    rank,
    floor,
    ridge,
    solver,
    eta_max,
    correction_tol,
    resid_tol,
    gtol,
    relgtol,
    maxiter,
    inner_maxiter,
)
    1 <= rank < p ||
        throw(ArgumentError(
            "rank must satisfy 1 <= rank < p; got rank=$rank and p=$p"
        ))

    floor > 0 ||
        throw(ArgumentError(
            "floor must be positive; got floor=$floor"
        ))

    ridge >= 0 ||
        throw(ArgumentError(
            "ridge must be nonnegative; got ridge=$ridge"
        ))

    solver in (:mm, :pcg) ||
        throw(ArgumentError(
            "solver must be :mm or :pcg; got $solver"
        ))

    0 <= eta_max < 1 ||
        throw(ArgumentError(
            "eta_max must satisfy 0 <= eta_max < 1; got eta_max=$eta_max"
        ))

    correction_tol >= 0 ||
        throw(ArgumentError(
            "correction_tol must be nonnegative; got correction_tol=$correction_tol"
        ))

    resid_tol > 0 ||
        throw(ArgumentError(
            "resid_tol must be positive; got resid_tol=$resid_tol"
        ))

    gtol > 0 ||
        throw(ArgumentError(
            "gtol must be positive; got gtol=$gtol"
        ))

    relgtol > 0 ||
        throw(ArgumentError(
            "relgtol must be positive; got relgtol=$relgtol"
        ))

    maxiter >= 1 ||
        throw(ArgumentError(
            "maxiter must be at least 1; got maxiter=$maxiter"
        ))

    inner_maxiter >= 1 ||
        throw(ArgumentError(
            "inner_maxiter must be at least 1; got inner_maxiter=$inner_maxiter"
        ))

    return nothing
end


function _solver_options(
    ::Type{T},
    rank,
    floor,
    ridge,
    solver,
    eta_max,
    correction_tol,
    resid_tol,
    gtol,
    relgtol,
    maxiter,
    inner_maxiter,
    verbose,
) where {T<:Real}

    spectral = SpectralOptions{T}(
        k = rank,
        rho = T(floor),
        ridge = T(ridge),
        correction_tol = T(correction_tol),
        resid_tol = T(resid_tol),
    )

    inner = InnerOptions{T}(
        solver = solver,
        maxiter = inner_maxiter,
        eta_max = T(eta_max),
    )

    outer = OuterOptions{T}(
        maxiter = maxiter,
        gtol = T(gtol),
        relgtol = T(relgtol),
        verbose = verbose,
    )

    return spectral, inner, outer
end


# ============================================================
# GLM interface
# ============================================================

"""
    glm(X, y, family, link; kwargs...)

Fit a generalized linear model using the Spectral-MM framework.

# Keyword arguments

- `beta0`: optional initial coefficient vector.
- `rank`: retained spectral rank.
- `floor`: positive spectral floor.
- `ridge`: nonnegative ridge penalty.
- `solver`: inner solver, either `:mm` or `:pcg`.
- `eta_max`: maximum inner forcing parameter.
- `correction_tol`: threshold for adaptive spectral correction.
- `resid_tol`: maximum acceptable spectral residual.
- `gtol`: absolute gradient stopping tolerance.
- `relgtol`: relative gradient stopping tolerance.
- `maxiter`: maximum number of outer iterations.
- `inner_maxiter`: maximum number of inner iterations.
- `verbose`: print iteration information.
"""
function glm(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    family::Distribution,
    link::GLM.Link;
    beta0::Union{Nothing,AbstractVector} = nothing,
    rank::Union{Nothing,Int} = nothing,
    floor::Real = 1e-6,
    ridge::Real = 0.0,
    solver::Symbol = :pcg,
    eta_max::Real = 0.5,
    correction_tol::Real = 0.01,
    resid_tol::Real = 0.5,
    gtol::Real = 1e-7,
    relgtol::Real = 1e-8,
    maxiter::Int = 200,
    inner_maxiter::Int = 100,
    verbose::Bool = false,
    kwargs...
) where {T<:Real}

    n, p = size(X)

    length(y) == n ||
        throw(DimensionMismatch(
            "length(y) = $(length(y)) does not match size(X, 1) = $n"
        ))

    p >= 2 ||
        throw(ArgumentError(
            "Spectral-MM requires at least two predictor columns"
        ))

    fam = _glm_family(family, link)
    β0 = _init_beta(X, y, fam, beta0)
    r = isnothing(rank) ? _default_rank(p) : rank

    _check_options(
        p, r, floor, ridge, solver, eta_max,
        correction_tol, resid_tol, gtol, relgtol,
        maxiter, inner_maxiter,
    )

    spectral, inner, outer = _solver_options(
        T, r, floor, ridge, solver, eta_max,
        correction_tol, resid_tol, gtol, relgtol,
        maxiter, inner_maxiter, verbose,
    )

    result = spectral_mm(
        X, y;
        family = fam,
        beta0 = β0,
        spectral = spectral,
        inner = inner,
        outer = outer,
        kwargs...
    )

    return SpectralGLM(
        result.beta,
        family,
        link,
        result,
        X,
        y,
        nothing,
        solver,
        r,
    )
end


"""
    glm(formula, data, family, link; kwargs...)

Fit a generalized linear model from a StatsModels formula using
the Spectral-MM framework.
"""
function glm(
    formula::StatsModels.FormulaTerm,
    data,
    family::Distribution,
    link::GLM.Link;
    kwargs...
)
    sch = StatsModels.schema(formula, data)
    f = StatsModels.apply_schema(
        formula,
        sch,
        StatsModels.StatisticalModel,
    )

    y, X = StatsModels.modelcols(f, data)

    model = glm(
        X,
        y,
        family,
        link;
        kwargs...
    )

    return SpectralGLM(
        model.coef,
        model.family,
        model.link,
        model.result,
        model.X,
        model.y,
        f,
        model.solver,
        model.rank,
    )
end


# ============================================================
# General iteratively reweighted models
# ============================================================

"""
    fit(X, y, family; kwargs...)

Fit a residual-based statistical model using the Spectral-MM framework.

Supported families include `Expectile`, `SmoothQuantile`,
`PseudoHuber`, and `StudentT`.
"""
function fit(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    family::Union{AsymmetricFamily,RobustFamily};
    rank::Union{Nothing,Int} = nothing,
    floor::Real = 1e-6,
    ridge::Real = 0.0,
    solver::Symbol = :pcg,
    eta_max::Real = 0.5,
    correction_tol::Real = 0.01,
    resid_tol::Real = 0.5,
    gtol::Real = 1e-7,
    relgtol::Real = 1e-8,
    maxiter::Int = 200,
    inner_maxiter::Int = 100,
    verbose::Bool = false,
    kwargs...
) where {T<:Real}

    n, p = size(X)

    length(y) == n ||
        throw(DimensionMismatch(
            "length(y) = $(length(y)) does not match size(X, 1) = $n"
        ))

    p >= 2 ||
        throw(ArgumentError(
            "Spectral-MM requires at least two predictor columns"
        ))

    r = isnothing(rank) ? _default_rank(p) : rank

    _check_options(
        p, r, floor, ridge, solver, eta_max,
        correction_tol, resid_tol, gtol, relgtol,
        maxiter, inner_maxiter,
    )

    spectral, inner, outer = _solver_options(
        T, r, floor, ridge, solver, eta_max,
        correction_tol, resid_tol, gtol, relgtol,
        maxiter, inner_maxiter, verbose,
    )

    result = spectral_mm(
        X, y;
        family = family,
        spectral = spectral,
        inner = inner,
        outer = outer,
        kwargs...
    )

    return SpectralModel(
        result.beta,
        family,
        result,
        X,
        y,
        solver,
        r,
    )
end


# ============================================================
# Display
# ============================================================

function Base.show(io::IO, m::SpectralModel)
    print(
        io,
        "SpectralModel(",
        typeof(m.family).name.name,
        "; solver=:",
        m.solver,
        ", rank=",
        m.rank,
        ", converged=",
        m.result.converged,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::SpectralModel)
    r = m.result

    println(io, "SpectralModel")
    println(io, "────────────────────────────────────────")
    println(io, "Family:             ", typeof(m.family).name.name)
    println(io, "Solver:             ", uppercase(String(m.solver)))
    println(io, "Spectral rank:      ", m.rank)
    println(io, "Observations:       ", StatsAPI.nobs(m))
    println(io, "Parameters:         ", length(m.coef))
    println(io, "Outer iterations:   ", r.iters)

    if !isempty(r.losses)
        println(io, "Final loss:         ", @sprintf("%.6e", r.losses[end]))
    end

    if !isempty(r.relgradnorms)
        println(io, "Relative gradient:  ", @sprintf("%.3e", r.relgradnorms[end]))
    end

    print(io, "Converged:          ", r.converged)
end