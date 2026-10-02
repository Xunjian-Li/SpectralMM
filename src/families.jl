# Low-level design-matrix APIs penalize all coefficients by default.
# High-level interfaces explicitly exclude the leading intercept coefficient.
penalty_norm2(b, penalize_intercept::Bool) = sum(abs2, view(b, (penalize_intercept ? 1 : 2):length(b)))
function add_penalty!(out, b, ridge, penalize_intercept::Bool)
    @inbounds @simd for j in (penalize_intercept ? 1 : 2):length(b)
        out[j] += ridge*b[j]
    end
    return out
end


## Helper functions

abstract type IRLSFamily end
abstract type GLMFamily <: IRLSFamily end
abstract type AsymmetricFamily <: IRLSFamily end
abstract type RobustFamily <: IRLSFamily end

struct BernoulliLogit <: GLMFamily end
struct BernoulliProbit <: GLMFamily end
struct BinomialLogit{N} <: GLMFamily
    n::N
end
struct PoissonLog <: GLMFamily end
struct NegativeBinomialLog{T <: Real} <: GLMFamily
    theta::T
end
struct GaussianIdentity <: GLMFamily end
struct GaussianLog <: GLMFamily end
struct GammaLog <: GLMFamily end
struct GammaInverse <: GLMFamily end
struct TweedieLog{T <: Real} <: GLMFamily
    p::T
end

BinomialLogit() = BinomialLogit(1.0)
NegativeBinomialLog() = NegativeBinomialLog(1.0)
TweedieLog() = TweedieLog(1.5)

struct SmoothQuantile{T<:Real} <: AsymmetricFamily
    tau::T
    epsilon::T
end

function SmoothQuantile(tau=0.5, epsilon=0.1)
    τ, ε = promote(float(tau), float(epsilon))
    zero(τ) < τ < one(τ) || throw(ArgumentError("tau must lie in (0,1)"))
    ε > zero(ε) || throw(ArgumentError("epsilon must be positive"))
    return SmoothQuantile{typeof(τ)}(τ, ε)
end

struct Expectile{T<:Real} <: AsymmetricFamily
    tau::T
end

function Expectile(tau=0.5)
    τ = float(tau)
    zero(τ) < τ < one(τ) || throw(ArgumentError("tau must lie in (0,1)"))
    return Expectile{typeof(τ)}(τ)
end

struct PseudoHuber{T<:Real} <: RobustFamily
    delta::T
end

function PseudoHuber(delta=1.345)
    δ = float(delta)
    δ > zero(δ) || throw(ArgumentError("delta must be positive"))
    return PseudoHuber{typeof(δ)}(δ)
end

struct StudentT{T<:Real} <: RobustFamily
    nu::T
    sigma::T
end

function StudentT(nu=4.0, sigma=1.0)
    ν, σ = promote(float(nu), float(sigma))
    ν > zero(ν) || throw(ArgumentError("nu must be positive"))
    σ > zero(σ) || throw(ArgumentError("sigma must be positive"))
    return StudentT{typeof(ν)}(ν, σ)
end

# GLM primitives for IRLS: η = Xβ, W = (dμ/dη)^2 / Var(μ), z = η + (y - μ)/(dμ/dη).

@inline function sigmoid_stable(x::T) where {T <: Real}
    if x >= zero(T)
        z = exp(-x)
        return inv(one(T) + z)
    else
        z = exp(x)
        return z / (one(T) + z)
    end
end

# The objective and its derivatives must use the same normal distribution.
@inline stdnorm_pdf(x::T) where {T<:Real} = exp(-T(0.5)*x^2)/sqrt(T(2)*T(pi))
@inline stdnorm_cdf(x::T) where {T<:Real} = Distributions.cdf(Distributions.Normal(zero(T),one(T)),x)
@inline stdnorm_logcdf(x::T) where {T<:Real} = Distributions.logcdf(Distributions.Normal(zero(T),one(T)),x)
@inline stdnorm_logpdf(x::T) where {T<:Real} = -x*x/T(2)-log(T(2)*T(pi))/T(2)
@inline probit_score(e::T,y) where {T<:Real} = begin
    z=y==one(y) ? e : -e
    h=exp(stdnorm_logpdf(e)-stdnorm_logcdf(z))
    y==one(y) ? -h : h
end
@inline probit_information(e::T) where {T<:Real} =
    exp(T(2)*stdnorm_logpdf(e)-stdnorm_logcdf(e)-stdnorm_logcdf(-e))

# Allow only rounding-scale objective increases; gradient tolerances are unchanged.
@inline objective_roundoff(f::T) where {T<:AbstractFloat} = T(8)*eps(T)*max(one(T),abs(f))
@inline clamp_prob(x::T) where {T <: Real} = min(one(T) - eps(T), max(eps(T), x))
@inline poisson_mean(η::T) where {T <: Real} = exp(min(η, T(700)))
@inline positive_eta(η::T) where {T <: Real} = max(η, sqrt(eps(T)))
@inline at(x, i) = x isa AbstractVector ? x[i] : x
@inline function finite_all(x)
    @inbounds for xi in x
        isfinite(xi) || return false
    end
    return true
end
@inline safe_dη(dη::T, floor::T) where {T <: Real} =
    abs(dη) >= floor ? dη : ifelse(dη < zero(T), -floor, floor)
all_positive(x) = x isa AbstractVector ? all(x .> zero(eltype(x))) : x > 0

check_y(::BernoulliLogit, y) =
    @assert all((y .== zero(eltype(y))) .| (y .== one(eltype(y)))) "BernoulliLogit requires y in {0,1}."
check_y(::BernoulliProbit, y) =
    @assert all((y .== zero(eltype(y))) .| (y .== one(eltype(y)))) "BernoulliProbit requires y in {0,1}."
function check_y(f::BinomialLogit, y)
    @assert all_positive(f.n) "BinomialLogit requires n > 0."
    @assert all((y .>= zero(eltype(y))) .& (y .<= f.n)) "BinomialLogit requires 0 <= y <= n."
end
check_y(::PoissonLog, y) = @assert all(y .>= zero(eltype(y))) "PoissonLog requires y >= 0."
function check_y(f::NegativeBinomialLog, y)
    @assert f.theta > 0 "NegativeBinomialLog requires theta > 0."
    @assert all(y .>= zero(eltype(y))) "NegativeBinomialLog requires y >= 0."
end
check_y(::GaussianIdentity, y) = nothing
check_y(::GaussianLog, y) = @assert all(y .> zero(eltype(y))) "GaussianLog requires y > 0."
check_y(::GammaLog, y) = @assert all(y .> zero(eltype(y))) "GammaLog requires y > 0."
check_y(::GammaInverse, y) = @assert all(y .> zero(eltype(y))) "GammaInverse requires y > 0."
function check_y(f::TweedieLog, y)
    @assert one(f.p) <= f.p <= 2 * one(f.p) "TweedieLog expects 1 <= p <= 2."
    @assert all(y .>= zero(eltype(y))) "TweedieLog requires y >= 0."
end

@inline mean_eta(::BernoulliLogit, η::T, i::Int = 1) where {T <: Real} = sigmoid_stable(η)
@inline dmean_eta(::BernoulliLogit, η::T, μ::T, i::Int = 1) where {T <: Real} = μ * (one(T) - μ)
@inline var_mu(::BernoulliLogit, μ::T, i::Int = 1) where {T <: Real} = μ * (one(T) - μ)

@inline mean_eta(::BernoulliProbit, η::T, i::Int = 1) where {T <: Real} = clamp_prob(stdnorm_cdf(η))
@inline dmean_eta(::BernoulliProbit, η::T, μ::T, i::Int = 1) where {T <: Real} = max(stdnorm_pdf(η), eps(T))
@inline var_mu(::BernoulliProbit, μ::T, i::Int = 1) where {T <: Real} = μ * (one(T) - μ)

@inline function mean_eta(f::BinomialLogit, η::T, i::Int = 1) where {T <: Real}
    return T(at(f.n, i)) * sigmoid_stable(η)
end

@inline function dmean_eta(f::BinomialLogit, η::T, μ::T, i::Int = 1) where {T <: Real}
    n = T(at(f.n, i))
    p = clamp_prob(μ / n)
    return n * p * (one(T) - p)
end

@inline function var_mu(f::BinomialLogit, μ::T, i::Int = 1) where {T <: Real}
    n = T(at(f.n, i))
    p = clamp_prob(μ / n)
    return n * p * (one(T) - p)
end

@inline mean_eta(::PoissonLog, η::T, i::Int = 1) where {T <: Real} = poisson_mean(η)
@inline dmean_eta(::PoissonLog, η::T, μ::T, i::Int = 1) where {T <: Real} = μ
@inline var_mu(::PoissonLog, μ::T, i::Int = 1) where {T <: Real} = μ

@inline mean_eta(::GaussianIdentity, η::T, i::Int = 1) where {T <: Real} = η
@inline dmean_eta(::GaussianIdentity, η::T, μ::T, i::Int = 1) where {T <: Real} = one(T)
@inline var_mu(::GaussianIdentity, μ::T, i::Int = 1) where {T <: Real} = one(T)

@inline mean_eta(::GaussianLog, η::T, i::Int = 1) where {T <: Real} = poisson_mean(η)
@inline dmean_eta(::GaussianLog, η::T, μ::T, i::Int = 1) where {T <: Real} = μ
@inline var_mu(::GaussianLog, μ::T, i::Int = 1) where {T <: Real} = one(T)

@inline mean_eta(::GammaLog, η::T, i::Int = 1) where {T <: Real} = poisson_mean(η)
@inline dmean_eta(::GammaLog, η::T, μ::T, i::Int = 1) where {T <: Real} = μ
@inline var_mu(::GammaLog, μ::T, i::Int = 1) where {T <: Real} = μ^2

@inline mean_eta(::GammaInverse, η::T, i::Int = 1) where {T <: Real} = inv(positive_eta(η))
@inline dmean_eta(::GammaInverse, η::T, μ::T, i::Int = 1) where {T <: Real} = -μ^2
@inline var_mu(::GammaInverse, μ::T, i::Int = 1) where {T <: Real} = μ^2

@inline mean_eta(f::NegativeBinomialLog, η::T, i::Int = 1) where {T <: Real} = poisson_mean(η)
@inline dmean_eta(f::NegativeBinomialLog, η::T, μ::T, i::Int = 1) where {T <: Real} = μ
@inline var_mu(f::NegativeBinomialLog, μ::T, i::Int = 1) where {T <: Real} = μ + μ^2 / T(f.theta)

@inline mean_eta(f::TweedieLog, η::T, i::Int = 1) where {T <: Real} = poisson_mean(η)
@inline dmean_eta(f::TweedieLog, η::T, μ::T, i::Int = 1) where {T <: Real} = μ
@inline var_mu(f::TweedieLog, μ::T, i::Int = 1) where {T <: Real} = μ^T(f.p)

function logloss_xb(
    ::BernoulliLogit,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds for i in eachindex(y)
        ηi = Xβ[i]

        if ηi > zero(T)
            val += ηi + log1p(exp(-ηi)) - y[i] * ηi
        else
            val += log1p(exp(ηi)) - y[i] * ηi
        end
    end

    if ridge > zero(T)
        val += T(0.5) * ridge * penalty_norm2(β, penalize_intercept)
    end

    return val
end

function logloss_xb(
    ::BernoulliProbit,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds for i in eachindex(y)
        val -= stdnorm_logcdf(y[i]==one(T) ? Xβ[i] : -Xβ[i])
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    f::BinomialLogit,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds for i in eachindex(y)
        n = T(at(f.n, i))
        ηi = Xβ[i]

        if ηi > zero(T)
            val += n * (ηi + log1p(exp(-ηi))) - y[i] * ηi
        else
            val += n * log1p(exp(ηi)) - y[i] * ηi
        end
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    ::PoissonLog,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds for i in eachindex(y)
        val += poisson_mean(Xβ[i]) - y[i] * Xβ[i]
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    f::NegativeBinomialLog,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    θ = T(f.theta)
    val = zero(T)

    @inbounds for i in eachindex(y)
        μi = poisson_mean(Xβ[i])
        val += (y[i] + θ) * log(θ + μi) - y[i] * Xβ[i]
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    ::GaussianIdentity,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds @simd for i in eachindex(y)
        ri = Xβ[i] - y[i]
        val += T(0.5) * ri^2
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    ::GaussianLog,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds @simd for i in eachindex(y)
        ri = poisson_mean(Xβ[i]) - y[i]
        val += T(0.5) * ri^2
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    ::Union{GammaLog, GammaInverse},
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds for i in eachindex(y)
        μi = mean_eta(GammaLog(), Xβ[i])
        val += y[i] / μi + log(μi)
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    f::GammaInverse,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    val = zero(T)

    @inbounds for i in eachindex(y)
        ηi = positive_eta(Xβ[i])
        val += y[i] * ηi - log(ηi)
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

function logloss_xb(
    f::TweedieLog,
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    p = T(f.p)
    val = zero(T)

    @inbounds for i in eachindex(y)
        μi = poisson_mean(Xβ[i])

        if abs(p - one(T)) <= sqrt(eps(T))
            val += μi - y[i] * Xβ[i]
        elseif abs(p - T(2)) <= sqrt(eps(T))
            val += y[i] / μi + log(μi)
        else
            val += μi^(T(2) - p) / (T(2) - p) - y[i] * μi^(one(T) - p) / (one(T) - p)
        end
    end

    return ridge > zero(T) ? val + T(0.5) * ridge * penalty_norm2(β, penalize_intercept) : val
end

logloss_xb(Xβ, y, β; ridge = zero(eltype(Xβ)), penalize_intercept::Bool=true) =
    logloss_xb(BernoulliLogit(), Xβ, y, β; ridge=ridge, penalize_intercept=penalize_intercept)

function weights_xb!(
    family::GLMFamily,
    μ::AbstractVector{T},
    w::AbstractVector{T},
    Xβ::AbstractVector{T};
    w_floor::T = T(1e-12)
) where {T <: Real}

    @inbounds for i in eachindex(Xβ)
        μi = mean_eta(family, Xβ[i], i)
        dμi = dmean_eta(family, Xβ[i], μi, i)
        vi = max(var_mu(family, μi, i), eps(T))
        μ[i] = μi
        w[i] = max(dμi^2 / vi, w_floor)
    end

    return μ, w
end

function weights_xb!(::BernoulliProbit, mu::AbstractVector{T}, w::AbstractVector{T},
                     eta::AbstractVector{T}; w_floor::T=T(1e-12)) where {T<:Real}
    for i in eachindex(eta)
        mu[i]=clamp_prob(stdnorm_cdf(eta[i]))
        w[i]=max(probit_information(eta[i]),w_floor)
    end
    return mu,w
end

function grad_weights_xb!(f::BernoulliProbit, g::AbstractVector{T}, mu::AbstractVector{T},
    w::AbstractVector{T}, score::AbstractVector{T}, X::AbstractMatrix{T}, eta::AbstractVector{T},
    y::AbstractVector{T}, beta::AbstractVector{T}; ridge::T=zero(T),
    penalize_intercept::Bool=true, w_floor::T=T(1e-12)) where {T<:Real}
    weights_xb!(f,mu,w,eta;w_floor)
    for i in eachindex(y); score[i]=probit_score(eta[i],y[i]); end
    mul!(g,transpose(X),score)
    ridge>zero(T) && add_penalty!(g,beta,ridge,penalize_intercept)
    return g,mu,w
end

function work_y!(::BernoulliProbit,z::AbstractVector{T},eta::AbstractVector{T},
    y::AbstractVector{T},mu::AbstractVector{T},w::AbstractVector{T};
    w_floor::T=T(1e-12)) where {T<:Real}
    for i in eachindex(y); z[i]=eta[i]-probit_score(eta[i],y[i])/max(w[i],w_floor); end
    return z
end

weights_xb!(μ, w, Xβ; w_floor = eltype(Xβ)(1e-12)) =
    weights_xb!(BernoulliLogit(), μ, w, Xβ; w_floor = w_floor)

function grad_weights_xb!(
    family::GLMFamily,
    g::AbstractVector{T},
    μ::AbstractVector{T},
    w::AbstractVector{T},
    resid::AbstractVector{T},
    X::AbstractMatrix{T},
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true,
    w_floor::T = T(1e-12)
) where {T <: Real}

    weights_xb!(
        family,
        μ,
        w,
        Xβ;
        w_floor = w_floor
    )

    @inbounds @simd for i in eachindex(resid)
        dμi = dmean_eta(family, Xβ[i], μ[i], i)
        vi = max(var_mu(family, μ[i], i), eps(T))
        resid[i] = (μ[i] - y[i]) * dμi / vi
    end

    mul!(g, transpose(X), resid)

    if ridge > zero(T)
        add_penalty!(g, β, ridge, penalize_intercept)
    end

    return g, μ, w
end

grad_weights_xb!(g, μ, w, resid, X, Xβ, y, β; ridge = zero(eltype(Xβ)), penalize_intercept::Bool=true, w_floor = eltype(Xβ)(1e-12)) =
    grad_weights_xb!(BernoulliLogit(), g, μ, w, resid, X, Xβ, y, β; ridge=ridge, penalize_intercept=penalize_intercept, w_floor = w_floor)

function work_y!(
    family::GLMFamily,
    z::AbstractVector{T},
    Xβ::AbstractVector{T},
    y::AbstractVector{T},
    μ::AbstractVector{T},
    w::AbstractVector{T};
    w_floor::T = T(1e-12)
) where {T <: Real}

    @inbounds for i in eachindex(z)
        dμi = safe_dη(dmean_eta(family, Xβ[i], μ[i], i), w_floor)
        z[i] = Xβ[i] + (y[i] - μ[i]) / dμi
    end

    return z
end

work_y!(z, Xβ, y, μ, w; w_floor = eltype(Xβ)(1e-12)) =
    work_y!(BernoulliLogit(), z, Xβ, y, μ, w; w_floor = w_floor)


function wls!(
    r::AbstractVector{T},
    wr::AbstractVector{T},
    g::AbstractVector{T},
    X::AbstractMatrix{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    mul!(r, X, β)
    r .-= z

    @inbounds @simd for i in eachindex(r)
        wr[i] = w[i] * r[i]
    end

    mul!(g, transpose(X), wr)

    if ridge > zero(T)
        add_penalty!(g, β, ridge, penalize_intercept)
    end

    q = T(0.5) * dot(r, wr)

    if ridge > zero(T)
        q += T(0.5) * ridge * penalty_norm2(β, penalize_intercept)
    end

    return q
end

function wls_r!(
    wr::AbstractVector{T},
    g::AbstractVector{T},
    X::AbstractMatrix{T},
    r::AbstractVector{T},
    w::AbstractVector{T},
    β::AbstractVector{T};
    ridge::T = zero(T), penalize_intercept::Bool=true
) where {T <: Real}

    @inbounds @simd for i in eachindex(r)
        wr[i] = w[i] * r[i]
    end

    mul!(g, transpose(X), wr)

    if ridge > zero(T)
        add_penalty!(g, β, ridge, penalize_intercept)
    end

    q = T(0.5) * dot(r, wr)

    if ridge > zero(T)
        q += T(0.5) * ridge * penalty_norm2(β, penalize_intercept)
    end

    return q
end


# ============================================================
# Unified IRLS interface for residual-based families
# ============================================================

check_y(::Union{AsymmetricFamily,RobustFamily}, y) =
    all(isfinite, y) || throw(ArgumentError("response must be finite"))

# Existing GLM Fisher weights do not depend explicitly on y.
function weights_xb!(
    family::GLMFamily,
    mu::AbstractVector{T},
    w::AbstractVector{T},
    eta::AbstractVector{T},
    y::AbstractVector{T};
    w_floor::T=T(1e-12)
) where {T<:Real}
    return weights_xb!(family, mu, w, eta; w_floor=w_floor)
end

function logloss_xb(
    f::SmoothQuantile, eta::AbstractVector{T}, y::AbstractVector{T},
    beta::AbstractVector{T}; ridge::T=zero(T), penalize_intercept::Bool=true
) where {T<:Real}
    tau, eps0, val = T(f.tau), T(f.epsilon), zero(T)
    @inbounds @simd for i in eachindex(y)
        r = y[i] - eta[i]
        val += (tau-T(0.5))*r + T(0.5)*sqrt(r*r + eps0*eps0)
    end
    return ridge > zero(T) ? val + T(0.5)*ridge*penalty_norm2(beta, penalize_intercept) : val
end

function logloss_xb(
    f::Expectile, eta::AbstractVector{T}, y::AbstractVector{T},
    beta::AbstractVector{T}; ridge::T=zero(T), penalize_intercept::Bool=true
) where {T<:Real}
    tau, val = T(f.tau), zero(T)
    @inbounds @simd for i in eachindex(y)
        r = y[i] - eta[i]
        a = r >= zero(T) ? tau : one(T)-tau
        val += T(0.5)*a*r*r
    end
    return ridge > zero(T) ? val + T(0.5)*ridge*penalty_norm2(beta, penalize_intercept) : val
end

function logloss_xb(
    f::PseudoHuber, eta::AbstractVector{T}, y::AbstractVector{T},
    beta::AbstractVector{T}; ridge::T=zero(T), penalize_intercept::Bool=true
) where {T<:Real}
    delta, val = T(f.delta), zero(T)
    @inbounds @simd for i in eachindex(y)
        r = y[i] - eta[i]
        val += delta*delta*(sqrt(one(T)+(r/delta)^2)-one(T))
    end
    return ridge > zero(T) ? val + T(0.5)*ridge*penalty_norm2(beta, penalize_intercept) : val
end

function logloss_xb(
    f::StudentT, eta::AbstractVector{T}, y::AbstractVector{T},
    beta::AbstractVector{T}; ridge::T=zero(T), penalize_intercept::Bool=true
) where {T<:Real}
    nu, sigma, val = T(f.nu), T(f.sigma), zero(T)
    c = nu*sigma*sigma
    @inbounds @simd for i in eachindex(y)
        r = y[i] - eta[i]
        val += T(0.5)*(nu+one(T))*log1p(r*r/c)
    end
    return ridge > zero(T) ? val + T(0.5)*ridge*penalty_norm2(beta, penalize_intercept) : val
end

function weights_xb!(
    f::SmoothQuantile, mu::AbstractVector{T}, w::AbstractVector{T},
    eta::AbstractVector{T}, y::AbstractVector{T}; w_floor::T=T(1e-12)
) where {T<:Real}
    eps0 = T(f.epsilon)
    copyto!(mu, eta)
    @inbounds @simd for i in eachindex(eta)
        r = y[i]-eta[i]
        s = sqrt(r*r+eps0*eps0)
        w[i] = max(eps0*eps0/(T(2)*s^3), w_floor)
    end
    return mu, w
end

function weights_xb!(
    f::Expectile, mu::AbstractVector{T}, w::AbstractVector{T},
    eta::AbstractVector{T}, y::AbstractVector{T}; w_floor::T=T(1e-12)
) where {T<:Real}
    tau = T(f.tau)
    copyto!(mu, eta)
    @inbounds @simd for i in eachindex(eta)
        w[i] = max(y[i]-eta[i] >= zero(T) ? tau : one(T)-tau, w_floor)
    end
    return mu, w
end

function weights_xb!(
    f::PseudoHuber, mu::AbstractVector{T}, w::AbstractVector{T},
    eta::AbstractVector{T}, y::AbstractVector{T}; w_floor::T=T(1e-12)
) where {T<:Real}
    delta = T(f.delta)
    copyto!(mu, eta)
    @inbounds @simd for i in eachindex(eta)
        u = (y[i]-eta[i])/delta
        # Exact positive Newton curvature.
        w[i] = max((one(T)+u*u)^(-T(1.5)), w_floor)
    end
    return mu, w
end

function weights_xb!(
    f::StudentT, mu::AbstractVector{T}, w::AbstractVector{T},
    eta::AbstractVector{T}, y::AbstractVector{T}; w_floor::T=T(1e-12)
) where {T<:Real}
    nu, sigma = T(f.nu), T(f.sigma)
    c = nu*sigma*sigma
    copyto!(mu, eta)
    @inbounds @simd for i in eachindex(eta)
        r = y[i]-eta[i]
        # Positive MM / scale-mixture weight, not exact Hessian curvature.
        w[i] = max((nu+one(T))/(c+r*r), w_floor)
    end
    return mu, w
end

function grad_weights_xb!(
    f::SmoothQuantile, g::AbstractVector{T}, mu::AbstractVector{T},
    w::AbstractVector{T}, q::AbstractVector{T}, X::AbstractMatrix{T},
    eta::AbstractVector{T}, y::AbstractVector{T}, beta::AbstractVector{T};
    ridge::T=zero(T), penalize_intercept::Bool=true, w_floor::T=T(1e-12)
) where {T<:Real}
    weights_xb!(f,mu,w,eta,y;w_floor=w_floor)
    tau, eps0 = T(f.tau), T(f.epsilon)
    @inbounds @simd for i in eachindex(q)
        r = y[i]-eta[i]
        q[i] = -(tau-T(0.5)+r/(T(2)*sqrt(r*r+eps0*eps0)))
    end
    mul!(g,transpose(X),q)
    ridge > zero(T) && add_penalty!(g, beta, ridge, penalize_intercept)
    return g,mu,w
end

function grad_weights_xb!(
    f::Expectile, g::AbstractVector{T}, mu::AbstractVector{T},
    w::AbstractVector{T}, q::AbstractVector{T}, X::AbstractMatrix{T},
    eta::AbstractVector{T}, y::AbstractVector{T}, beta::AbstractVector{T};
    ridge::T=zero(T), penalize_intercept::Bool=true, w_floor::T=T(1e-12)
) where {T<:Real}
    weights_xb!(f,mu,w,eta,y;w_floor=w_floor)
    tau = T(f.tau)
    @inbounds @simd for i in eachindex(q)
        r = y[i]-eta[i]
        a = r >= zero(T) ? tau : one(T)-tau
        q[i] = -a*r
    end
    mul!(g,transpose(X),q)
    ridge > zero(T) && add_penalty!(g, beta, ridge, penalize_intercept)
    return g,mu,w
end

function grad_weights_xb!(
    f::PseudoHuber, g::AbstractVector{T}, mu::AbstractVector{T},
    w::AbstractVector{T}, q::AbstractVector{T}, X::AbstractMatrix{T},
    eta::AbstractVector{T}, y::AbstractVector{T}, beta::AbstractVector{T};
    ridge::T=zero(T), penalize_intercept::Bool=true, w_floor::T=T(1e-12)
) where {T<:Real}
    weights_xb!(f,mu,w,eta,y;w_floor=w_floor)
    delta = T(f.delta)
    @inbounds @simd for i in eachindex(q)
        r = y[i]-eta[i]
        q[i] = -r/sqrt(one(T)+(r/delta)^2)
    end
    mul!(g,transpose(X),q)
    ridge > zero(T) && add_penalty!(g, beta, ridge, penalize_intercept)
    return g,mu,w
end

function grad_weights_xb!(
    f::StudentT, g::AbstractVector{T}, mu::AbstractVector{T},
    w::AbstractVector{T}, q::AbstractVector{T}, X::AbstractMatrix{T},
    eta::AbstractVector{T}, y::AbstractVector{T}, beta::AbstractVector{T};
    ridge::T=zero(T), penalize_intercept::Bool=true, w_floor::T=T(1e-12)
) where {T<:Real}
    weights_xb!(f,mu,w,eta,y;w_floor=w_floor)
    @inbounds @simd for i in eachindex(q)
        q[i] = w[i]*(eta[i]-y[i])
    end
    mul!(g,transpose(X),q)
    ridge > zero(T) && add_penalty!(g, beta, ridge, penalize_intercept)
    return g,mu,w
end

# GLM working response: keep the user's original implementation.
# Residual-based working responses:
function work_y!(
    f::SmoothQuantile, z::AbstractVector{T}, eta::AbstractVector{T},
    y::AbstractVector{T}, mu::AbstractVector{T}, w::AbstractVector{T};
    w_floor::T=T(1e-12)
) where {T<:Real}
    tau, eps0 = T(f.tau), T(f.epsilon)
    @inbounds @simd for i in eachindex(z)
        r = y[i]-eta[i]
        psi = tau-T(0.5)+r/(T(2)*sqrt(r*r+eps0*eps0))
        z[i] = eta[i] + psi/max(w[i],w_floor)
    end
    return z
end

function work_y!(
    ::Expectile, z::AbstractVector{T}, eta::AbstractVector{T},
    y::AbstractVector{T}, mu::AbstractVector{T}, w::AbstractVector{T};
    w_floor::T=T(1e-12)
) where {T<:Real}
    copyto!(z,y)
    return z
end

function work_y!(
    f::PseudoHuber, z::AbstractVector{T}, eta::AbstractVector{T},
    y::AbstractVector{T}, mu::AbstractVector{T}, w::AbstractVector{T};
    w_floor::T=T(1e-12)
) where {T<:Real}
    delta = T(f.delta)
    @inbounds @simd for i in eachindex(z)
        r = y[i]-eta[i]
        psi = r/sqrt(one(T)+(r/delta)^2)
        z[i] = eta[i] + psi/max(w[i],w_floor)
    end
    return z
end

function work_y!(
    ::StudentT, z::AbstractVector{T}, eta::AbstractVector{T},
    y::AbstractVector{T}, mu::AbstractVector{T}, w::AbstractVector{T};
    w_floor::T=T(1e-12)
) where {T<:Real}
    # For the Student-t MM surrogate q_i = w_i(eta_i-y_i), hence z=y.
    copyto!(z,y)
    return z
end

@inline function finite_max(x::AbstractVector{T}) where {T<:Real}
    v=T(-Inf)
    @inbounds for xi in x
        isfinite(xi) && (v=max(v,xi))
    end
    return v
end

@inline function finite_min(x::AbstractVector{T}) where {T<:Real}
    v=T(Inf)
    @inbounds for xi in x
        isfinite(xi) && (v=min(v,xi))
    end
    return v
end

@inline function admissible_eta(family::IRLSFamily, eta::AbstractVector{T}) where {T<:Real}
    if family isa Union{PoissonLog,NegativeBinomialLog,GammaLog,GaussianLog,TweedieLog}
        return finite_all(eta) && finite_max(eta) <= T(50)
    elseif family isa GammaInverse
        return finite_all(eta) && finite_min(eta) > T(1e-8)
    else
        return finite_all(eta)
    end
end
