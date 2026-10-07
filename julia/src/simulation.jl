
function generate_correlated_design(m::Int,p::Int;rho=0.5,seed=1,T=Float64)
    rng=MersenneTwister(seed)
    X=Matrix{T}(undef,m,p)
    s=sqrt(one(T)-T(rho)^2)
    randn!(rng,@view(X[:,1]))
    @inbounds for j in 2:p
        randn!(rng,@view(X[:,j]))
        @views @. X[:,j]=T(rho)*X[:,j-1]+s*X[:,j]
    end
    return X
end

@inline _normal_pdf(x::T) where {T<:Real}=exp(-x*x/T(2))/sqrt(T(2*pi))

function normal_expectile(tau::T;tol::T=T(1e-12)) where {T<:Real}
    lo,hi=T(-10),T(10)
    for _ in 1:200
        e=(lo+hi)/T(2)
        phi=_normal_pdf(e); Phi=stdnorm_cdf(e)
        pos=phi-e*(one(T)-Phi)
        neg=phi+e*Phi
        f=tau*pos-(one(T)-tau)*neg
        f>zero(T) ? (lo=e) : (hi=e)
        hi-lo<=tol*(one(T)+abs(e)) && return (lo+hi)/T(2)
    end
    return (lo+hi)/T(2)
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::BernoulliLogit;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    mu=sigmoid_stable.(eta); y=T.(rand(rng,length(mu)).<mu)
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::BernoulliProbit;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    mu=stdnorm_cdf.(eta); y=T.(rand(rng,length(mu)).<mu)
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::BinomialLogit;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    n=T(f.n); prob=sigmoid_stable.(eta); mu=n.*prob
    y=T[rand(rng,Binomial(round(Int,n),Float64(prob[i]))) for i in eachindex(prob)]
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},::PoissonLog;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta; mu=exp.(clamp.(eta,T(-30),T(30)))
    y=T[rand(rng,Poisson(Float64(v))) for v in mu]
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::NegativeBinomialLog;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta; mu=exp.(clamp.(eta,T(-30),T(30)))
    theta=T(f.theta)
    y=T[rand(rng,NegativeBinomial(Float64(theta),Float64(theta/(theta+v)))) for v in mu]
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},::GaussianIdentity;seed=1,beta0=0.0,sigma=1.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta; mu=copy(eta)
    y=eta.+T(sigma).*randn(rng,T,length(eta))
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},::GaussianLog;seed=1,beta0=0.0,sigma=1.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta; mu=exp.(clamp.(eta,T(-30),T(30)))
    y=mu.+T(sigma).*randn(rng,T,length(eta))
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},::GammaLog;seed=1,beta0=0.0,shape=2.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta; mu=exp.(clamp.(eta,T(-30),T(30)))
    a=T(shape); y=T[rand(rng,Gamma(Float64(a),Float64(v/a))) for v in mu]
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},::GammaInverse;seed=1,beta0=2.0,shape=2.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    minimum(eta)>zero(T) || throw(ArgumentError("GammaInverse simulation requires positive linear predictors"))
    mu=inv.(eta); a=T(shape); y=T[rand(rng,Gamma(Float64(a),Float64(v/a))) for v in mu]
    return y,mu,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::SmoothQuantile;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    q=T(quantile(Normal(),Float64(f.tau)))
    err=randn(rng,T,length(eta)).-q
    return eta.+err,eta,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::Expectile;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    shift=normal_expectile(T(f.tau))
    err=randn(rng,T,length(eta)).-shift
    return eta.+err,eta,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},::PseudoHuber;seed=1,beta0=0.0,df=3.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    err=T.(rand(rng,TDist(Float64(df)),length(eta)))
    return eta.+err,eta,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::StudentT;seed=1,beta0=0.0) where {T}
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta
    err=T(f.sigma).*T.(rand(rng,TDist(Float64(f.nu)),length(eta)))
    return eta.+err,eta,eta
end

function simulate_y_from_beta(X::AbstractMatrix{T},beta::AbstractVector{T},f::TweedieLog;seed=1,beta0=0.0) where {T}
    p=T(f.p); one(T)<p<T(2) || throw(ArgumentError("Tweedie simulation implemented for 1<p<2"))
    rng=MersenneTwister(seed); eta=T(beta0).+X*beta; mu=exp.(clamp.(eta,T(-30),T(30)))
    y=zeros(T,length(mu))
    alpha=(T(2)-p)/(p-one(T))
    @inbounds for i in eachindex(mu)
        lam=mu[i]^(T(2)-p)/(T(2)-p)
        scale=(p-one(T))*mu[i]^(p-one(T))
        n=rand(rng,Poisson(Float64(lam)))
        y[i]=n==0 ? zero(T) : T(rand(rng,Gamma(Float64(n*alpha),Float64(scale))))
    end
    return y,mu,eta
end
