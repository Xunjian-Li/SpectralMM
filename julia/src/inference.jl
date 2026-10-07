# Full post-fit statistical curvature, separate from the matrix-free optimizer.
function _weighted_information(X, w)
    n,p=size(X)
    if X isa SparseMatrixCSC
        return Matrix(Symmetric(transpose(X)*(Diagonal(w)*X)))
    end
    H=zeros(Float64,p,p)
    for first in 1:256:n
        rows=first:min(n,first+255)
        block=@view X[rows,:]
        H .+= transpose(block)*(w[rows].*block)
    end
    return Matrix(Symmetric(H))
end
function _julia_inference(X,y,beta,family,result; inference=:auto,inference_max_p=50,
                          cov_type=:auto,level=.95,use_t=nothing,dispersion=nothing,
                          ridge=0.,gtol=1e-7,relgtol=1e-8)
    n,p=size(X)
    gate=_inference_gate(inference,inference_max_p,p,n,ridge)
    gate!==nothing && return _finish_inference(gate,inference)
    residual=family isa Union{AsymmetricFamily,RobustFamily}
    kind=cov_type===:auto ? (residual ? :sandwich : :model) : cov_type
    residual && kind!==:sandwich && return _finish_inference(_inference_unavailable("residual models require sandwich covariance"),inference)
    kind===:sandwich && dispersion!==nothing && return _finish_inference(_inference_unavailable("dispersion is not used by sandwich covariance"),inference)
    eta=X*beta
    g=zeros(eltype(beta),p); mu=zeros(eltype(beta),n); w=similar(mu); score=similar(mu)
    grad_weights_xb!(family,g,mu,w,score,X,eta,y,beta;ridge=zero(eltype(beta)))
    gradscale=1+first(result.gradnorms)
    if !(norm(g)<=gtol || norm(g)/gradscale<=relgtol)
        return _finish_inference(_inference_unavailable("fit did not satisfy the gradient convergence criterion"),inference)
    end
    estimated_family=family isa Union{GaussianIdentity,GaussianLog,GammaLog,GammaInverse,TweedieLog}
    pearson=0.
    if !residual
        for i in eachindex(y)
            v=var_mu(family,mu[i],i)
            if !(isfinite(v) && v>0)
                return _finish_inference(_inference_unavailable("degenerate fitted variance"),inference)
            end
            derivative=dmean_eta(family,eta[i],mu[i],i)
            w[i]=family isa BernoulliProbit ? probit_information(eta[i]) : derivative^2/v
            if kind===:sandwich
                if family isa BernoulliProbit
                    h=abs(score[i]); z=y[i]==one(y[i]) ? eta[i] : -eta[i]
                    w[i]=h*(h+z)
                elseif family isa GammaLog
                    w[i]=y[i]/mu[i]
                elseif family isa NegativeBinomialLog
                    t=family.theta; w[i]=t*mu[i]*(t+y[i])/(t+mu[i])^2
                elseif family isa GaussianLog
                    w[i]=mu[i]*(2mu[i]-y[i])
                elseif family isa TweedieLog
                    a=family.p; w[i]=(2-a)*mu[i]^(2-a)+(a-1)*y[i]*mu[i]^(1-a)
                end
            end
            estimated_family && (pearson+=(y[i]-mu[i])^2/v)
        end
        if family isa Union{BernoulliLogit,BernoulliProbit,BinomialLogit}
            trials=family isa BinomialLogit ? [at(family.n,i) for i in eachindex(y)] : ones(n)
            margin=ifelse.(y.==0,-eta,eta)
            if all((y.==0).|(y.==trials)) && all(margin.>=0) && any(margin.>0)
                return _finish_inference(_inference_unavailable("separation detected; ordinary Wald inference is not reliable"),inference)
            end
        end
    else
        for i in eachindex(y)
            r=y[i]-eta[i]
            if family isa SmoothQuantile
                a=family.epsilon; w[i]=a^2/(2*(r*r+a*a)^1.5)
            elseif family isa Expectile
                w[i]=r>=0 ? family.tau : 1-family.tau
            elseif family isa PseudoHuber
                w[i]=(1+(r/family.delta)^2)^(-1.5)
            else
                c=family.nu*family.sigma^2
                w[i]=(family.nu+1)*(c-r*r)/(c+r*r)^2
            end
        end
    end
    if !all(isfinite,w) || !all(isfinite,score)
        return _finish_inference(_inference_unavailable("non-finite inference weights"),inference)
    end
    H=_weighted_information(X,w)
    if !all(diag(H).>0)
        return _finish_inference(_inference_unavailable("information matrix is not positive definite"),inference)
    end
    scale=1 ./ sqrt.(diag(H)); scaled=scale.*H.*transpose(scale)
    factor=cholesky(Symmetric(scaled);check=false)
    if !issuccess(factor)
        return _finish_inference(_inference_unavailable("information matrix is singular or not positive definite"),inference)
    end
    inverse_scaled=factor\Matrix{Float64}(I,p,p)
    rcond=1/(opnorm(scaled,1)*opnorm(inverse_scaled,1))
    if !(isfinite(rcond) && rcond>1e-12)
        return _finish_inference(_inference_unavailable("information matrix is singular or ill-conditioned"),inference)
    end
    inverse=scale.*inverse_scaled.*transpose(scale)
    df=Float64(n-p)
    estimated=kind===:model && dispersion===nothing && estimated_family
    phi=kind===:sandwich ? 1. : (dispersion===nothing ? (estimated ? pearson/df : 1.) : Float64(dispersion))
    if !(isfinite(phi) && phi>0)
        return _finish_inference(_inference_unavailable("estimated dispersion is zero or non-finite"),inference)
    end
    covariance=kind===:model ? phi*inverse : (n/df)*inverse*_weighted_information(X,score.^2)*inverse
    covariance=Matrix(Symmetric((covariance+transpose(covariance))/2))
    if !all(isfinite,covariance) || !all(diag(covariance).>0)
        return _finish_inference(_inference_unavailable("degenerate covariance matrix"),inference)
    end
    return _wald_result(beta,covariance,kind,estimated,df,phi,rcond,level,use_t)
end
