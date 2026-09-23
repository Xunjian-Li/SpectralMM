using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using SpectralIRLS
using BenchmarkTools,DataFrames,GLM,LinearAlgebra,Random,Statistics
import Distributions

glm_spec(::SpectralIRLS.BernoulliLogit)=(Distributions.Bernoulli(),LogitLink())
glm_spec(::SpectralIRLS.BernoulliProbit)=(Distributions.Bernoulli(),ProbitLink())
glm_spec(::SpectralIRLS.GammaLog)=(Distributions.Gamma(),LogLink())
glm_spec(::SpectralIRLS.GammaInverse)=(Distributions.Gamma(),InverseLink())
glm_spec(::SpectralIRLS.GaussianIdentity)=(Distributions.Normal(),IdentityLink())
glm_spec(::SpectralIRLS.GaussianLog)=(Distributions.Normal(),LogLink())
glm_spec(::SpectralIRLS.PoissonLog)=(Distributions.Poisson(),LogLink())
glm_spec(f::SpectralIRLS.NegativeBinomialLog)=(Distributions.NegativeBinomial(f.theta),LogLink())

glm_supported(::SpectralIRLS.IRLSFamily)=false
glm_supported(::SpectralIRLS.GLMFamily)=true
glm_supported(::SpectralIRLS.TweedieLog)=false
glm_supported(::SpectralIRLS.BinomialLogit)=false

function evaluate_solution(X,y,beta,family;ridge=0.0,w_floor=1e-12,grad_scale=1.0)
    T=eltype(X); m,p=size(X)
    eta=X*beta; mu=zeros(T,m); w=zeros(T,m); r=zeros(T,m); g=zeros(T,p)
    loss=logloss_xb(family,eta,y,beta;ridge=T(ridge))
    grad_weights_xb!(family,g,mu,w,r,X,eta,y,beta;ridge=T(ridge),w_floor=T(w_floor))
    gn=norm(g)
    return loss,gn,gn/grad_scale
end

function initial_grad_scale(X,y,family;ridge=0.0,w_floor=1e-12)
    T=eltype(X); m,p=size(X)
    beta=zeros(T,p); eta=zeros(T,m); mu=zeros(T,m); w=zeros(T,m); r=zeros(T,m); g=zeros(T,p)
    grad_weights_xb!(family,g,mu,w,r,X,eta,y,beta;ridge=T(ridge),w_floor=T(w_floor))
    return one(T)+norm(g)
end

function bench_time(f;samples=5)
    b=@benchmarkable $f() evals=1
    b.params.samples=samples
    return median(run(b)).time/1e6
end

function benchmark_irls_methods(
    X,y,family::IRLSFamily;
    spectral,inner,outer,ridge=0.0,maxiter=1000,gtol=1e-8,relgtol=1e-8,
    outer_nesterov=true,samples=5
)
    T=eltype(X); grad_scale=initial_grad_scale(X,y,family;ridge=ridge); results=NamedTuple[]

    f=()->irls_cholesky(X,y;family=family,ridge=T(ridge),maxiter=maxiter,
        gtol=T(gtol),relgtol=T(relgtol),outer_nesterov=outer_nesterov,verbose=false)
    fit=f(); tm=bench_time(f;samples=samples)
    loss,gn,rgn=evaluate_solution(X,y,fit.beta,family;ridge=ridge,grad_scale=grad_scale)
    push!(results,(Method="Cholesky",Time_ms=tm,Iter=fit.iters,Loss=loss,GradNorm=gn,RelGradNorm=rgn,Converged=fit.converged))

    f=()->spectral_mm(X,y;family=family,spectral=spectral,inner=inner,outer=outer)
    fit=f(); tm=bench_time(f;samples=samples)
    loss,gn,rgn=evaluate_solution(X,y,fit.beta,family;ridge=ridge,grad_scale=grad_scale)
    push!(results,(Method="Spectral-MM",Time_ms=tm,Iter=fit.iters,Loss=loss,GradNorm=gn,RelGradNorm=rgn,Converged=fit.converged))

    for solver in (:cg,:cgls,:crls,:lsqr,:lsmr)
        f=()->irls_krylov(X,y;family=family,solver=solver,ridge=T(ridge),maxiter=maxiter,
            gtol=T(gtol),relgtol=T(relgtol),outer_nesterov=outer_nesterov,verbose=false)
        fit=f(); tm=bench_time(f;samples=samples)
        loss,gn,rgn=evaluate_solution(X,y,fit.beta,family;ridge=ridge,grad_scale=grad_scale)
        push!(results,(Method=uppercase(string(solver)),Time_ms=tm,Iter=fit.iters,Loss=loss,GradNorm=gn,RelGradNorm=rgn,Converged=fit.converged))
    end

    if glm_supported(family) && ridge==0
        D,L=glm_spec(family)
        f=()->glm(X,y,D,L;maxiter=maxiter,atol=gtol,rtol=relgtol)
        fit=f(); tm=bench_time(f;samples=samples)
        loss,gn,rgn=evaluate_solution(X,y,coef(fit),family;ridge=0.0,grad_scale=grad_scale)
        push!(results,(Method="GLM.jl",Time_ms=tm,Iter=missing,Loss=loss,GradNorm=gn,RelGradNorm=rgn,Converged=true))
    end

    df=DataFrame(results)
    df.LossGap=df.Loss.-minimum(df.Loss)
    sort!(df,:Time_ms)
    return df
end

seed=3124
p=5000; m=2p; T=Float64
beta_true=randn(MersenneTwister(seed),T,p)./sqrt(T(p))
beta0_true=T(0.5)
X=generate_correlated_design(m,p;rho=0.5,seed=1,T=T)

# Change only this line to run the four new examples.
family=SmoothQuantile(0.25,0.1)
# family=Expectile(0.25)
# family=PseudoHuber(1.345)
# family=StudentT(4.0,1.0)

y,mu,eta=simulate_y_from_beta(X,beta_true,family;seed=seed,beta0=beta0_true)
Xfit=hcat(ones(T,m),X)

spectral=SpectralOptions{T}(k=5,krylovdim=12,ridge=0.0,resid_tol=5e-1,correction_tol=5e-2)
inner=InnerOptions{T}(solver=:pcg,maxiter=200,eta_max=0.6,forcing_c=10.0,forcing_alpha=0.0,abstol=1e-5,nesterov=true)
outer=OuterOptions{T}(maxiter=200,gtol=1e-6,relgtol=1e-8,nesterov=true,verbose=false)

results=benchmark_irls_methods(Xfit,y,family;spectral=spectral,inner=inner,outer=outer,
    ridge=0.0,maxiter=1000,gtol=1e-6,relgtol=1e-8,outer_nesterov=true,samples=5)
show(results,allrows=true,allcols=true)
