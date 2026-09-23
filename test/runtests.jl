using Test,LinearAlgebra,Random
using SpectralIRLS

@testset "family gradients" begin
    rng=MersenneTwister(1); n,p=30,5
    X=randn(rng,n,p); beta=randn(rng,p); y=randn(rng,n)
    for f in (SmoothQuantile(0.25,0.2),Expectile(0.25),PseudoHuber(1.345),StudentT(4.0,1.0))
        eta=X*beta; mu=zeros(n); w=zeros(n); q=zeros(n); g=zeros(p)
        grad_weights_xb!(f,g,mu,w,q,X,eta,y,beta)
        @test all(isfinite,g)
        @test all(>(0),w)
        d=randn(rng,p); h=1e-6
        fp=logloss_xb(f,X*(beta+h*d),y,beta+h*d)
        fm=logloss_xb(f,X*(beta-h*d),y,beta-h*d)
        @test isapprox(dot(g,d),(fp-fm)/(2h);rtol=2e-4,atol=2e-5)
    end
end

@testset "small regression smoke tests" begin
    rng=MersenneTwister(2); n,p=100,8
    X0=randn(rng,n,p); X=hcat(ones(n),X0); beta=randn(rng,p)/sqrt(p)
    for f in (SmoothQuantile(0.5,0.2),Expectile(0.5),PseudoHuber(1.345),StudentT(4.0,1.0))
        y,_,_=simulate_y_from_beta(X0,beta,f;seed=3,beta0=0.2)
        fit=irls_cholesky(X,y;family=f,ridge=1e-6,maxiter=50,outer_nesterov=false,verbose=false)
        @test all(isfinite,fit.beta)
        @test isfinite(logloss_xb(f,X*fit.beta,y,fit.beta;ridge=1e-6))
    end
end
