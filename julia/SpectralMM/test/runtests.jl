using Test, SpectralMM, Random, SparseArrays, Distributions, GLM, CSV, LinearAlgebra
rng=MersenneTwister(71)
X=randn(rng,100,3); y=Float64.(rand(rng,100).<0.5)
@testset "Public backend contract" begin
    # Pure Julia fitting must not locate/build a native library.
    withenv("SPECTRALMM_LIBRARY"=>"/nonexistent/spectralmm-library") do
        default=SpectralMM.fit(X,y;family=:bernoulli)
        explicit=SpectralMM.fit(X,y;family=:bernoulli,backend=:julia)
        @test default.backend===:julia
        @test coef(default)==coef(explicit)
        @test SpectralMM.glm(X,y,Bernoulli(),LogitLink()).backend===:julia
    end
    for family in (:bernoulli,:pseudo_huber,:negative_binomial,:tweedie,:binomial)
        response=family in (:negative_binomial,:tweedie) ? y.+1 : y
        for A in (X,sparse(X))
            j=SpectralMM.fit(A,response;family,backend=:julia)
            c=SpectralMM.fit(A,response;family,backend=:cpp)
            @test j.backend===:julia && c.backend===:cpp
            @test isapprox(coef(j),coef(c);atol=1e-5,rtol=1e-5)
            @test isapprox(predict(j,A),predict(c,A);atol=1e-5,rtol=1e-5)
            # Existing numerical backends can stop at different gradient residuals.
            # Unavailable inference must remain explicit and accessors must reject it.
            for m in (j,c)
                @test m.inference.status in ("ok","unavailable")
                if m.inference.status=="unavailable"
                    @test !isempty(m.inference.reason)
                    @test_throws ArgumentError stderror(m)
                end
            end
            if j.inference.status=="ok" && c.inference.status=="ok"
                @test isapprox(stderror(j),stderror(c);atol=1e-5,rtol=1e-5)
            end
            @test occursin("Julia",sprint(show,MIME"text/plain"(),j))
            @test occursin("C++",sprint(show,MIME"text/plain"(),c))
        end
    end
    @test SpectralMM.fit(Float32.(X),Float32.(y);family=:bernoulli).backend===:julia
    @test_throws ArgumentError SpectralMM.fit(X,y;backend=:unknown)
    for backend in (:julia,:cpp)
        @test_throws ArgumentError SpectralMM.fit(X,y;backend,family=:poisson,family_options=(tau=.5,))
        @test_throws DimensionMismatch SpectralMM.fit(X,y;backend,family=:binomial,family_options=(trials=[1.,2.],))
        m=SpectralMM.fit(X,y;backend,family=:binomial,family_options=(trials=fill(2.,100),))
        @test length(predict(m,X[1:4,:];trials=2.))==4
        @test_throws DimensionMismatch predict(m,X[1:4,:])
    end
end

@testset "Probit objective, gradient and tails" begin
    f=SpectralMM.BernoulliProbit()
    A=hcat(ones(100),X); beta=[.2,.1,-.1,.05]
    g=zeros(4); mu=zeros(100); w=zeros(100); score=zeros(100)
    SpectralMM.grad_weights_xb!(f,g,mu,w,score,A,A*beta,y,beta)
    fd=setprecision(256) do
        B=BigFloat.(A); response=BigFloat.(y); b=BigFloat.(beta); h=big"1e-25"
        [begin
            plus=copy(b); minus=copy(b); plus[j]+=h; minus[j]-=h
            (SpectralMM.logloss_xb(f,B*plus,response,plus)-
             SpectralMM.logloss_xb(f,B*minus,response,minus))/(2h)
         end for j in eachindex(b)]
    end
    @test maximum(abs.(g-Float64.(fd)))<1e-11
    for eta in (-100.,-40.,-12.,-10.000001,-10.,-8.,-1.,0.,1.,8.,10.,10.000001,12.,40.,100.), response in (0.,1.)
        loss=-Distributions.logcdf(Normal(),response==1 ? eta : -eta)
        derivative=exp(Distributions.logpdf(Normal(),eta)+loss)*(response==1 ? -1 : 1)
        @test isapprox(SpectralMM.probit_score(eta,response),derivative;atol=1e-11,rtol=1e-11)
        @test isapprox(SpectralMM.logloss_xb(f,[eta],[response],[eta]),loss;atol=1e-11,rtol=1e-12)
        native=SpectralMM.fit(ones(1,1),[response];backend=:cpp,family=:probit,
            intercept=false,beta0=[eta],rank=0,maxiter=1,inference=false,trace=true,
            accept_stalled=false,accept_negligible=false)
        @test isapprox(first(native.trace).loss,loss;atol=1e-11,rtol=1e-12)
        @test isapprox(first(native.trace).gradnorm,abs(derivative);atol=1e-11,rtol=1e-11)
    end
end

@testset "Previously stalled small-data fits" begin
    # Preserve actual cases that exposed inconsistent objective derivatives and
    # false line-search rejection at floating-point rounding scale.
    data=CSV.File(joinpath(@__DIR__,"..","examples","data.csv"))
    features=hcat(data.x1,data.x2,data.x3)
    for name in (:probit,:negative_binomial,:student_t), storage in (identity,sparse)
        response=Float64.(getproperty(data,name)); A=storage(features)
        params=name===:negative_binomial ? (theta=4.,) : name===:student_t ? (nu=4.,sigma=.5) : NamedTuple()
        j=SpectralMM.fit(A,response;family=name,family_options=params,solver=:cho,
            beta0=zeros(4),rank=3,maxiter=500,gtol=1e-6,relgtol=1e-8,accept_stalled=false)
        @test last(j.result.gradnorms)<=1e-6 || last(j.result.relgradnorms)<=1e-8
        @test j.inference.status=="ok"
        @test j.result.iters<50
        if name===:probit
            reference=GLM.glm(hcat(ones(100),features),response,Bernoulli(),ProbitLink();maxiter=100,atol=1e-14,rtol=1e-14)
            @test isapprox(coef(j),coef(reference);atol=1e-6,rtol=1e-6)
            @test isapprox(stderror(j),stderror(reference);atol=1e-6,rtol=1e-6)
        end
    end
end

@testset "Convergence on the final allowed step" begin
    for backend in (:julia,:cpp), solver in (:pcg,:mm,:cg,:cho,:cgls,:crls,:lsqr,:lsmr)
        m=SpectralMM.fit(ones(10,1),ones(10);family=:gaussian,backend,solver,
            intercept=false,beta0=[0.],rank=0,maxiter=1,gtol=1e-4,relgtol=1e-8,inference=false)
        if backend===:julia
            @test last(m.result.gradnorms)<=1e-4 || last(m.result.relgradnorms)<=1e-8
            @test m.result.converged
        else
            @test m.info.gradient_converged
            @test m.info.converged
        end
    end
end

@testset "All-family objective derivatives" begin
    data=CSV.File(joinpath(@__DIR__,"..","examples","data.csv"))
    A=hcat(ones(100),data.x1,data.x2,data.x3)
    parameters=Dict(:negative_binomial=>(theta=4.,),:tweedie=>(power=1.5,),
        :binomial=>(trials=4.,),:smooth_quantile=>(tau=.25,smoothing=.1),
        :expectile=>(tau=.25,),:pseudo_huber=>(delta=1.,),:student_t=>(nu=4.,sigma=.5))
    for name in propertynames(data)[4:end]
        family,link,params=SpectralMM._named_family(name,get(parameters,name,NamedTuple()),100)
        f=link===nothing ? family : SpectralMM._glm_family(family,link)
        response=Float64.(getproperty(data,name)); beta=[name===:gamma_inverse ? 2. : .2,.1,-.1,.05]
        g=zeros(4); mu=zeros(100); w=zeros(100); score=zeros(100)
        SpectralMM.grad_weights_xb!(f,g,mu,w,score,A,A*beta,response,beta)
        loss=SpectralMM.logloss_xb(f,A*beta,response,beta)
        fd=setprecision(256) do
            B=BigFloat.(A); z=BigFloat.(response); b=BigFloat.(beta); h=big"1e-25"
            [begin
                plus=copy(b); minus=copy(b); plus[j]+=h; minus[j]-=h
                (SpectralMM.logloss_xb(f,B*plus,z,plus)-SpectralMM.logloss_xb(f,B*minus,z,minus))/(2h)
             end for j in eachindex(b)]
        end
        @test isapprox(g,Float64.(fd);atol=1e-10,rtol=1e-11)
        c=SpectralMM.fit(A,response;family=name,family_options=params,backend=:cpp,
            intercept=false,beta0=beta,rank=3,maxiter=1,inference=false,trace=true)
        @test isapprox(first(c.trace).loss,loss;atol=1e-10,rtol=1e-12)
        @test isapprox(first(c.trace).gradnorm,norm(g);atol=1e-10,rtol=1e-11)
    end
end

@testset "Inference safeguards remain strict" begin
    for backend in (:julia,:cpp)
        unfinished=SpectralMM.fit(X,y;backend,family=:bernoulli,beta0=zeros(4),
            maxiter=1,gtol=1e-12,relgtol=1e-12,accept_stalled=false)
        @test unfinished.inference.status=="unavailable"
        @test_throws ArgumentError stderror(unfinished)
        separated=SpectralMM.fit(ones(20,1),ones(20);backend,family=:probit,intercept=false)
        @test separated.inference.status=="unavailable"
        @test_throws ArgumentError stderror(separated)
        deficient=SpectralMM.fit(hcat(X[:,1],X[:,1]),y;backend,family=:gaussian,solver=:cho)
        @test deficient.inference.status=="unavailable"
        @test_throws ArgumentError stderror(deficient)
    end
end

include("model_api.jl")

include("readme_examples.jl")
