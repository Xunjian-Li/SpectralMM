using Test, Random, SpectralMM, Distributions, GLM, DataFrames, StatsModels
@testset "Iteration likelihood and non-GLM prediction" begin
    rng=MersenneTwister(82); X=randn(rng,100,3); y=.3 .+ X*[.1,.2,-.1] .+ .2*randn(rng,100)
    new=randn(rng,7,3)
    for backend in (:julia,:cpp), f in (PseudoHuber(),Expectile(.3),SmoothQuantile(.3,.2),StudentT(4.,1.)), intercept in (true,false)
        m=SpectralMM.fit(X,y,f;backend,intercept,inference=false)
        expected=intercept ? new*coef(m)[2:end].+coef(m)[1] : new*coef(m)
        @test predict(m,new)≈expected
        dat=DataFrame(x1=X[:,1],x2=X[:,2],x3=X[:,3],y=y)
        formula=intercept ? @formula(y~x1+x2+x3) : @formula(y~0+x1+x2+x3)
        name=f isa PseudoHuber ? :pseudo_huber : f isa Expectile ? :expectile : f isa SmoothQuantile ? :smooth_quantile : :student_t
        options=f isa PseudoHuber ? (delta=f.delta,) : f isa Expectile ? (tau=f.tau,) : f isa SmoothQuantile ? (tau=f.tau,smoothing=f.epsilon) : (nu=f.nu,sigma=f.sigma)
        fm=SpectralMM.fit(formula,dat;family=name,family_options=options,backend,inference=false)
        @test predict(fm,DataFrame(x1=new[:,1],x2=new[:,2],x3=new[:,3]))≈expected atol=1e-8
    end
    for backend in (:julia,:cpp), solver in (:pcg,:mm,:cho,:cg,:cgls,:crls,:lsqr,:lsmr)
        opts=(;family=:gaussian,backend,solver,inference=false,ridge=.1,dispersion=.7)
        quiet=SpectralMM.fit(X,y;opts...)
        mktemp() do path,io
            m=redirect_stdout(io) do
                SpectralMM.fit(X,y;opts...,verbose=true)
            end
            flush(io); text=read(path,String)
            @test coef(m)==coef(quiet)
            @test all(label->occursin(label,text),("LogLik","GradNorm","Stepsize"))
            @test !occursin("EigRes",text) && !occursin("InnerRes",text)
            ll=parse(Float64,match(r"Final LogLik: ([^\n]+)",text).captures[1])
            @test ll≈loglikelihood(m) rtol=1e-6
        end
    end
    for family in (:gaussian,:gaussian_log,:bernoulli,:probit,:poisson,:gamma,:gamma_inverse,:negative_binomial,:binomial,:student_t)
        response=family in (:bernoulli,:probit,:binomial) ? Float64.(y.>0) : family in (:poisson,:negative_binomial) ? Float64.(y.>0).+1 : family in (:gamma,:gamma_inverse,:gaussian_log) ? exp.(y) : y
        for dispersion in (nothing,.7)
            m=SpectralMM.fit(X,response;family,backend=:cpp,trace=true,inference=false,ridge=.1,dispersion,maxiter=4)
            @test last(m.trace).loglikelihood≈loglikelihood(m) rtol=1e-9 atol=1e-8
        end
    end
end

@testset "Unified summary and q aliases" begin
    rng=MersenneTwister(4); X=randn(rng,100,3); y=.3 .+ .2*X[:,1] .+ .3*randn(rng,100)
    @test Expectile().q == 0.5
    @test Expectile(.3).q == 0.3
    @test Expectile(Float32(.3)) isa Expectile{Float32}
    @test Expectile{Float64}(.3).tau == 0.3
    @test_throws ArgumentError Expectile(0)
    @test_throws ArgumentError Expectile(1)
    @test_throws ArgumentError Expectile(q=NaN)
    @test Expectile(q=.3).q==Expectile(tau=.3).tau
    @test SmoothQuantile(q=.3).q==SmoothQuantile(tau=.3).tau
    @test_throws ArgumentError Expectile(q=.3,tau=.3)
    @test_throws ArgumentError SmoothQuantile(q=.3,tau=.3)
    for backend in (:julia,:cpp), family in (:expectile,:smooth_quantile)
        a=SpectralMM.fit(X,y;family,backend,family_options=(q=.3,))
        b=SpectralMM.fit(X,y;family,backend,family_options=(tau=.3,))
        @test coef(a)==coef(b)
        @test_throws ArgumentError SpectralMM.fit(X,y;family,backend,family_options=(q=.3,tau=.3))
        text=sprint(show,MIME"text/plain"(),a)
        @test count("Family:",text)==2
        @test occursin("q=0.3",text) && !occursin("tau",text)
    end
    for backend in (:julia,:cpp), (family,response,stat) in ((:gaussian,y,"t"),(:bernoulli,Float64.(y.>0),"z"))
        m=SpectralMM.fit(X,response;family,backend)
        text=sprint(show,MIME"text/plain"(),m)
        @test startswith(text,"SpectralMM Regression Model")
        @test occursin(stat*" statistic",text) && occursin("p-value",text)
        @test first(findfirst("Stopping criterion",text))<first(findfirst("Inference:",text))
    end
end
