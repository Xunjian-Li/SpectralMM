using DataFrames, StatsModels
@testset "Statistical API and formulas" begin
    data=CSV.read(joinpath(@__DIR__,"..","examples","data.csv"),DataFrame)
    X=Matrix(data[:,[:x1,:x2,:x3]])
    cases=[(:gaussian,Normal(),IdentityLink()),(:bernoulli,Bernoulli(),LogitLink()),
        (:probit,Bernoulli(),ProbitLink()),(:poisson,Poisson(),LogLink()),
        (:gamma,Gamma(),LogLink()),(:negative_binomial,NegativeBinomial(4.,.5),LogLink())]
    rows=NamedTuple[]
    for (name,family,link) in cases, backend in (:julia,:cpp)
        y=data[!,name]; dd=copy(data); dd.y=y
        m=SpectralMM.glm(X,y,family,link;backend,start=zeros(4))
        f=SpectralMM.glm(@formula(y~x1+x2+x3),dd,family,link;backend,start=zeros(4))
        a=SpectralMM.glm(hcat(ones(100),X),y,family,link;backend,intercept=false,start=zeros(4))
        @test coef(m)≈coef(f) atol=1e-10
        @test coef(m)≈coef(a) atol=1e-10
        @test fitted(m)≈fitted(f) atol=1e-10
        @test predict(m;type=:link)≈predict(f,dd;type=:link) atol=1e-10
        @test diagnostics(m).objective≈diagnostics(f).objective atol=1e-9
        @test predict(f,dd)≈fitted(m) atol=1e-10
        @test residuals(m)≈y-fitted(m)
        @test nobs(m)==100
        @test m.inference.status=="ok"
        @test size(vcov(m))==(4,4)
        @test size(confint(m))==(4,2)
        @test length(StatsModels.coefnames(f))==4
        @test occursin("Estimate",sprint(show,MIME"text/plain"(),coeftable(f)))
        @test diagnostics(m).converged
        for (quantity,values) in (("coef",coef(m)),("eta",predict(m;type=:link)),("mu",fitted(m)),
            ("objective",[diagnostics(m).objective]),("deviance",[deviance(m)]),
            ("loglikelihood",[loglikelihood(m)]),("converged",[Float64(diagnostics(m).converged)]))
            for (i,v) in enumerate(values)
                push!(rows,(backend=string(backend),family=string(name),quantity=quantity,index=i-1,value=v))
            end
        end
    end
    if haskey(ENV,"SPECTRALMM_API_EXPORT"); CSV.write(ENV["SPECTRALMM_API_EXPORT"],rows); end
    d=DataFrame(x=sin.(0:99),g=repeat(["a","b","c","a"],25))
    d.y = 0.3 .+ 0.4 .* d.x .+ 0.2 .* (d.g .== "b") .- 0.1 .* (d.g .== "c") .+ 0.1 .* cos.((0:99) .* 0.7)
    for formula in (@formula(y~x+g),@formula(y~0+x+g)), backend in (:julia,:cpp)
        f=SpectralMM.glm(formula,d,Normal(),IdentityLink();backend)
        _,A=StatsModels.modelcols(f.formula,d)
        m=SpectralMM.glm(A,d.y,Normal(),IdentityLink();backend,intercept=false)
        @test coef(f)≈coef(m) atol=1e-10
        nd=DataFrame(x=[.2,.5],g=["c","b"])
        @test predict(f,nd)≈StatsModels.modelcols(f.formula.rhs,nd)*coef(f)
        @test_throws Exception predict(f,DataFrame(x=[1.],g=["unseen"]))
    end
    big=randn(MersenneTwister(9),80,51); y=sin.(1:80)
    for backend in (:julia,:cpp)
        m=SpectralMM.glm(big,y,Normal(),IdentityLink();backend,control=SpectralMMControl(maxiter=1))
        @test m.inference.status=="skipped"
        @test_throws ArgumentError vcov(m)
        @test_throws ArgumentError SpectralMM.glm(big,y,Normal(),IdentityLink();backend,weights=ones(80))
    end
    f=SpectralMM.fit(@formula(y~x),d;family=:pseudo_huber)
    @test f.link===nothing
    @test length(fitted(f))==100
    @test_throws ArgumentError deviance(f)
    @test_throws ArgumentError SpectralMM.glm(X,data.bernoulli,Bernoulli(),LogitLink();start=zeros(4),beta0=zeros(4))
end
