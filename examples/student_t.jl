using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using SpectralIRLS,Random,LinearAlgebra

rng=MersenneTwister(3124)
m,p=2000,500
X=generate_correlated_design(m,p;rho=0.5,seed=1)
beta_true=randn(rng,p)/sqrt(p)
family=StudentT(4.0,1.0)
y,_,_=simulate_y_from_beta(X,beta_true,family;seed=3124,beta0=0.5)
Xfit=hcat(ones(m),X)

spectral=SpectralOptions{Float64}(k=5,krylovdim=20,ridge=1e-6,correction_tol=5e-2)
inner=InnerOptions{Float64}(solver=:pcg,maxiter=100,eta_max=0.6,forcing_c=10.0,abstol=1e-6)
outer=OuterOptions{Float64}(maxiter=100,gtol=1e-6,relgtol=1e-8,nesterov=true,verbose=true)

fit=spectral_mm(Xfit,y;family=family,spectral=spectral,inner=inner,outer=outer)
println("converged = ",fit.converged,", iterations = ",fit.iters)
