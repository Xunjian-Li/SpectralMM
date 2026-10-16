using SpectralMM, LinearAlgebra, SparseArrays, Random, Test
@testset "Native accelerated MM matches Julia inner momentum" begin
rng=MersenneTwister(214)
X=randn(rng,100,9); X[:,1].=1.; y=randn(rng,100); b0=zeros(9)
for A in (X,sparse(X)), limit in (2,6,30)
 j=SpectralMM.spectral_mm(A,y; family=SpectralMM.GaussianIdentity(),beta0=b0,
   spectral=SpectralMM.SpectralOptions{Float64}(k=2,krylovdim=9,rho=1e-6,ridge=0.),
   inner=SpectralMM.InnerOptions{Float64}(solver=:mm,maxiter=limit,eta_max=.01,forcing_c=10.,forcing_alpha=0.,nesterov=true),
   outer=SpectralMM.OuterOptions{Float64}(maxiter=1,gtol=1e-12,relgtol=1e-12,nesterov=false,accept_stalled=false,verbose=false))
 c=SpectralMM._CppBackend.fit(A,y; family=:gaussian,beta0=b0,intercept=false,inference=false,
   solver=:mm,rank=2,krylovdim=9,floor=1e-6,ridge=0.,inner_maxiter=limit,eta_max=.01,
   maxiter=1,gtol=1e-12,relgtol=1e-12,nesterov=false,accept_stalled=false,accept_negligible=false)
 @test j.beta ≈ c.coef atol=1e-8 rtol=1e-8
end
end
