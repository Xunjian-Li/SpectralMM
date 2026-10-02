# Generate article-style shared binary fixtures and time original Julia Spectral-MM.
using SpectralMM, LinearAlgebra, SparseArrays, Random, Statistics, Printf
BLAS.set_num_threads(1)
p=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1000
out=length(ARGS)>1 ? ARGS[2] : "build/family-performance"
mkpath(joinpath(out,"data"))
cases=[("gaussian",SpectralMM.GaussianIdentity()),("bernoulli",SpectralMM.BernoulliLogit()),
 ("probit",SpectralMM.BernoulliProbit()),("poisson",SpectralMM.PoissonLog()),
 ("gamma",SpectralMM.GammaLog()),("negative_binomial",SpectralMM.NegativeBinomialLog(4.)),
 ("smooth_quantile",SpectralMM.SmoothQuantile(.25,.1)),("expectile",SpectralMM.Expectile(.25)),
 ("pseudo_huber",SpectralMM.PseudoHuber(1.)),("student_t",SpectralMM.StudentT(4.))]
open(joinpath(out,"Julia.csv"),"w") do io
 println(io,"language,family,storage,n,p,p_total,method,median_ms,min_ms,max_ms,samples,converged,quality_pass,iterations,inner_iterations,loss,gradnorm,relgrad")
 for (name,family) in cases, storage in ("dense","sparse")
  n=2p; rng=MersenneTwister(3124); truth=randn(rng,p)./sqrt(p)
  if storage=="dense"
   Z=SpectralMM.generate_correlated_design(n,p;rho=.5,seed=1)
  else
   rows=repeat(1:n,inner=20); cols=rand(rng,1:p,length(rows)); vals=randn(rng,length(rows))
   Z=sparse(rows,cols,vals,n,p)
  end
  y,_,_=SpectralMM.simulate_y_from_beta(Z,truth,family;seed=3124,beta0=.5)
  X=hcat(storage=="dense" ? ones(n) : sparse(ones(n)),Z)
  b0=family isa SpectralMM.GLMFamily ? SpectralMM._init_beta(X,y,family,nothing) : zeros(p+1)
  base=joinpath(out,"data","$(name)_$(storage)_p$p")
  if storage=="dense"; write(base*"_X.bin",X)
  else
   write(base*"_values.bin",X.nzval); write(base*"_indices.bin",Int32.(X.rowval.-1)); write(base*"_indptr.bin",Int32.(X.colptr.-1))
  end
  write(base*"_y.bin",y); write(base*"_beta0.bin",b0)
  run()=SpectralMM.spectral_mm(X,y;family,beta0=b0,
   spectral=SpectralMM.SpectralOptions{Float64}(k=5,rho=1e-6,krylovdim=12,ridge=0.,resid_tol=.5,correction_tol=.05),
   inner=SpectralMM.InnerOptions{Float64}(solver=:pcg,maxiter=200,eta_max=.6,forcing_c=10.,forcing_alpha=0.,nesterov=true),
   outer=SpectralMM.OuterOptions{Float64}(maxiter=1000,gtol=1e-6,relgtol=1e-8,nesterov=true,verbose=false))
  run(); times=Float64[]; result=nothing
  for sample in 1:3
   GC.gc(); push!(times,1000*@elapsed(result=run()))
  end
  g=zeros(p+1); mu=zeros(n); w=zeros(n); q=zeros(n)
  SpectralMM.grad_weights_xb!(family,g,mu,w,q,X,X*b0,y,b0)
  scale=1+norm(g)
  SpectralMM.grad_weights_xb!(family,g,mu,w,q,X,X*result.beta,y,result.beta)
  loss=SpectralMM.logloss_xb(family,X*result.beta,y,result.beta)
  gn=norm(g); rel=gn/scale; quality=gn<=1e-6 || rel<=1e-8
  println(io,join(("Julia",name,storage,n,p,p+1,"Spectral-MM",median(times),minimum(times),maximum(times),3,result.converged,quality,result.iters,sum(result.inner_iters),loss,gn,rel),',')); flush(io)
  write(base*"_julia_coef.bin",result.beta)
  println("Julia $name $storage: $(round(median(times),digits=2)) ms; quality=$quality"); flush(stdout)
 end
end
