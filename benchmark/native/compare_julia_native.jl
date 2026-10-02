# Run from the repository root after installing SpectralMM.
# Timings exclude startup/JIT warm-up and include wrapper/result construction.
using SpectralMM, GLM, Distributions, LinearAlgebra, Statistics, Random, Printf
BLAS.set_num_threads(1)
function compare(n,d)
 rng=MersenneTwister(764);X=randn(rng,n,d)*.4;b=randn(rng,d)/sqrt(d)
 y=Float64.(rand(rng,n).<1 ./ (1 .+ exp.(-(.2 .+ X*b))))
 p=d+1;rank=min(5,p-1);kd=min(p,max(12,rank+1));b0=zeros(p)
 calls=Dict{String,Function}()
 for inf in (false,true)
  calls["Julia inference=$(inf)"]=()->SpectralMM.glm(X,y,Bernoulli(),LogitLink();
   beta0=b0,solver=:pcg,rank=rank,maxiter=500,inner_maxiter=100,floor=1e-6,
   eta_max=.5,correction_tol=.01,resid_tol=.5,gtol=1e-6,relgtol=1e-8,
   accept_stalled=false,inference=inf,verbose=false)
  calls["C++ via Julia inference=$(inf)"]=()->SpectralMM.fit(X,y;backend=:cpp,family=:bernoulli,
   beta0=b0,solver=:pcg,rank=rank,krylovdim=kd,maxiter=500,inner_maxiter=100,floor=1e-6,
   eta_max=.5,correction_tol=.01,resid_tol=.5,gtol=1e-6,relgtol=1e-8,
   accept_stalled=false,accept_negligible=false,inference=inf,verbose=false)
 end
 for f in values(calls),i in 1:4;f();end
 times=Dict(k=>Float64[] for k in keys(calls));batch=d==3 ? 50 : 3
 for round in 1:7
  for k in shuffle(rng,collect(keys(calls)))
   GC.gc();push!(times[k],1e3*@elapsed(for _ in 1:batch;calls[k]();end)/batch)
  end
 end
 println("\nn=$n, features=$d, rank=$rank, krylovdim=$kd; median per call, seven interleaved rounds")
 for k in sort(collect(keys(calls)))
  m=calls[k]()
  if m.backend===:cpp
   it=m.info.iterations; grad=m.info.relgradnorm
  else
   it=m.result.iters;grad=last(m.result.relgradnorms)
  end
  @printf("%-32s %.5f ms; iterations=%d; relgrad=%.3e; inference=%s\n",k,median(times[k]),it,grad,m.inference.status)
 end
 for inf in (false,true)
  ratio=median(times["C++ via Julia inference=$(inf)"])/median(times["Julia inference=$(inf)"])
  @printf("C++/Julia time ratio (inference=%s): %.3f\n",string(inf),ratio)
 end
 flush(stdout)
end
compare(100,3)
compare(2000,30)
compare(2000,100)
