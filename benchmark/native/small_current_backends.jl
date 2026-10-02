using SpectralMM, GLM, Distributions, CSV, DataFrames, LinearAlgebra, Statistics, Random, Printf, Libdl
BLAS.set_num_threads(parse(Int, get(ENV,"SPECTRALMM_BENCH_THREADS","1")))
println("Julia ",VERSION,"; Julia threads=",Threads.nthreads(),"; BLAS threads=",BLAS.get_num_threads(),"; BLAS=",BLAS.get_config())
println("VECLIB_MAXIMUM_THREADS=",get(ENV,"VECLIB_MAXIMUM_THREADS","unset"))
data=CSV.read("examples/data.csv",DataFrame); X=Matrix{Float64}(data[:,[:x1,:x2,:x3]])
options=(beta0=zeros(4),solver=:pcg,rank=3,maxiter=500,inner_maxiter=100,floor=1e-6,eta_max=.5,correction_tol=.01,resid_tol=.5,gtol=1e-6,relgtol=1e-8,accept_stalled=false,verbose=false)
cases=[(:bernoulli,Bernoulli(),LogitLink(),NamedTuple()),(:poisson,Poisson(),LogLink(),NamedTuple()),(:pseudo_huber,PseudoHuber(1.),nothing,(delta=1.,)),(:student_t,StudentT(4.,.5),nothing,(nu=4.,sigma=.5))]
rows=NamedTuple[]; rng=MersenneTwister(90230)
function batchtime(f,n)
 t=time_ns();for _ in 1:n;f();end;(time_ns()-t)/n/1000
end
for (name,family,link,params) in cases, inf in (false,true)
 y=Vector{Float64}(data[!,name])
 jf=link===nothing ? ()->SpectralMM.fit(X,y,family;options...,inference=inf) : ()->SpectralMM.glm(X,y,family,link;options...,inference=inf)
 cf=()->SpectralMM.fit(X,y;backend=:cpp,options...,family=name,family_options=params,inference=inf,krylovdim=4,accept_negligible=false)
 for _ in 1:10;jf();cf();end
 jm=jf();cm=cf();diff=maximum(abs.(coef(jm)-coef(cm)))
 @assert diff<1e-5
 @assert last(jm.result.gradnorms)<=1e-6 || last(jm.result.relgradnorms)<=1e-8
 @assert cm.info.gradient_converged
 @assert jm.inference.status==cm.inference.status
 if inf;@assert isapprox(stderror(jm),stderror(cm);atol=1e-5,rtol=1e-5);end
 jt=Float64[];ct=Float64[]
 for round in 1:21
  for cpp in shuffle(rng,[false,true])
   GC.gc();push!(cpp ? ct : jt,batchtime(cpp ? cf : jf,100))
  end
 end
 push!(rows,(family=string(name),inference=inf,julia_us=median(jt),cpp_us=median(ct),cpp_over_julia=median(ct)/median(jt),julia_q25=quantile(jt,.25),julia_q75=quantile(jt,.75),cpp_q25=quantile(ct,.25),cpp_q75=quantile(ct,.75),julia_iterations=jm.result.iters,cpp_iterations=cm.info.iterations,max_coef_difference=diff,julia_relgrad=last(jm.result.relgradnorms),cpp_relgrad=cm.info.relgradnorm))
 println(last(rows));flush(stdout)
end
CSV.write(get(ENV,"SPECTRALMM_BENCH_OUTPUT","benchmark/native/results/small-current-backends.csv"),DataFrame(rows))
println("C++ library: ",SpectralMM._CppBackend.library_path())
