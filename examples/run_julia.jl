# Run from the repository root after examples/run_python.py.
using SpectralMM, GLM, Distributions, StatsAPI, CSV, DataFrames, LinearAlgebra, Printf
BLAS.set_num_threads(1)
data=CSV.read("examples/data.csv",DataFrame)
X=Matrix{Float64}(data[:,[:x1,:x2,:x3]])
cases=[
 ("gaussian",Normal(),IdentityLink()),("bernoulli",Bernoulli(),LogitLink()),
 ("probit",Bernoulli(),ProbitLink()),("poisson",Poisson(),LogLink()),("gamma",Gamma(),LogLink()),
 ("negative_binomial",SpectralMM.NegativeBinomialLog(4.),nothing),
 ("gaussian_log",Normal(),LogLink()),("gamma_inverse",Gamma(),InverseLink()),
 ("tweedie",SpectralMM.TweedieLog(1.5),nothing),("binomial",SpectralMM.BinomialLogit(4.),nothing),
 ("smooth_quantile",SmoothQuantile(.25,.1),nothing),("expectile",Expectile(.25),nothing),
 ("pseudo_huber",PseudoHuber(1.),nothing),("student_t",StudentT(4.,.5),nothing)]
rows=NamedTuple[]
open("examples/results/Julia.txt","w") do io
 redirect_stdout(io) do
  for (name,family,link) in cases
   println("\n=== $name: 100 observations, 3 features, 4 coefficients ===")
   y=Vector{Float64}(data[!,name]);b0=name=="gamma_inverse" ? [2.,0.,0.,0.] : zeros(4)
   opts=(;beta0=b0,solver=:pcg,rank=3,maxiter=500,gtol=1e-6,relgtol=1e-8,
           accept_stalled=false,verbose=true)
   if link!==nothing
    model=SpectralMM.glm(X,y,family,link;opts...)
   elseif family isa Union{SpectralMM.AsymmetricFamily,SpectralMM.RobustFamily}
    model=SpectralMM.fit(X,y,family;opts...)
   else
    # These three internal GLM families currently use the public low-level API.
    # It expects an explicit intercept column and does not attach inference.
    A=hcat(ones(100),X)
    result=spectral_mm(A,y;family,beta0=b0,
      spectral=SpectralOptions{Float64}(k=3,rho=1e-6,ridge=0.,krylovdim=4,correction_tol=.01),
      inner=InnerOptions{Float64}(solver=:pcg,eta_max=.5),
      outer=OuterOptions{Float64}(maxiter=500,gtol=1e-6,relgtol=1e-8,accept_stalled=false,verbose=true))
    println("Coefficients (intercept first): ",result.beta)
    println("Inference: not attached by the low-level Julia API; see Python/R results.")
    model=nothing
   end
   if model!==nothing
    show(stdout,MIME"text/plain"(),model);println()
    println("First five fitted values: ",predict(model,X)[1:5])
    result=model.result;inference=model.inference;estimates=coef(model)
   else
    estimates=result.beta;inference=(status="low_level",)
   end
   ok=inference.status=="ok"
   for (j,term) in enumerate(["(Intercept)","x1","x2","x3"])
    push!(rows,(family=name,term,estimate=estimates[j],
      std_error=ok ? inference.std_error[j] : missing,
      statistic=ok ? inference.statistic[j] : missing,
      p_value=ok ? inference.p_value[j] : missing,
      lower=ok ? inference.conf_int[j,1] : missing,upper=ok ? inference.conf_int[j,2] : missing,
      wald_chisq=ok ? inference.wald_chisq[j] : missing,
      wald_p_value=ok ? inference.wald_p_value[j] : missing,
      loss=last(result.losses),relgradnorm=last(result.relgradnorms),iterations=result.iters,
      converged=last(result.gradnorms)<=1e-6 || last(result.relgradnorms)<=1e-8,
      inference_status=inference.status))
   end
  end
 end
end
CSV.write("examples/results/Julia.csv",DataFrame(rows))
println("Julia completed all 14 small-data examples.")
