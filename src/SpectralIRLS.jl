module SpectralIRLS

using LinearAlgebra
using Random
using Statistics
using Printf
using Distributions
using Krylov

# Portable no-op replacement for TimerOutputs.@timeit.
# Replace by TimerOutputs if detailed profiling is desired.
macro timeit(timer,label,ex)
    return esc(ex)
end
const SMM_TIMER = nothing

include("families.jl")
include("lanczos.jl")
include("spectral.jl")
include("cholesky.jl")
include("krylov.jl")
include("simulation.jl")

export IRLSFamily,GLMFamily,AsymmetricFamily,RobustFamily
export BernoulliLogit,BernoulliProbit,BinomialLogit,PoissonLog,NegativeBinomialLog
export GaussianIdentity,GaussianLog,GammaLog,GammaInverse,TweedieLog
export SmoothQuantile,Expectile,PseudoHuber,StudentT
export SpectralOptions,InnerOptions,OuterOptions,MMResult
export IRLSCholeskyResult,IRLSKrylovResult
export spectral_mm,irls_cholesky,irls_krylov
export generate_correlated_design,simulate_y_from_beta,normal_expectile
export logloss_xb,weights_xb!,grad_weights_xb!,work_y!
export reset_smm_timer!,show_smm_timer!

end
