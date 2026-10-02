module SpectralMM

using SparseArrays
using LinearAlgebra
using Random
using Statistics
using Printf
using Distributions
using Krylov

import GLM
import StatsAPI
import StatsAPI: coef, predict, stderror, vcov, confint
import StatsModels


# ============================================================
# Source files
# ============================================================

include("families.jl")
include("iteration_log.jl")
include("cholesky.jl")
include("krylov.jl")
include("lanczos.jl")
include("spectral.jl")
include("simulation.jl")
include("inference_common.jl")
include("inference.jl")
include("glm.jl")
include("statsapi.jl")
include("cpp_backend.jl")

include("backend_api.jl")
const FittedModel = _CppBackend.FittedModel
StatsAPI.coef(m::FittedModel) = m.coef
StatsAPI.predict(m::FittedModel, X::AbstractMatrix; kwargs...) = _CppBackend.predict(m, X; kwargs...)
StatsAPI.stderror(m::FittedModel) = _CppBackend.stderror(m)
StatsAPI.vcov(m::FittedModel) = _CppBackend.vcov(m)
StatsAPI.confint(m::FittedModel) = _CppBackend.confint(m)



# ============================================================
# Public API
# ============================================================

export glm, fit, coef, predict, stderror, vcov, confint
export SpectralGLM, SpectralModel, FittedModel

export Expectile
export SmoothQuantile
export PseudoHuber
export StudentT

export SpectralOptions
export InnerOptions
export OuterOptions

export spectral_mm
export MMResult


end

# export IRLSFamily, GLMFamily, AsymmetricFamily, RobustFamily
# export BernoulliLogit, BernoulliProbit, BinomialLogit, PoissonLog, NegativeBinomialLog
# export GaussianIdentity, GaussianLog, GammaLog, GammaInverse, TweedieLog
# export SmoothQuantile, Expectile, PseudoHuber, StudentT
# export SpectralOptions, InnerOptions, OuterOptions, MMResult
# export IRLSCholeskyResult, IRLSKrylovResult
# export spectral_mm, irls_cholesky, irls_krylov
# export generate_correlated_design, simulate_y_from_beta, normal_expectile
# export logloss_xb, weights_xb!, grad_weights_xb!, work_y!
# export reset_smm_timer!, show_smm_timer!

# export glm, SpectralGLM

# end