module SpectralMM

using LinearAlgebra
using Random
using Statistics
using Printf
using Distributions
using Krylov

import GLM
import StatsAPI
import StatsModels


# ============================================================
# Source files
# ============================================================

include("families.jl")
include("cholesky.jl")
include("krylov.jl")
include("lanczos.jl")
include("spectral.jl")
include("simulation.jl")
include("glm.jl")
include("statsapi.jl")


# ============================================================
# Public API
# ============================================================

export glm, fit
export SpectralGLM, SpectralModel

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