using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "julia"))

using SpectralMM, BenchmarkTools, DataFrames, GLM
using LinearAlgebra, Random, Statistics
import Distributions

glm_spec(::SpectralMM.BernoulliLogit)   = (Distributions.Bernoulli(), LogitLink())
glm_spec(::SpectralMM.BernoulliProbit)  = (Distributions.Bernoulli(), ProbitLink())
glm_spec(::SpectralMM.GaussianIdentity) = (Distributions.Normal(), IdentityLink())
glm_spec(::SpectralMM.PoissonLog)       = (Distributions.Poisson(), LogLink())
glm_spec(::SpectralMM.GammaLog)         = (Distributions.Gamma(), LogLink())
glm_spec(::SpectralMM.GammaInverse)     = (Distributions.Gamma(), InverseLink())
glm_spec(f::SpectralMM.NegativeBinomialLog) = (Distributions.NegativeBinomial(f.theta), LogLink())

glm_supported(::SpectralMM.IRLSFamily) = false
glm_supported(::SpectralMM.GLMFamily) = true
glm_supported(::SpectralMM.TweedieLog) = false
glm_supported(::SpectralMM.BinomialLogit) = false

function initial_beta(X, y, family)
    T = eltype(X)
    family isa SpectralMM.GLMFamily ?
        SpectralMM._init_beta(X, y, family, nothing) :
        zeros(T, size(X, 2))
end

function evaluate_solution(X, y, β, family; ridge=0.0, scale=1.0)
    T = eltype(X); n, p = size(X)
    η = X * β
    μ, w, r, g = zeros(T,n), zeros(T,n), zeros(T,n), zeros(T,p)

    loss = SpectralMM.logloss_xb(family, η, y, β; ridge=T(ridge))
    SpectralMM.grad_weights_xb!(
        family, g, μ, w, r, X, η, y, β;
        ridge=T(ridge), w_floor=T(1e-12))

    gn = norm(g)
    return loss, gn, gn / scale
end

function grad_scale(X, y, family, β0; ridge=0.0)
    _, gn, _ = evaluate_solution(X, y, β0, family; ridge=ridge)
    1 + gn
end

function bench_time(f; samples=5)
    b = @benchmarkable $f() evals=1
    b.params.samples = samples
    median(run(b)).time / 1e6
end

relerr(β, βref) = norm(β - βref) / max(1, norm(βref))

function benchmark_problem(
    X, y, family;
    spectral, inner, outer,
    ridge=0.0,
    maxiter=1000,
    gtol=1e-6,
    relgtol=1e-8,
    outer_nesterov=true,
    krylov_solvers=(:cgls, :crls, :lsqr, :lsmr),
    samples=5,
    compute_reference=true,
)
    T = eltype(X)
    n, p = size(X)

    β0 = initial_beta(X, y, family)
    scale = grad_scale(X, y, family, β0; ridge=ridge)

    βref = if compute_reference
        SpectralMM.irls_cholesky(
            X, y;
            family=family,
            beta0=β0,
            ridge=T(ridge),
            maxiter=5000,
            gtol=T(1e-10),
            relgtol=T(1e-12),
            outer_nesterov=false,
            verbose=false,
        ).beta
    else
        nothing
    end

    rows = NamedTuple[]

    function add!(name, f; rank=missing)
        fit = f()
        tm = bench_time(f; samples=samples)

        loss, gn, rgn = evaluate_solution(
            X, y, fit.beta, family;
            ridge=ridge,
            scale=scale,
        )

        ii = hasproperty(fit, :inner_iters) ? fit.inner_iters : Int[]

        push!(rows, (
            Method=name,
            Time_ms=tm,
            Outer=fit.iters,
            Inner=isempty(ii) ? missing : sum(ii),
            Loss=loss,
            GradNorm=gn,
            RelGrad=rgn,
            RelErr=βref === nothing ? missing : relerr(fit.beta, βref),
            Rank=rank,
            Converged=fit.converged,
        ))
    end

    add!("Cholesky", () -> SpectralMM.irls_cholesky(
        X, y;
        family=family,
        beta0=β0,
        ridge=T(ridge),
        maxiter=maxiter,
        gtol=T(gtol),
        relgtol=T(relgtol),
        outer_nesterov=outer_nesterov,
        verbose=false,
    ))

    add!("Spectral-MM", () -> SpectralMM.spectral_mm(
        X, y;
        family=family,
        beta0=β0,
        spectral=spectral,
        inner=inner,
        outer=outer,
    ); rank=spectral.k)

    for solver in krylov_solvers
        add!(uppercase(string(solver)), () -> SpectralMM.irls_krylov(
            X, y;
            family=family,
            beta0=β0,
            solver=solver,
            ridge=T(ridge),
            maxiter=maxiter,
            gtol=T(gtol),
            relgtol=T(relgtol),
            outer_nesterov=outer_nesterov,
            verbose=false,
        ))
    end

    if glm_supported(family) && ridge == 0
        D, L = glm_spec(family)

        f = () -> GLM.glm(
            X, y, D, L;
            maxiter=maxiter,
            atol=gtol,
            rtol=relgtol,
        )

        fit = f()
        tm = bench_time(f; samples=samples)
        β = coef(fit)

        loss, gn, rgn = evaluate_solution(
            X, y, β, family; ridge=ridge, scale=scale)

        push!(rows, (
            Method="GLM.jl",
            Time_ms=tm,
            Outer=missing,
            Inner=missing,
            Loss=loss,
            GradNorm=gn,
            RelGrad=rgn,
            RelErr=βref === nothing ? missing : relerr(β, βref),
            Rank=missing,
            Converged=true,
        ))
    end
    
    df = DataFrame(rows)

    if βref === nothing
        df.LossGap = fill(missing, nrow(df))
    else
        loss_ref, _, _ = evaluate_solution(
            X, y, βref, family;
            ridge=ridge,
            scale=scale,
        )
        df.LossGap = df.Loss .- loss_ref
    end

    df.Family = fill(string(nameof(typeof(family))), nrow(df))
    df.n = fill(n, nrow(df))
    df.p = fill(p, nrow(df))

    sort!(df, :Time_ms)
    return df
end