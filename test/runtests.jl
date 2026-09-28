using Test
using Random
using LinearAlgebra
using Statistics
using Distributions
using GLM
using StatsAPI
using StatsModels
using DataFrames
using SpectralMM


# ============================================================
# Helpers
# ============================================================

function make_regression_data(; n=400, p=20, seed=1234)
    rng = MersenneTwister(seed)
    X = randn(rng, n, p)
    β = randn(rng, p) / sqrt(p)
    y = X * β + randn(rng, n)
    return X, y, β
end

function expectile_reference(X, y, τ; tol=1e-11, maxiter=1000)
    β = X \ y
    for _ in 1:maxiter
        r = y - X * β
        w = ifelse.(r .>= 0, τ, 1 - τ)
        sw = sqrt.(w)
        βnew = (X .* sw) \ (y .* sw)
        norm(βnew - β) <= tol * (1 + norm(β)) && return βnew
        β = βnew
    end
    return β
end

function pseudohuber_reference(X, y, δ; tol=1e-11, maxiter=200)
    β = X \ y
    for _ in 1:maxiter
        r = y - X * β
        u = r ./ δ
        ψ = r ./ sqrt.(1 .+ u.^2)
        w = (1 .+ u.^2).^(-1.5)
        g = -(X' * ψ)
        H = X' * (X .* w)
        norm(g) <= tol * (1 + norm(X' * y)) && return β

        d = -(H \ g)
        f0 = sum(δ^2 .* (sqrt.(1 .+ u.^2) .- 1))
        α = 1.0

        while α > 1e-12
            βnew = β + α * d
            rnew = y - X * βnew
            fnew = sum(δ^2 .* (sqrt.(1 .+ (rnew ./ δ).^2) .- 1))
            if fnew <= f0 + 1e-4 * α * dot(g, d)
                β = βnew
                break
            end
            α *= 0.5
        end
    end
    return β
end

function smoothquantile_reference(X, y, τ, ε; tol=1e-11, maxiter=200)
    β = X \ y
    for _ in 1:maxiter
        r = y - X * β
        s = sqrt.(r.^2 .+ ε^2)
        q = .-(τ - 0.5 .+ r ./ (2 .* s))
        w = ε^2 ./ (2 .* s.^3)
        g = X' * q
        H = X' * (X .* w)
        norm(g) <= tol * (1 + norm(X' * y)) && return β

        d = -(H \ g)
        f0 = sum((τ - 0.5) .* r .+ 0.5 .* s)
        α = 1.0

        while α > 1e-12
            βnew = β + α * d
            rnew = y - X * βnew
            fnew = sum((τ - 0.5) .* rnew .+
                       0.5 .* sqrt.(rnew.^2 .+ ε^2))
            if fnew <= f0 + 1e-4 * α * dot(g, d)
                β = βnew
                break
            end
            α *= 0.5
        end
    end
    return β
end

function studentt_reference(X, y, ν, σ; tol=1e-11, maxiter=1000)
    β = X \ y
    for _ in 1:maxiter
        r = y - X * β
        w = (ν + 1) ./ (ν * σ^2 .+ r.^2)
        sw = sqrt.(w)
        βnew = (X .* sw) \ (y .* sw)
        norm(βnew - β) <= tol * (1 + norm(β)) && return βnew
        β = βnew
    end
    return β
end

relerr(x, y) = norm(x - y) / max(norm(y), eps())


# ============================================================
# GLM families
# ============================================================

@testset "GLM family agreement" begin
    rng = MersenneTwister(1234)
    n, p = 1000, 8
    X = hcat(ones(n), randn(rng, n, p - 1))
    β = randn(rng, p) / (2sqrt(p))

    # Bernoulli + Logit
    η = X * β
    μ = 1 ./ (1 .+ exp.(-η))
    y = Float64.(rand(rng, n) .< μ)

    ms = SpectralMM.glm(X, y, Bernoulli(), LogitLink(); rank=5)
    mg = GLM.glm(X, y, Bernoulli(), LogitLink())
    @test relerr(coef(ms), coef(mg)) < 1e-5

    # Bernoulli + Probit
    μ = cdf.(Normal(), η)
    y = Float64.(rand(rng, n) .< μ)

    ms = SpectralMM.glm(X, y, Bernoulli(), ProbitLink(); rank=5)
    mg = GLM.glm(X, y, Bernoulli(), ProbitLink())
    @test relerr(coef(ms), coef(mg)) < 5e-5

    # Gaussian + Identity
    y = η + randn(rng, n)

    ms = SpectralMM.glm(X, y, Normal(), IdentityLink(); rank=5)
    mg = GLM.glm(X, y, Normal(), IdentityLink())
    @test relerr(coef(ms), coef(mg)) < 1e-6

    # Gaussian + Log
    μ = exp.(η)
    y = μ .* exp.(0.2 .* randn(rng, n))
    @test all(y .> 0)

    ms = SpectralMM.glm(X, y, Normal(), LogLink(); rank=5)
    mg = GLM.glm(X, y, Normal(), LogLink())
    @test relerr(coef(ms), coef(mg)) < 1e-5

    # Poisson + Log
    μ = exp.(η)
    y = Float64.(rand.(rng, Poisson.(μ)))

    ms = SpectralMM.glm(X, y, Poisson(), LogLink(); rank=5)
    mg = GLM.glm(X, y, Poisson(), LogLink())
    @test relerr(coef(ms), coef(mg)) < 1e-5

    # Gamma + Log
    μ = exp.(η)
    shape = 4.0
    y = [rand(rng, Gamma(shape, μi / shape)) for μi in μ]

    ms = SpectralMM.glm(X, y, Gamma(), LogLink(); rank=5)
    mg = GLM.glm(X, y, Gamma(), LogLink())
    @test relerr(coef(ms), coef(mg)) < 1e-5

    # Gamma + Inverse
    βinv = zeros(p)
    βinv[1] = 1.5
    βinv[2:end] .= 0.05 .* randn(rng, p - 1)
    ηinv = X * βinv
    @test minimum(ηinv) > 0

    μ = 1.0 ./ ηinv
    y = [rand(rng, Gamma(shape, μi / shape)) for μi in μ]

    ms = SpectralMM.glm(X, y, Gamma(), InverseLink(); rank=5)
    mg = GLM.glm(X, y, Gamma(), InverseLink())
    @test relerr(coef(ms), coef(mg)) < 1e-5
end


# ============================================================
# Residual models
# ============================================================

@testset "SpectralModel API" begin
    X, y, _ = make_regression_data()
    m = SpectralMM.fit(X, y, Expectile(0.25); rank=5, solver=:pcg)

    @test m isa SpectralMM.SpectralModel
    @test m.result isa SpectralMM.MMResult
    @test coef(m) == m.coef
    @test response(m) == y
    @test modelmatrix(m) == X
    @test nobs(m) == size(X, 1)
    @test length(coef(m)) == size(X, 2)
    @test length(fitted(m)) == size(X, 1)
    @test fitted(m) ≈ X * coef(m)
    @test predict(m) ≈ fitted(m)
    @test coefnames(m) == ["x$(j)" for j in 1:size(X, 2)]
    @test predict(m, X[1:10, :]) ≈ X[1:10, :] * coef(m)
    @test m.solver == :pcg
    @test m.rank == 5
    @test m.result.converged
end

@testset "Expectile" begin
    X, y, _ = make_regression_data()

    m = SpectralMM.fit(X, y, Expectile(0.5); rank=5)
    βref = X \ y
    @test relerr(coef(m), βref) < 1e-6
    @test relerr(X * coef(m), X * βref) < 1e-6

    τ = 0.25
    m = SpectralMM.fit(X, y, Expectile(τ); rank=5)
    βref = expectile_reference(X, y, τ)
    @test relerr(coef(m), βref) < 1e-6
    @test relerr(X * coef(m), X * βref) < 1e-6
end

@testset "PseudoHuber" begin
    X, y, _ = make_regression_data()
    δ = 1.345

    m = SpectralMM.fit(X, y, PseudoHuber(δ); rank=5)
    βref = pseudohuber_reference(X, y, δ)

    @test relerr(coef(m), βref) < 1e-6
    @test relerr(X * coef(m), X * βref) < 1e-6
end

@testset "SmoothQuantile" begin
    X, y, _ = make_regression_data()
    τ, ε = 0.25, 0.1

    m = SpectralMM.fit(X, y, SmoothQuantile(τ, ε); rank=5)
    βref = smoothquantile_reference(X, y, τ, ε)

    @test relerr(coef(m), βref) < 1e-6
    @test relerr(X * coef(m), X * βref) < 1e-6
end

@testset "StudentT" begin
    X, y, _ = make_regression_data()
    ν, σ = 4.0, 1.0

    m = SpectralMM.fit(X, y, StudentT(ν, σ); rank=5)
    βref = studentt_reference(X, y, ν, σ)

    @test relerr(coef(m), βref) < 1e-6
    @test relerr(X * coef(m), X * βref) < 1e-6
end


# ============================================================
# GLM API
# ============================================================

@testset "GLM matrix interface" begin
    rng = MersenneTwister(1234)
    n, p = 500, 10
    X = hcat(ones(n), randn(rng, n, p - 1))
    β = randn(rng, p) / sqrt(p)

    η = X * β
    μ = 1 ./ (1 .+ exp.(-η))
    y = Float64.(rand(rng, n) .< μ)

    m = SpectralMM.glm(
        X, y, Bernoulli(), LogitLink();
        rank=5, solver=:pcg
    )

    @test m isa SpectralMM.SpectralGLM
    @test m.result isa SpectralMM.MMResult
    @test coef(m) == m.coef
    @test response(m) == y
    @test modelmatrix(m) == X
    @test nobs(m) == n
    @test length(coef(m)) == p
    @test length(fitted(m)) == n
    @test all((0 .<= fitted(m)) .& (fitted(m) .<= 1))
    @test coefnames(m) == ["x$(j)" for j in 1:p]

    pred = predict(m, X[1:10, :])
    @test length(pred) == 10
    @test all((0 .<= pred) .& (pred .<= 1))

    @test m.solver == :pcg
    @test m.rank == 5
    @test m.result.converged
end

@testset "Formula interface" begin
    rng = MersenneTwister(1234)
    n = 500
    x1, x2 = randn(rng, n), randn(rng, n)

    η = 0.3 .+ 0.7 .* x1 .- 0.4 .* x2
    μ = 1 ./ (1 .+ exp.(-η))
    y = Float64.(rand(rng, n) .< μ)
    df = DataFrame(y=y, x1=x1, x2=x2)

    m = SpectralMM.glm(
        @formula(y ~ x1 + x2), df,
        Bernoulli(), LogitLink();
        rank=2
    )

    @test m isa SpectralMM.SpectralGLM
    @test coefnames(m) == ["(Intercept)", "x1", "x2"]
    @test length(coef(m)) == 3
    @test nobs(m) == n

    newdata = DataFrame(x1=[0.1, -0.2], x2=[0.3, 0.5])
    pred = predict(m, newdata)

    @test length(pred) == 2
    @test all((0 .<= pred) .& (pred .<= 1))
end

@testset "Categorical formula" begin
    rng = MersenneTwister(1234)
    n = 300

    df = DataFrame(
        y=Float64.(rand(rng, n) .< 0.5),
        x1=randn(rng, n),
        group=rand(rng, ["A", "B", "C"], n)
    )

    m = SpectralMM.glm(
        @formula(y ~ x1 + group), df,
        Bernoulli(), LogitLink();
        rank=3
    )

    names = coefnames(m)
    @test "(Intercept)" in names
    @test "x1" in names
    @test "group: B" in names
    @test "group: C" in names

    newdata = DataFrame(x1=[0.0, 1.0], group=["A", "C"])
    @test length(predict(m, newdata)) == 2
end


# ============================================================
# Options and solvers
# ============================================================

@testset "Default spectral rank" begin
    rng = MersenneTwister(1234)

    X, y = randn(rng, 200, 8), randn(rng, 200)
    @test SpectralMM.fit(X, y, Expectile(0.5)).rank == 7

    X, y = randn(rng, 200, 25), randn(rng, 200)
    @test SpectralMM.fit(X, y, Expectile(0.5)).rank == 10
end

@testset "Inner solvers" begin
    X, y, _ = make_regression_data(n=300, p=15)

    m1 = SpectralMM.fit(X, y, Expectile(0.5); rank=5, solver=:pcg)
    m2 = SpectralMM.fit(
        X, y, Expectile(0.5);
        rank=5, solver=:mm, inner_maxiter=300
    )

    @test relerr(coef(m1), coef(m2)) < 1e-5
end


# ============================================================
# Invalid inputs
# ============================================================

@testset "Invalid inputs" begin
    rng = MersenneTwister(1234)
    X, y = randn(rng, 100, 10), randn(rng, 100)
    f = Expectile(0.5)

    @test_throws DimensionMismatch SpectralMM.fit(X, y[1:99], f)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; rank=0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; rank=10)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; floor=0.0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; ridge=-1e-6)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; solver=:invalid)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; eta_max=1.0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; correction_tol=-1.0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; resid_tol=0.0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; gtol=0.0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; relgtol=0.0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; maxiter=0)
    @test_throws ArgumentError SpectralMM.fit(X, y, f; inner_maxiter=0)
end
