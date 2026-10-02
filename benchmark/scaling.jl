include("common.jl")

using CSV

const SEED = 3124
const T = Float64

ps = [500, 1000, 2000, 5000]
# ps = [5000]

families = [
    # GLM
    SpectralMM.GaussianIdentity(),
    SpectralMM.BernoulliLogit(),
    SpectralMM.BernoulliProbit(),
    SpectralMM.PoissonLog(),
    SpectralMM.GammaLog(),
    SpectralMM.NegativeBinomialLog(4.0),

    # Residual models
    SpectralMM.SmoothQuantile(0.25, 0.1),
    SpectralMM.Expectile(0.25),
    SpectralMM.PseudoHuber(1.0),
    SpectralMM.StudentT(4.0),
]

all_results = DataFrame[]

for family in families
    println()
    println("="^80)
    println("Family: ", string(nameof(typeof(family))))
    println("="^80)

    for p in ps
        n = 2p

        println()
        println("n = $n, p = $p")

        rng = MersenneTwister(SEED)

        beta_true = randn(rng, T, p) ./ sqrt(T(p))
        beta0_true = T(0.5)

        X = SpectralMM.generate_correlated_design(
            n,
            p;
            rho = 0.5,
            seed = 1,
            T = T,
        )

        y, mu, eta = SpectralMM.simulate_y_from_beta(
            X,
            beta_true,
            family;
            seed = SEED,
            beta0 = beta0_true,
        )

        Xfit = hcat(ones(T, n), X)

        # ============================================================
        # Spectral-MM options
        # Restore the settings used in the previous benchmark
        # ============================================================

        spectral = SpectralMM.SpectralOptions{T}(
            k = 5,
            rho = 1e-6,
            krylovdim = 12,
            ridge = 0.0,
            resid_tol = 5e-1,
            correction_tol = 5e-2,
        )

        inner = SpectralMM.InnerOptions{T}(
            solver = :pcg,
            maxiter = 200,
            eta_max = 0.6,
            forcing_c = 10.0,
            forcing_alpha = 0.0,
            nesterov = true,
        )

        outer = SpectralMM.OuterOptions{T}(
            maxiter = 1000,
            gtol = 1e-6,
            relgtol = 1e-8,
            nesterov = true,
            verbose = false,
        )

        # ============================================================
        # Benchmark
        # ============================================================

        result = benchmark_problem(
            Xfit,
            y,
            family;
            spectral = spectral,
            inner = inner,
            outer = outer,
            ridge = 0.0,
            maxiter = 1000,
            gtol = 1e-6,
            relgtol = 1e-8,
            outer_nesterov = true,
            krylov_solvers = (:cgls, :crls, :lsqr, :lsmr),
            samples = 5,
            compute_reference = true,
        )

        show(
            result[:, [
                :Method,
                :Time_ms,
                :Outer,
                :Inner,
                :RelGrad,
                :RelErr,
            ]];
            allrows = true,
            allcols = true,
        )
        println()

        push!(all_results, result)

        GC.gc()
    end
end

results = vcat(all_results...)

println()
println("="^80)
println("ALL RESULTS")
println("="^80)

show(
    results[:, [
        :Family,
        :n,
        :p,
        :Method,
        :Time_ms,
        :Outer,
        :Inner,
        :RelGrad,
        :RelErr,
    ]];
    allrows = true,
    allcols = true,
)

println()

CSV.write("benchmark/scaling_results.csv", results)