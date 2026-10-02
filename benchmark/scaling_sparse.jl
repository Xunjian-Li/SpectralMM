include("common.jl")
using SparseArrays
using CSV

const SEED = 3124
const T = Float64
const NNZ_PER_ROW = 20

ps = [500, 1000, 2000, 5000]

families = [
    SpectralMM.GaussianIdentity(),
    SpectralMM.BernoulliLogit(),
    SpectralMM.BernoulliProbit(),
    SpectralMM.PoissonLog(),
    SpectralMM.GammaLog(),
    SpectralMM.NegativeBinomialLog(4.0),
    SpectralMM.SmoothQuantile(0.25, 0.1),
    SpectralMM.Expectile(0.25),
    SpectralMM.PseudoHuber(1.0),
    SpectralMM.StudentT(4.0),
]

function sparse_design(rng, n, p; nnz_per_row=NNZ_PER_ROW)
    m = n * nnz_per_row
    I = repeat(1:n, inner=nnz_per_row)
    J = rand(rng, 1:p, m)
    V = randn(rng, T, m)
    sparse(I, J, V, n, p)
end

all_results = DataFrame[]

for family in families
    println("\n", "="^80)
    println("Family: ", nameof(typeof(family)))
    println("="^80)

    for p in ps
        n = 2p
        rng = MersenneTwister(SEED)

        beta_true = randn(rng, T, p) ./ sqrt(T(p))
        beta0_true = T(0.5)

        X = sparse_design(rng, n, p)
        Xfit = hcat(sparse(ones(T, n)), X)

        y, mu, eta = SpectralMM.simulate_y_from_beta(
            X, beta_true, family;
            seed=SEED, beta0=beta0_true,
        )

        density = nnz(X) / (n * p)

        println("\nn = $n, p = $p")
        println("nnz/row ≈ ", round(nnz(X) / n, digits=2),
                ", density = ", round(density, digits=5),
                ", nnz = ", nnz(X))

        spectral = SpectralMM.SpectralOptions{T}(
            k=5, krylovdim=12, rho=1e-6, ridge=0.0,
            resid_tol=5e-1, correction_tol=5e-2,
        )

        inner = SpectralMM.InnerOptions{T}(
            solver=:pcg, maxiter=200,
            eta_max=0.6, forcing_c=10.0,
            forcing_alpha=0.0, nesterov=true,
        )

        outer = SpectralMM.OuterOptions{T}(
            maxiter=1000, gtol=1e-6,
            relgtol=1e-8, nesterov=true,
            verbose=false,
        )

        result = benchmark_problem(
            Xfit, y, family;
            spectral=spectral,
            inner=inner,
            outer=outer,
            ridge=0.0,
            maxiter=1000,
            gtol=1e-6,
            relgtol=1e-8,
            outer_nesterov=true,
            krylov_solvers=(:cgls, :crls, :lsqr, :lsmr),
            samples=5,
            compute_reference=true,
        )

        result[!, :Density] .= density
        result[!, :NNZ] .= nnz(X)

        show(
            result[:, [:Method, :Time_ms, :Outer, :Inner, :RelGrad, :RelErr]];
            allrows=true, allcols=true,
        )
        println()

        push!(all_results, result)
        GC.gc()
    end
end

results = vcat(all_results...)

println("\n", "="^80)
println("ALL SPARSE RESULTS")
println("="^80)

show(
    results[:, [
        :Family, :n, :p, :Density, :NNZ,
        :Method, :Time_ms, :Outer, :Inner, :RelGrad, :RelErr,
    ]];
    allrows=true, allcols=true,
)

println()
CSV.write("benchmark/scaling_sparse_results.csv", results)