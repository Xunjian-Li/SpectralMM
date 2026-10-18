# Pure-Julia benchmark: one loop for dense and sparse inputs.
isdefined(@__MODULE__, :benchmark_problem) || include("common.jl")
using CSV, SparseArrays

function julia_cases()
    [SpectralMM.GaussianIdentity(), SpectralMM.BernoulliLogit(),
     SpectralMM.BernoulliProbit(), SpectralMM.PoissonLog(), SpectralMM.GammaLog(),
     SpectralMM.NegativeBinomialLog(4.), SpectralMM.SmoothQuantile(.25,.1),
     SpectralMM.Expectile(.25), SpectralMM.PseudoHuber(1.), SpectralMM.StudentT(4.)]
end

function julia_input(p, family, storage, settings)
    n=2p; rng=MersenneTwister(settings.seed)
    truth=randn(rng,p)./sqrt(Float64(p))
    X=if storage=="dense"
        SpectralMM.generate_correlated_design(n,p;rho=.5,seed=1,T=Float64)
    else
        rows=repeat(1:n,inner=settings.nnz_per_row)
        cols=rand(rng,1:p,length(rows)); vals=randn(rng,length(rows))
        sparse(rows,cols,vals,n,p)
    end
    y,_,_=SpectralMM.simulate_y_from_beta(X,truth,family;seed=settings.seed,beta0=.5)
    Xfit=hcat(storage=="dense" ? ones(n) : sparse(ones(n)),X)
    return Xfit,y,storage=="sparse" ? nnz(X) : n*p
end

function run_julia(settings, output, storage; legacy=false, runner=benchmark_problem)
    storage in ("dense","sparse","both") || throw(ArgumentError("storage must be dense, sparse or both"))
    stores=storage=="both" ? ("dense","sparse") : (storage,)
    for store in stores
        sizes=store=="dense" ? settings.dense_sizes : settings.sparse_sizes
        !isempty(sizes) && all(p->p>0 && 1<=settings.rank<settings.krylovdim<=p+1,sizes) ||
            throw(ArgumentError("require nonempty sizes and 1 <= rank < krylovdim <= p+1"))
    end
    settings.samples>0 || throw(ArgumentError("samples must be positive"))
    !legacy && ispath(output) && throw(ArgumentError("choose a new --output directory"))
    configure_benchmark_threads(get(ENV,"SPECTRALMM_BENCH_THREADS",string(settings.threads)))
    mkpath(output)
    if !legacy
        open(joinpath(output,"settings.txt"),"w") do io
            show(io,settings)
            println(io,"\nJulia: ",VERSION,"; Julia threads: ",Threads.nthreads(),"; BLAS threads: ",BLAS.get_num_threads())
            println(io,"BLAS: ",BLAS.get_config())
        end
    end
    spectral=SpectralMM.SpectralOptions{Float64}(k=settings.rank,krylovdim=settings.krylovdim,
        rho=settings.floor,ridge=0.,resid_tol=settings.resid_tol,correction_tol=settings.correction_tol)
    inner=SpectralMM.InnerOptions{Float64}(solver=:pcg,maxiter=settings.inner_maxiter,
        eta_max=settings.eta_max,forcing_c=10.,forcing_alpha=0.,nesterov=settings.inner_nesterov)
    outer=SpectralMM.OuterOptions{Float64}(maxiter=settings.maxiter,gtol=settings.gtol,
        relgtol=settings.relgtol,nesterov=settings.outer_nesterov,accept_stalled=settings.accept_stalled,verbose=false)
    outputs=Dict{String,DataFrame}()
    for store in stores
        rows=DataFrame[]
        sizes=store=="dense" ? settings.dense_sizes : settings.sparse_sizes
        for family in julia_cases(), p in sizes
            X,y,nz=julia_input(p,family,store,settings)
            println("\nFamily: ",nameof(typeof(family)),"; ",store,"; n=",size(X,1),"; predictors=",p)
            result=runner(X,y,family;spectral,inner,outer,ridge=0.,maxiter=settings.maxiter,
                gtol=settings.gtol,relgtol=settings.relgtol,outer_nesterov=settings.outer_nesterov,
                krylov_solvers=(:cgls,:crls,:lsqr,:lsmr),samples=settings.samples,compute_reference=true)
            if store=="sparse"
                result[!,:Density]=fill(nz/(2p*p),nrow(result))
                result[!,:NNZ]=fill(nz,nrow(result))
            end
            show(result;allrows=true,allcols=true);println()
            push!(rows,result);GC.gc()
        end
        outputs[store]=vcat(rows...)
        name=legacy ? (store=="dense" ? "scaling_results.csv" : "scaling_sparse_results.csv") : store*".csv"
        CSV.write(joinpath(output,name),outputs[store])
    end
    return outputs
end

function julia_main(args)
    if "--help" in args
        println("julia --project=benchmark/julia benchmark/run_julia.jl --output benchmark/runs/julia-new [--storage dense|sparse|both] [--settings path]")
        return
    end
    options=Dict("--settings"=>joinpath(@__DIR__,"julia_settings.jl"),"--storage"=>"both")
    iseven(length(args)) || throw(ArgumentError("arguments must be --key value pairs"))
    for i in 1:2:length(args)
        args[i] in ("--settings","--storage","--output") || throw(ArgumentError("unknown argument: $(args[i])"))
        options[args[i]]=args[i+1]
    end
    haskey(options,"--output") || throw(ArgumentError("--output is required"))
    settings=include(abspath(options["--settings"]))
    run_julia(settings,abspath(options["--output"]),options["--storage"])
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && julia_main(ARGS)
