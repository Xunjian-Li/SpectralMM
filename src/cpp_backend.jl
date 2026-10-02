module _CppBackend
using Libdl, SparseArrays, LinearAlgebra, Printf
import Distributions
include("inference_common.jl")
export fit, predict, FittedModel, stderror, vcov, confint
include("iteration_log.jl")
include("../deps/build_support.jl")
struct MatrixView
    n::Int64; p::Int64; nnz::Int64; storage::Int32
    values::Ptr{Cdouble}; indptr::Ptr{Int64}; indices::Ptr{Int64}
end
struct Options
    family::Int32; nesterov::Int32
    rank::Int64; maxiter::Int64; inner_maxiter::Int64; krylovdim::Int64
    ridge::Cdouble; floor::Cdouble; gtol::Cdouble; relgtol::Cdouble
    eta_max::Cdouble; correction_tol::Cdouble; resid_tol::Cdouble
end
struct FamilyOptions
    theta::Cdouble; tau::Cdouble; smoothing::Cdouble; delta::Cdouble
    nu::Cdouble; sigma::Cdouble; power::Cdouble
    trials::Ptr{Cdouble}; trials_count::Int64
end
const FAMILIES = (:gaussian, :bernoulli, :probit, :poisson, :gamma, :negative_binomial,
    :smooth_quantile, :expectile, :pseudo_huber, :student_t, :gaussian_log,
    :gamma_inverse, :tweedie, :binomial)
const FAMILY_PARAMS = Dict(:negative_binomial=>(:theta,), :smooth_quantile=>(:tau,:smoothing),
    :expectile=>(:tau,), :pseudo_huber=>(:delta,), :student_t=>(:nu,:sigma),
    :tweedie=>(:power,), :binomial=>(:trials,))
struct SolverOptions
    solver::Int32; jacobi::Int32; rtol::Cdouble; atol::Cdouble
end
struct StopOptions
    accept_negligible::Int32; accept_stalled::Int32
    negligible_step_tol::Cdouble; step_reltol::Cdouble; stalled_relgtol::Cdouble
end
struct Info
    iterations::Int64; inner_iterations::Int64; restarts::Int64; corrections::Int64
    converged::Int32; termination::Int32
    loss::Cdouble; gradnorm::Cdouble; relgradnorm::Cdouble; eigresidual::Cdouble
end
struct InferenceInfo
    cov_type::Int32; estimated_dispersion::Int32
    df_resid::Cdouble; dispersion::Cdouble; rcond::Cdouble
end
struct TraceDetail
    iteration::Int64; inner_iterations::Int64; restarts::Int64; corrections::Int64; inner::Int64
    loss::Cdouble; gradnorm::Cdouble; relgradnorm::Cdouble; inner_residual::Cdouble; eigresidual::Cdouble; step::Cdouble
    spectrum::Int32
end
struct FittedModel
    coef::Vector{Float64}
    family::Symbol
    family_options::Dict{Symbol,Any}
    info::NamedTuple
    intercept::Bool
    inference::NamedTuple
    trace::Union{Nothing,Vector{NamedTuple}}
end
Base.getproperty(m::FittedModel, name::Symbol) = name === :backend ? :cpp : getfield(m,name)
Base.propertynames(m::FittedModel, private::Bool=false) = (fieldnames(typeof(m))..., :backend)
const _cached_library_path = Ref("")
function library_path()
    if haskey(ENV, "SPECTRALMM_LIBRARY")
        return abspath(ENV["SPECTRALMM_LIBRARY"])
    end
    lock(_library_lock) do
        isempty(_cached_library_path[]) && (_cached_library_path[] = BuildSupport.ensure_library())
        _cached_library_path[]
    end
end

# Keep each validated library loaded for the process lifetime. Repeated dlopen/
# dlclose can dominate small fits. Key by path so an explicit library override
# can select a different build without invalidating active function pointers.
struct LibraryAPI
    handle::Ptr{Cvoid}
    fit::Ptr{Cvoid}
    infer::Ptr{Cvoid}
    defaults::Options
end
const _library_handles = Dict{String,LibraryAPI}()
const _library_lock = ReentrantLock()
function _library_api()
    path = abspath(library_path())
    lock(_library_lock) do
        get!(_library_handles, path) do
            handle = Libdl.dlopen(path)
            try
                ccall(Libdl.dlsym(handle, :smm_abi_version), Int32, ()) == 1 || error("native ABI mismatch")
                defaults = Ref{Options}()
                ccall(Libdl.dlsym(handle, :smm_default_options), Cvoid, (Ref{Options},), defaults)
                LibraryAPI(handle, Libdl.dlsym(handle, :smm_fit_logged),
                           Libdl.dlsym(handle, :smm_infer), defaults[])
            catch
                Libdl.dlclose(handle)
                rethrow()
            end
        end
    end
end

"""Fit a real dense or SparseMatrixCSC matrix using the shared C++ core.
All fourteen family identifiers are supported; fixed parameters use family_options.
An intercept is fitted by default. Use intercept=false for an explicit design matrix.
beta0/coef include the intercept first. Ridge excludes it by default.
solver accepts :pcg/:spectral, :mm, :cho/:cholesky, :cg, :cgls, :crls, :lsqr, :lsmr.
"""
function fit(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
             family=:gaussian, family_options=NamedTuple(), beta0=nothing, intercept::Bool=true, penalize_intercept::Bool=false, accept_negligible=true, accept_stalled=true,
             negligible_step_tol=1e-14, step_reltol=sqrt(eps(Float64)), stalled_relgtol=1e-4,
             inference=:auto, inference_max_p=50, cov_type=:auto, level=.95,
             use_t=nothing, dispersion=nothing, solver=:pcg, preconditioner=:jacobi,
             krylov_rtol=1e-4, krylov_atol=1e-8, verbose::Bool=false, trace::Bool=false, kwargs...)
    _check_inference_options(inference,inference_max_p,cov_type,level,use_t,dispersion)
    solver = solver===:spectral ? :pcg : solver===:cholesky ? :cho : solver
    choices=(:cg,:cgls,:crls,:lsqr,:lsmr,:cho,:mm)
    solver===:pcg || solver in choices || throw(ArgumentError("unknown solver"))
    preconditioner in (:jacobi,:none) || throw(ArgumentError("unknown preconditioner"))
    settings=Ref(SolverOptions(solver===:pcg ? 0 : findfirst(==(solver),choices)-1,
                              preconditioner===:jacobi,krylov_rtol,krylov_atol))
    family = Symbol(family)
    family in FAMILIES || throw(ArgumentError("unsupported family: $family"))
    parameters = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in pairs(family_options))
    allowed = get(FAMILY_PARAMS, family, ())
    all(k -> k in allowed, keys(parameters)) || throw(ArgumentError("unknown parameter for $family"))
    trials = Float64[]
    if haskey(parameters, :trials)
        value = parameters[:trials]
        value isa Real || value isa AbstractVector{<:Real} || throw(ArgumentError("trials must be a real scalar or vector"))
        trials = value isa Real ? [Float64(value)] : Vector{Float64}(value)
        isempty(trials) && throw(ArgumentError("trials must not be empty"))
        parameters[:trials] = copy(trials)
    end
    fp = FamilyOptions(get(parameters,:theta,1.),get(parameters,:tau,.5),get(parameters,:smoothing,.1),
        get(parameters,:delta,1.345),get(parameters,:nu,4.),get(parameters,:sigma,1.),
        get(parameters,:power,1.5),isempty(trials) ? C_NULL : pointer(trials),length(trials))
    n, d = size(X)
    p = d + intercept
    if intercept && n>0 && any(j -> !iszero(X[1,j]) && all(==(X[1,j]), view(X,:,j)), 1:d)
        @warn "X contains a nonzero constant column; remove it or use intercept=false to avoid a redundant intercept"
    end
    length(y) == n || throw(DimensionMismatch("y must have length n"))
    yy = y isa Vector{Float64} ? y : Vector{Float64}(y)
    bb = beta0 === nothing ? Float64[] : (beta0 isa Vector{Float64} ? beta0 : Vector{Float64}(beta0))
    beta0 === nothing || length(bb) == p || throw(DimensionMismatch("beta0 must have length p"))
    if X isa SparseMatrixCSC
        xx = X isa SparseMatrixCSC{Float64,Int64} ? X : SparseMatrixCSC{Float64,Int64}(X)
        values = nonzeros(xx)
        indptr = xx.colptr .- Int64(1)
        indices = rowvals(xx) .- Int64(1)
        storage = Int32(1)
    else
        xx = X isa Matrix{Float64} ? X : Matrix{Float64}(X)
        values = vec(xx)
        indptr = Int64[]; indices = Int64[]; storage = Int32(0)
    end
    coef = zeros(p); info = Ref{Info}(); errorbuf = zeros(UInt8,1024)
    api = _library_api()
    begin
        names = fieldnames(Options)
        for key in keys(kwargs)
            key in names && key != :family || throw(ArgumentError("unknown option: $key"))
        end
        vals = map(names) do key
            key == :family ? Int32(findfirst(==(family), FAMILIES)-1) : get(kwargs,key,getfield(api.defaults,key))
        end
        opts = Options(vals...)
        stops = StopOptions(accept_negligible,accept_stalled,negligible_step_tol,step_reltol,stalled_relgtol)
        opts.maxiter>0 || throw(ArgumentError("maxiter must be positive"))
        capture=verbose || trace
        rows=Vector{TraceDetail}(undef,capture ? opts.maxiter+1 : 0)
        rowcount=Ref{Int64}(0)
        GC.@preserve values indptr indices yy bb coef errorbuf settings rows trials begin
            desc = MatrixView(n,d,storage==1 ? length(values) : 0,storage,
                              pointer(values),pointer(indptr),pointer(indices))
            status = ccall(api.fit, Int32,
                (Ref{MatrixView},Ptr{Cdouble},Ptr{Cdouble},Ref{Options},Ref{FamilyOptions},Ptr{Cvoid},Ref{StopOptions},Ptr{Cdouble},Ref{Info},Ptr{Cvoid},Int64,Ptr{Int64},Ptr{UInt8},Csize_t,Int32,Int32),
                desc, yy, beta0===nothing ? C_NULL : pointer(bb), opts, fp, solver===:pcg ? C_NULL : Base.unsafe_convert(Ptr{SolverOptions},settings), stops, coef, info, capture ? pointer(rows) : C_NULL, length(rows), rowcount, errorbuf, length(errorbuf), intercept, penalize_intercept)
            status == 0 || throw(ArgumentError(unsafe_string(pointer(errorbuf))))
        end
        fields = fieldnames(Info)
        diagnostics = NamedTuple{fields}(map(k -> k==:converged ? Bool(getfield(info[],k)) : getfield(info[],k), fields))
        reasons = (:gradient,:maxiter,:line_search_failed,:inner_breakdown,:negligible_step,:stalled_step)
        diagnostics = merge(diagnostics, (solver=solver, rank=opts.rank, gradient_converged=info[].gradnorm<=opts.gtol || info[].relgradnorm<=opts.relgtol,
            termination_reason=reasons[info[].termination+1]))
        states=(:none,:initial,:reuse,:correct,:restart,:correct_fail,:restart_correct,:restart_fail)
        history=NamedTuple[]
        if capture
            for row in @view rows[1:rowcount[]]
                fields_row=NamedTuple{fieldnames(TraceDetail)}(Tuple(getfield(row,k) for k in fieldnames(TraceDetail)))
                push!(history,merge(fields_row,(spectrum=states[row.spectrum+1],)))
            end
        end
        if verbose
            rank=opts.rank==0 ? (p<20 ? p-1 : 10) : opts.rank
            print_trace_header(solver,solver in (:pcg,:mm) ? rank : nothing)
            for r in history
                print_iteration(r.iteration,r.loss,r.relgradnorm,r.inner,r.inner_residual,r.eigresidual,r.step,r.spectrum)
            end
            if diagnostics.gradient_converged
                print_convergence(diagnostics.iterations,diagnostics.loss,diagnostics.relgradnorm)
            else
                labels=("gradient","maximum iterations reached","line search failed","inner breakdown","negligible step","stalled near tolerance")
                print_termination(labels[diagnostics.termination+1],diagnostics.iterations,diagnostics.loss,diagnostics.relgradnorm)
            end
        end
        gate=_inference_gate(inference,inference_max_p,p,n,opts.ridge,diagnostics.gradient_converged)
        if gate===nothing
            covariance=Matrix{Float64}(undef,p,p); meta=Ref{InferenceInfo}()
            GC.@preserve values indptr indices yy coef covariance errorbuf trials begin
                status=ccall(api.infer,Int32,
                    (Ref{MatrixView},Ptr{Cdouble},Ptr{Cdouble},Ref{Options},Ref{FamilyOptions},Int32,Int32,Cdouble,
                     Ptr{Cdouble},Ref{InferenceInfo},Ptr{UInt8},Csize_t),
                    desc,yy,coef,opts,fp,intercept,findfirst(==(cov_type),(:auto,:model,:sandwich))-1,
                    dispersion===nothing ? NaN : dispersion,covariance,meta,errorbuf,length(errorbuf))
                if status!=0
                    gate=_inference_unavailable(unsafe_string(pointer(errorbuf)))
                else
                    m=meta[]
                    gate=_wald_result(coef,covariance,m.cov_type==1 ? :model : :sandwich,
                        Bool(m.estimated_dispersion),m.df_resid,m.dispersion,m.rcond,level,use_t)
                end
            end
        end
        return FittedModel(coef,family,parameters,diagnostics,intercept,_finish_inference(gate,inference),trace ? history : nothing)
    end
end
function predict(m::FittedModel, X::AbstractMatrix; trials=nothing)
    size(X,2) == length(m.coef)-m.intercept || throw(DimensionMismatch("prediction feature count differs from training"))
    eta = m.intercept ? X*view(m.coef,2:length(m.coef)) .+ m.coef[1] : X*m.coef
    m.family in (:gaussian,:smooth_quantile,:expectile,:pseudo_huber,:student_t) && return eta
    m.family == :gamma_inverse && return 1 ./ max.(eta,sqrt(eps(Float64)))
    if m.family == :probit
        return Distributions.cdf.(Ref(Distributions.Normal()),eta)
    end
    if m.family in (:bernoulli,:binomial)
        mu = map(eta) do e
            e >= 0 ? inv(1+exp(-e)) : (z=exp(e); z/(1+z))
        end
        if m.family == :binomial
            counts = trials===nothing ? get(m.family_options,:trials,[1.]) : trials
            counts isa Real || counts isa AbstractVector{<:Real} || throw(ArgumentError("prediction trials must be real"))
            values = counts isa Real ? [counts] : counts
            length(values) in (1,length(mu)) || throw(DimensionMismatch("prediction trials must be scalar or length n"))
            all(v->isfinite(v) && v>0,values) || throw(ArgumentError("prediction trials must be finite and positive"))
            return mu .* values
        end
        return mu
    end
    exp.(min.(eta,700.))
end

stderror(m::FittedModel) = _require_inference(m).std_error
vcov(m::FittedModel) = _require_inference(m).covariance
confint(m::FittedModel) = _require_inference(m).conf_int

function Base.show(io::IO, ::MIME"text/plain", m::FittedModel)
    println(io,"SpectralMM (",m.family,"); parameters=",length(m.coef),"; converged=",m.info.converged)
    println(io,"Backend: C++; Solver: ",uppercase(string(m.info.solver)))
    r=m.inference
    if r.status!="ok"
        print(io,"Inference ",r.status,": ",r.reason)
        return
    end
    println(io,"Covariance: ",r.cov_type,"; reference: ",r.statistic_type)
    @printf(io,"%-18s %12s %12s %12s %12s\n","Term","Estimate","Std. Error","Statistic","p-value")
    for j in 1:min(20,length(m.coef))
        name=m.intercept && j==1 ? "(Intercept)" : "x$(j-m.intercept)"
        @printf(io,"%-18s %12.5g %12.5g %12.5g %12.5g\n",name,m.coef[j],r.std_error[j],r.statistic[j],r.p_value[j])
    end
    length(m.coef)>20 && print(io,"Further rows are available through model.inference.")
end

function __init__()
    empty!(_library_handles)
    _cached_library_path[] = ""
end

end
