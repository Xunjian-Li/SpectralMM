# Unified IRLS-Krylov solver.
# Dependencies:
#   using LinearAlgebra, Printf, Krylov, LinearOperators
#
# Expected package helpers:
#   GLMFamily, BernoulliLogit, check_y, logloss_xb, grad_weights_xb!,
#   work_y!, finite_all, finite_max, finite_min
#
# Design:
#   :cg                  -> matrix-free H = X'WX + ridge*D
#   :cgls/:crls/:lsqr/:lsmr -> matrix-free A = sqrt(W)X
# All solvers optionally use the same diagonal/Jacobi information.
# For LS solvers this is implemented as matrix-free right column scaling:
#       s = D^{-1/2} u,  Atilde = A D^{-1/2},
# where D = diag(X'WX + ridge*I).
# No Gram matrix or weighted design matrix is materialized.

using SparseArrays

mutable struct IRLSKrylovResult{T}
    beta::Vector{T}
    losses::Vector{T}
    gradnorms::Vector{T}
    relgradnorms::Vector{T}
    inner_iters::Vector{Int}
    stepsizes::Vector{T}
    iters::Int
    converged::Bool
    solver::Symbol
end

mutable struct WLSHessianOp{T,TX<:AbstractMatrix{T}} <: AbstractMatrix{T}
    X::TX
    w::Vector{T}
    ridge::T
    penalize_intercept::Bool
    Xv::Vector{T}
    WXv::Vector{T}
end
Base.size(H::WLSHessianOp) = (size(H.X,2), size(H.X,2))
Base.eltype(::Type{WLSHessianOp{T,TX}}) where {T,TX} = T
Base.eltype(::WLSHessianOp{T}) where {T} = T
function LinearAlgebra.mul!(out::AbstractVector{T}, H::WLSHessianOp{T}, v::AbstractVector{T}) where {T}
    mul!(H.Xv, H.X, v)
    @inbounds @simd for i in eachindex(H.Xv); H.WXv[i] = H.w[i] * H.Xv[i]; end
    mul!(out, transpose(H.X), H.WXv)
    if H.ridge > zero(T)
        if H.penalize_intercept
            @inbounds @simd for j in eachindex(v); out[j] += H.ridge * v[j]; end
        else
            @inbounds @simd for j in 2:length(v); out[j] += H.ridge * v[j]; end
        end
    end
    return out
end
function Base.:*(H::WLSHessianOp{T}, v::AbstractVector{T}) where {T}
    out = similar(v); mul!(out, H, v); out
end

# A = sqrt(W)X. For unpenalized intercept, ridge is handled by an augmented
# operator below rather than Krylov's scalar λ.
mutable struct WeightedDesignOp{T,TX<:AbstractMatrix{T}} <: AbstractMatrix{T}
    X::TX
    sqrtw::Vector{T}
    tmpm::Vector{T}
end
Base.size(A::WeightedDesignOp) = size(A.X)
Base.eltype(::Type{WeightedDesignOp{T,TX}}) where {T,TX} = T
Base.eltype(::WeightedDesignOp{T}) where {T} = T
function LinearAlgebra.mul!(out::AbstractVector{T}, A::WeightedDesignOp{T}, v::AbstractVector{T}) where {T}
    mul!(out, A.X, v)
    @inbounds @simd for i in eachindex(out); out[i] *= A.sqrtw[i]; end
    return out
end
function LinearAlgebra.mul!(out::AbstractVector{T}, At::Adjoint{T,<:WeightedDesignOp{T}}, u::AbstractVector{T}) where {T}
    A = parent(At)
    @inbounds @simd for i in eachindex(u); A.tmpm[i] = A.sqrtw[i] * u[i]; end
    mul!(out, transpose(A.X), A.tmpm)
    return out
end
Base.:*(A::WeightedDesignOp{T}, v::AbstractVector{T}) where {T} = (out=zeros(T,size(A,1)); mul!(out,A,v); out)
Base.:*(At::Adjoint{T,<:WeightedDesignOp{T}}, u::AbstractVector{T}) where {T} = (out=zeros(T,size(parent(At),2)); mul!(out,At,u); out)


# Right-scaled operator Atilde = A*R, where
# R = diag(invscale), invscale[j] = 1/sqrt(diag(H)[j]).
# The LS solver works in u-coordinates and the Newton/IRLS correction is s = R*u.
mutable struct RightScaledOp{T,TA<:AbstractMatrix{T}} <: AbstractMatrix{T}
    A::TA
    invscale::Vector{T}
    tmpn::Vector{T}
end
Base.size(R::RightScaledOp) = size(R.A)
Base.eltype(::Type{RightScaledOp{T,TA}}) where {T,TA} = T
Base.eltype(::RightScaledOp{T}) where {T} = T
function LinearAlgebra.mul!(out::AbstractVector{T}, R::RightScaledOp{T}, u::AbstractVector{T}) where {T}
    @inbounds @simd for j in eachindex(u); R.tmpn[j] = R.invscale[j] * u[j]; end
    mul!(out, R.A, R.tmpn)
    return out
end
function LinearAlgebra.mul!(out::AbstractVector{T}, Rt::Adjoint{T,<:RightScaledOp{T}}, v::AbstractVector{T}) where {T}
    R = parent(Rt)
    mul!(out, adjoint(R.A), v)
    @inbounds @simd for j in eachindex(out); out[j] *= R.invscale[j]; end
    return out
end
Base.:*(R::RightScaledOp{T}, u::AbstractVector{T}) where {T} = (out=zeros(T,size(R,1)); mul!(out,R,u); out)
Base.:*(Rt::Adjoint{T,<:RightScaledOp{T}}, v::AbstractVector{T}) where {T} = (out=zeros(T,size(parent(Rt),2)); mul!(out,Rt,v); out)

# Augmented operator [sqrt(W)X; sqrt(ridge)D], needed only when the intercept
# is not penalized. If all coefficients are penalized, LS solvers use the
# native Tikhonov parameter instead.
mutable struct AugmentedWLSOp{T,TA<:WeightedDesignOp{T}} <: AbstractMatrix{T}
    A::TA
    sqrtridge::T
    penalize_intercept::Bool
    tmpn::Vector{T}
end
Base.size(B::AugmentedWLSOp) = (size(B.A,1)+size(B.A,2), size(B.A,2))
Base.eltype(::Type{AugmentedWLSOp{T,TA}}) where {T,TA} = T
Base.eltype(::AugmentedWLSOp{T}) where {T} = T
function LinearAlgebra.mul!(out::AbstractVector{T}, B::AugmentedWLSOp{T}, v::AbstractVector{T}) where {T}
    m, p = size(B.A)
    mul!(view(out,1:m), B.A, v)
    tail = view(out,m+1:m+p)
    @. tail = B.sqrtridge * v
    !B.penalize_intercept && (tail[1] = zero(T))
    return out
end
function LinearAlgebra.mul!(out::AbstractVector{T}, Bt::Adjoint{T,<:AugmentedWLSOp{T}}, u::AbstractVector{T}) where {T}
    B = parent(Bt); m, p = size(B.A)
    mul!(out, adjoint(B.A), view(u,1:m))
    tail = view(u,m+1:m+p)
    if B.penalize_intercept
        @inbounds @simd for j in 1:p; out[j] += B.sqrtridge * tail[j]; end
    else
        @inbounds @simd for j in 2:p; out[j] += B.sqrtridge * tail[j]; end
    end
    return out
end
Base.:*(B::AugmentedWLSOp{T}, v::AbstractVector{T}) where {T} = (out=zeros(T,size(B,1)); mul!(out,B,v); out)
Base.:*(Bt::Adjoint{T,<:AugmentedWLSOp{T}}, u::AbstractVector{T}) where {T} = (out=zeros(T,size(parent(Bt),2)); mul!(out,Bt,u); out)

function build_wls_diag!(d::AbstractVector{T}, X::AbstractMatrix{T}, w::AbstractVector{T};
                         ridge::T=zero(T), penalize_intercept::Bool=true) where {T}
    fill!(d, zero(T))
    @inbounds for j in axes(X,2)
        a = zero(T)
        @simd for i in axes(X,1); a += w[i] * abs2(X[i,j]); end
        d[j] = a
    end
    if ridge > zero(T)
        j0 = penalize_intercept ? 1 : 2
        @inbounds for j in j0:length(d); d[j] += ridge; end
    end
    @inbounds @simd for j in eachindex(d); d[j] = max(d[j], eps(T)); end
    return d
end

function build_wls_diag!(d::AbstractVector{T}, X::SparseMatrixCSC{T}, w::AbstractVector{T};
                         ridge::T=zero(T), penalize_intercept::Bool=true) where {T}
    fill!(d, zero(T))
    rows, vals = rowvals(X), nonzeros(X)

    @inbounds for j in axes(X, 2)
        for k in nzrange(X, j)
            d[j] += w[rows[k]] * abs2(vals[k])
        end
    end

    if ridge > zero(T)
        @views d[penalize_intercept ? 1 : 2:end] .+= ridge
    end

    @. d = max(d, eps(T))
    return d
end

function irls_krylov(
    X::AbstractMatrix{T}, y::AbstractVector{T};
    family::IRLSFamily=BernoulliLogit(), solver::Symbol=:cg, ridge::T=T(1e-6),
    beta0::Union{Nothing,AbstractVector{T}}=nothing,
    maxiter::Int=100, gtol::T=T(1e-7), relgtol::T=T(1e-8), w_floor::T=T(1e-12),
    krylov_reltol::T=T(1e-4), krylov_abstol::T=T(1e-8), krylov_maxiter::Int=200,
    preconditioner::Symbol=:jacobi, warmstart::Bool=false,
    line_search::Bool=true, alpha_min::T=T(1e-10), backtrack_factor::T=T(0.5),
    c_armijo::T=T(1e-4), penalize_intercept::Bool=true,
    outer_nesterov::Bool=false, restart_outer_on_line_search_failed::Bool=true,
    max_line_search_failures::Int=2, step_reltol::T=sqrt(eps(T)),
    stalled_relgtol::T=sqrt(relgtol), accept_stalled_as_converged::Bool=true,
    dispersion=nothing, verbose::Bool=true
) where {T<:Real}

    solver in (:cg,:cgls,:crls,:lsqr,:lsmr) ||
        throw(ArgumentError("solver must be :cg, :cgls, :crls, :lsqr, or :lsmr"))
    preconditioner in (:none,:jacobi) ||
        throw(ArgumentError("preconditioner must be :none or :jacobi"))
    m, p = size(X)
    length(y) == m || throw(DimensionMismatch("length(y) must equal size(X,1)"))
    check_y(family, y)

    β0 = beta0 === nothing ? zeros(T,p) : Vector{T}(beta0)
    length(β0) == p || throw(DimensionMismatch("length(beta0) must equal size(X,2)"))

    β_base, β_prev, β_eval, β_trial = copy(β0), copy(β0), similar(β0), similar(β0)
    Xβ_base, Xβ_prev, Xβ, Xβ_new, Xs = zeros(T,m), zeros(T,m), zeros(T,m), zeros(T,m), zeros(T,m)
    mul!(Xβ_base,X,β_base); copyto!(Xβ_prev,Xβ_base)

    μ, w, sqrtw, z, wz, grad_resid = zeros(T,m), zeros(T,m), zeros(T,m), zeros(T,m), zeros(T,m), zeros(T,m)
    g, rhs, s, s_prev, diagH = zeros(T,p), zeros(T,p), zeros(T,p), zeros(T,p), zeros(T,p)
    invscale, u0 = ones(T,p), zeros(T,p)
    Xv, WXv, lstmpm, augtmpn, scaletmpn = zeros(T,m), zeros(T,m), zeros(T,m), zeros(T,p), zeros(T,p)
    H = WLSHessianOp(X,w,ridge,penalize_intercept,Xv,WXv)
    A = WeightedDesignOp(X,sqrtw,lstmpm)
    B = AugmentedWLSOp(A,sqrt(max(ridge,zero(T))),penalize_intercept,augtmpn)
    AR = RightScaledOp(A,invscale,scaletmpn)
    BR = RightScaledOp(B,invscale,scaletmpn)
    M = Diagonal(diagH)

    losses, gradnorms = Vector{T}(undef,maxiter), Vector{T}(undef,maxiter)
    relgradnorms, inner_iters = Vector{T}(undef,maxiter), Vector{Int}(undef,maxiter)
    stepsizes = Vector{T}(undef,maxiter)
    t_outer, grad_scale, base_loss = one(T), one(T), T(Inf)
    line_search_failures, converged = 0, false

    logger=verbose ? IterationLog(X,y,family,ridge,penalize_intercept,w_floor,solver;dispersion) : nothing
    logger===nothing || log_point!(logger,0,β_base,Xβ_base)
    termination_reason="maximum iterations reached"

    function finish(iter,nsteps,ok)
        resize!(losses,iter); resize!(gradnorms,iter); resize!(relgradnorms,iter)
        resize!(inner_iters,nsteps); resize!(stepsizes,nsteps)
        # Preserve the initial scale and append the returned endpoint when it
        # differs from the most recent evaluation point (e.g. iteration limit).
        if β_base != β_eval
            grad_weights_xb!(family,g,μ,w,grad_resid,X,Xβ_base,y,β_base;
                ridge=ridge,penalize_intercept=penalize_intercept,w_floor=w_floor)
            push!(losses,logloss_xb(family,Xβ_base,y,β_base;ridge=ridge,penalize_intercept=penalize_intercept))
            push!(gradnorms,norm(g))
            push!(relgradnorms,norm(g)/grad_scale)
        end
        # A final accepted step may meet tolerance exactly at the iteration limit.
        ok = ok || last(gradnorms)<=gtol || last(relgradnorms)<=relgtol
        logger===nothing || finish_log!(logger,nsteps,β_base,Xβ_base,gtol,relgtol,termination_reason,ok)
        IRLSKrylovResult(copy(β_base),losses,gradnorms,relgradnorms,inner_iters,stepsizes,nsteps,ok,solver)
    end

    for iter in 1:maxiter
        outer_restarted = false
        if outer_nesterov && iter > 1
            θ = (t_outer-one(T))/(t_outer+one(T))
            @. β_eval = β_base + θ*(β_base-β_prev)
            @. Xβ = Xβ_base + θ*(Xβ_base-Xβ_prev)
        else
            copyto!(β_eval,β_base); copyto!(Xβ,Xβ_base)
        end

        f0 = logloss_xb(family,Xβ,y,β_eval; ridge=ridge, penalize_intercept=penalize_intercept)
        if iter == 1
            base_loss = f0
        elseif outer_nesterov && f0 > base_loss
            copyto!(β_eval,β_base); copyto!(Xβ,Xβ_base)
            f0, t_outer = base_loss, one(T)
        end

        grad_weights_xb!(family,g,μ,w,grad_resid,X,Xβ,y,β_eval; ridge=ridge, penalize_intercept=penalize_intercept,w_floor=w_floor)
        gnorm = norm(g)
        iter == 1 && (grad_scale = one(T)+gnorm)
        relgnorm = gnorm/grad_scale
        losses[iter], gradnorms[iter], relgradnorms[iter] = f0, gnorm, relgnorm

        if gnorm <= gtol || relgnorm <= relgtol
            copyto!(β_base,β_eval); copyto!(Xβ_base,Xβ)
            converged = true
            return finish(iter,iter-1,true)
        end

        # All inner methods solve for a correction s around β_eval.
        # CG: (X'WX + λD)s = -g.
        # LS methods use the score RHS and -sqrt(λ)Dβ_eval below,
        # giving the same penalized correction system as CG.
        if preconditioner === :jacobi
            build_wls_diag!(diagH,X,w; ridge=ridge,penalize_intercept=penalize_intercept)
            @inbounds @simd for j in 1:p; invscale[j] = inv(sqrt(diagH[j])); end
        end

        if solver === :cg
            @. rhs = -g
            use_prec = preconditioner === :jacobi
            if warmstart && iter > 1
                snew, stats = use_prec ?
                    Krylov.cg(H,rhs,s_prev; M=M,ldiv=true,atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter) :
                    Krylov.cg(H,rhs,s_prev; atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter)
            else
                snew, stats = use_prec ?
                    Krylov.cg(H,rhs; M=M,ldiv=true,atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter) :
                    Krylov.cg(H,rhs; atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter)
            end
        else
            @inbounds @simd for i in 1:m
                sqrtw[i] = sqrt(w[i])
                wz[i] = -grad_resid[i]/sqrtw[i]
            end
            # Penalization acts on beta_eval+s, not just on the correction s.
            use_aug = ridge > zero(T)
            use_prec = preconditioner === :jacobi
            Als = use_aug ? (use_prec ? BR : B) : (use_prec ? AR : A)
            if use_aug
                @. augtmpn = -sqrt(ridge)*β_eval
                penalize_intercept || (augtmpn[1] = zero(T))
                bls = vcat(wz, augtmpn)
            else
                bls = wz
            end
            λls = zero(T)

            if solver === :lsmr
                unew, stats = Krylov.lsmr(Als,bls; λ=λls,atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter)
            elseif solver === :lsqr
                unew, stats = Krylov.lsqr(Als,bls; λ=λls,atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter)
            elseif solver === :cgls
                unew, stats = Krylov.cgls(Als,bls; λ=λls,atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter)
            else
                unew, stats = Krylov.crls(Als,bls; λ=λls,atol=krylov_abstol,rtol=krylov_reltol,itmax=krylov_maxiter)
            end

            if use_prec
                @inbounds @simd for j in 1:p; s[j] = invscale[j] * unew[j]; end
                snew = s
            else
                snew = unew
            end
        end

        copyto!(s,snew)
        inner_it = stats.niter
        if !finite_all(s)
            termination_reason="nonfinite step"
            return finish(iter,iter-1,false)
        end
        copyto!(s_prev,s)

        inner_residual=NaN
        if logger!==nothing
            mul!(logger.g,H,s)
            logger.g .+= g
            inner_residual=norm(logger.g)/max(gnorm,floatmin(T))
        end
        mul!(Xs,X,s)
        α, fnew, gd = one(T), T(Inf), dot(g,s)
        if line_search
            while α >= alpha_min
                @. β_trial = β_eval + α*s
                @. Xβ_new = Xβ + α*Xs
                invalid = !admissible_eta(family, Xβ_new)
                if invalid; α *= backtrack_factor; continue; end
                fnew = logloss_xb(family,Xβ_new,y,β_trial; ridge=ridge, penalize_intercept=penalize_intercept)
                if !isfinite(fnew); α *= backtrack_factor; continue; end
                ((gd < zero(T) && fnew <= min(f0+c_armijo*α*gd,base_loss)+objective_roundoff(base_loss)) ||
                 (gd >= zero(T) && fnew <= min(f0,base_loss)+objective_roundoff(base_loss))) && break
                α *= backtrack_factor
            end

            if α < alpha_min
                line_search_failures += 1
                termination_reason="line search failed"
                if accept_stalled_as_converged && relgnorm <= stalled_relgtol
                    copyto!(β_base,β_eval); copyto!(Xβ_base,Xβ)
                    inner_iters[iter], stepsizes[iter], converged = inner_it, zero(T), true
                    return finish(iter,iter,true)
                elseif line_search_failures >= max_line_search_failures
                    inner_iters[iter], stepsizes[iter] = inner_it, zero(T)
                    return finish(iter,iter,false)
                elseif outer_nesterov && restart_outer_on_line_search_failed && iter > 1
                    copyto!(β_trial,β_base); copyto!(Xβ_new,Xβ_base)
                    fnew, α, t_outer, outer_restarted = base_loss, zero(T), one(T), true
                    fill!(s_prev,zero(T))
                else
                    return finish(iter,iter-1,false)
                end
            end
        else
            @. β_trial = β_eval + s
            @. Xβ_new = Xβ + Xs
            fnew = logloss_xb(family,Xβ_new,y,β_trial; ridge=ridge, penalize_intercept=penalize_intercept)
        end

        inner_iters[iter], stepsizes[iter] = inner_it, α
        step_norm = abs(α)*norm(s)
        α > zero(T) && (line_search_failures = 0)

        logger===nothing || log_point!(logger,iter,β_trial,Xβ_new;
            inner=inner_it,residual=inner_residual,step=α)

        if accept_stalled_as_converged && relgnorm <= stalled_relgtol &&
           step_norm <= step_reltol*(one(T)+norm(β_eval))
            copyto!(β_base,β_trial); copyto!(Xβ_base,Xβ_new)
            termination_reason="stalled near tolerance"
            converged = true
            return finish(iter,iter,true)
        end

        if outer_nesterov && outer_restarted
            # Base iterate unchanged.
        elseif outer_nesterov
            copyto!(β_prev,β_base); copyto!(Xβ_prev,Xβ_base)
            copyto!(β_base,β_trial); copyto!(Xβ_base,Xβ_new)
            base_loss = fnew
            t_outer = T(0.5)*(one(T)+sqrt(one(T)+T(4)*t_outer^2))
        else
            copyto!(β_base,β_trial); copyto!(Xβ_base,Xβ_new)
            base_loss = fnew
        end

    end

    termination_reason="maximum iterations reached"
    return finish(maxiter,maxiter,converged)
end
