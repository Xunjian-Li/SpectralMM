
mutable struct MMResult{T}
    beta::Vector{T}
    losses::Vector{T}
    gradnorms::Vector{T}
    relgradnorms::Vector{T}
    eigresiduals::Vector{T}
    inner_iters::Vector{Int}
    inner_stats::Vector{T}
    outer_steps::Vector{T}
    restart_flags::Vector{Bool}
    correction_failed_flags::Vector{Bool}
    restarts::Int
    iters::Int
    converged::Bool
end

# Spectral-MM core solver
#
# Intended for inclusion in a package module. This file defines option/workspace
# types and the main `spectral_mm` routine. Hot loops are allocation-conscious;
# fitted values are cached to avoid redundant X*v products where possible.
# Options



Base.@kwdef struct SpectralOptions{T<:Real}
    k::Int = 10
    rho::T = T(1e-3)
    ridge::T = T(1e-6)
    resid_tol::T = T(5e-1)

    restart_every::Int = 0
    correction_tol::T = T(5e-2)  # trigger correction when relative projected Hessian change exceeds this

    krylovdim::Union{Nothing,Int} = nothing
    max_krylovdim::Union{Nothing,Int} = nothing
    lanczos_tol::T = T(1e-8)
    lanczos_retries::Int = 3
    nesterov::Bool = true
end

Base.@kwdef struct InnerOptions{T<:Real}
    solver::Symbol = :psd
    maxiter::Int = 100
    # Inner forcing: eta_m = min(eta_max, forcing_c * grad_ratio^forcing_alpha)
    eta_max::T = T(0.9)
    forcing_c::T = T(100.0)
    forcing_alpha::T = T(0.0)
    abstol::T = T(1e-10)
    alpha_min::T = T(1e-10)
    alpha_max::T = T(10)
    shrink::T = T(0.5)
    nesterov::Bool = false
end

Base.@kwdef struct OuterOptions{T<:Real}
    maxiter::Int = 200
    gtol::T = T(1e-7)
    relgtol::T = T(1e-8)
    w_floor::T = T(1e-12)
    safeguard::Bool = true
    nesterov::Bool = true
    restart_on_correction_failure::Bool = true
    step_reltol::T = sqrt(eps(T))
    stalled_relgtol::T = T(1e-4)
    accept_stalled::Bool = true
    verbose::Bool = true
end

# Inner-solver workspace

struct InnerWorkspace{T}
    # Common
    beta_inner::Vector{T}
    Xbeta_inner::Vector{T}
    residual::Vector{T}
    Wr::Vector{T}
    g::Vector{T}
    factor_tmp::Vector{T}

    # PSD
    beta_old::Vector{T}
    beta_y::Vector{T}
    beta_trial::Vector{T}
    Xbeta_old::Vector{T}
    Xbeta_y::Vector{T}
    d::Vector{T}
    Xd::Vector{T}

    # PCG
    delta::Vector{T}
    r_cg::Vector{T}
    z_cg::Vector{T}
    p_cg::Vector{T}
    Hp::Vector{T}
    Xp::Vector{T}
    WXp::Vector{T}
    Xdelta::Vector{T}
end

function InnerWorkspace(::Type{T}, m::Int, p::Int, k::Int) where {T}
    vp() = zeros(T, p)
    vm() = zeros(T, m)

    return InnerWorkspace(
        vp(), vm(), vm(), vm(), vp(), zeros(T, k),
        vp(), vp(), vp(), vm(), vm(), vp(), vm(),
        vp(), vp(), vp(), vp(), vp(), vm(), vm(), vm()
    )
end

# Spectral workspace

mutable struct SpectralWorkspace{T}
    V::Matrix{T}
    XV::Matrix{T}
    lambda::Vector{T}
    Vnew::Matrix{T}
    XVnew::Matrix{T}
    lambda_new::Vector{T}
    HV::Matrix{T}
    WXV::Matrix{T}
    dw::Vector{T}
    dWXV::Matrix{T}

    B::Matrix{T}
    E::Matrix{T}
    A::Matrix{T}
    Vraw::Matrix{T}
    XVraw::Matrix{T}
    Ssmall::Matrix{T}
    d_eigs::Vector{T}
    factor_coeff::Vector{T}
end

function SpectralWorkspace(::Type{T}, m::Int, p::Int, k::Int) where {T}
    r = k + 1

    return SpectralWorkspace(
        zeros(T, p, r),
        zeros(T, m, r),
        zeros(T, r),
        zeros(T, p, r),
        zeros(T, m, r),
        zeros(T, r),
        zeros(T, p, r),
        zeros(T, m, r),
        zeros(T, m),
        zeros(T, m, r),

        zeros(T, r, r),
        zeros(T, r, r),
        zeros(T, r, r),
        zeros(T, p, r),
        zeros(T, m, r),
        zeros(T, r, r),
        zeros(T, k),
        zeros(T, k)
    )
end

# Outer workspace
struct OuterWorkspace{T}
    beta_eval::Vector{T}
    beta_new::Vector{T}
    direction::Vector{T}
    Xbeta::Vector{T}
    Xbeta_base::Vector{T}
    Xbeta_new::Vector{T}
    Xdirection::Vector{T}

    mu::Vector{T}
    mu_new::Vector{T}
    w::Vector{T}
    w_old::Vector{T}
    w_new::Vector{T}
    w_spectral::Vector{T}
    g::Vector{T}
    grad_resid::Vector{T}
    z::Vector{T}
end

function OuterWorkspace(::Type{T}, m::Int, p::Int) where {T}
    vp() = zeros(T, p)
    vm() = zeros(T, m)

    return OuterWorkspace(
        vp(), vp(), vp(), vm(), vm(), vm(), vm(), vm(), vm(),
        vm(), vm(), vm(), vm(), vp(), vm(), vm()
    )
end

# Basic WLS operations
function wls_hess_mul!(
    out::AbstractVector{T},
    Xv::AbstractVector{T},
    WXv::AbstractVector{T},
    X::AbstractMatrix{T},
    w::AbstractVector{T},
    v::AbstractVector{T};
    ridge::T=zero(T)
) where {T<:Real}
    mul!(Xv, X, v)
    @inbounds @simd for i in eachindex(Xv); WXv[i] = w[i] * Xv[i]; end
    mul!(out, transpose(X), WXv)
    if ridge > zero(T)
        @. out += ridge * v
    end
    return out
end

function wls_hess_from_Xv!(
    out::AbstractVector{T},
    WXv::AbstractVector{T},
    X::AbstractMatrix{T},
    w::AbstractVector{T},
    v::AbstractVector{T},
    Xv::AbstractVector{T};
    ridge::T=zero(T)
) where {T<:Real}
    @inbounds @simd for i in eachindex(Xv); WXv[i] = w[i] * Xv[i]; end
    mul!(out, transpose(X), WXv)
    if ridge > zero(T)
        @. out += ridge * v
    end
    return out
end

function wls_grad_from_resid!(
    g::AbstractVector{T},
    Wr::AbstractVector{T},
    X::AbstractMatrix{T},
    w::AbstractVector{T},
    r::AbstractVector{T},
    beta::AbstractVector{T};
    ridge::T=zero(T)
) where {T<:Real}

    @inbounds @simd for i in eachindex(r); Wr[i] = w[i] * r[i]; end
    mul!(g, transpose(X), Wr)
    if ridge > zero(T)
        @. g += ridge * beta
    end
    return g
end

# PSD inner solver
function solve_wls_spectral_psd!(
    beta_out::AbstractVector{T},
    Xbeta_out::AbstractVector{T},
    beta_start::AbstractVector{T},
    Xbeta_start::AbstractVector{T},
    X::AbstractMatrix{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    Vk::AbstractMatrix{T},
    factor_base::T,
    factor_coeff::AbstractVector{T},
    ws::InnerWorkspace{T};
    ridge::T=zero(T),
    maxiter::Int=50,
    forcing::T=T(0.1),
    abstol::T=T(1e-10),
    alpha_min::T=T(1e-10),
    alpha_max::T=T(10),
    nesterov::Bool=false
) where {T<:Real}

    copyto!(beta_out, beta_start)
    copyto!(Xbeta_out, Xbeta_start)

    if nesterov
        copyto!(ws.beta_old, beta_start)
        copyto!(ws.beta_y, beta_start)
        copyto!(ws.Xbeta_old, Xbeta_start)
        copyto!(ws.Xbeta_y, Xbeta_start)
    end

    @. ws.residual = Xbeta_start - z
    wls_grad_from_resid!(ws.g, ws.Wr, X, w, ws.residual, beta_start; ridge=ridge)
    g0 = norm(ws.g)
    g0 <= abstol && return 0, zero(T), false
    t = one(T)
    eta = one(T)

    for iter in 1:maxiter
        gnorm = norm(ws.g)
        !isfinite(gnorm) && return iter - 1, eta, true
        eta = gnorm / g0
        if gnorm <= abstol || eta <= forcing
            return iter - 1, eta, false
        end

        # d = -(M + rho I)^(-1) g
        @timeit SMM_TIMER "inner/inv apply" begin
            inv_apply!(ws.d, ws.factor_tmp, ws.g, Vk, factor_base, factor_coeff)
        end
        @. ws.d = -ws.d
        !finite_all(ws.d) && return iter - 1, eta, true

        # Xd is reused for the exact quadratic line search and updates.
        @timeit SMM_TIMER "inner/X*d" begin
            mul!(ws.Xd, X, ws.d)
        end
        !finite_all(ws.Xd) && return iter - 1, eta, true
        curvature = zero(T)
        @inbounds @simd for i in eachindex(ws.Xd); curvature += w[i] * ws.Xd[i]^2; end
        ridge > zero(T) && (curvature += ridge * dot(ws.d, ws.d))
        slope = dot(ws.g, ws.d)
        alpha = curvature > zero(T) ? -slope / curvature : one(T)
        alpha = isfinite(alpha) ? clamp(alpha, alpha_min, alpha_max) : alpha_min

        # No inner backtracking: exact quadratic step is taken directly.
        if nesterov
            # Save the previous accepted iterate for momentum.
            copyto!(ws.beta_old, beta_out)
            copyto!(ws.Xbeta_old, Xbeta_out)
            @. beta_out = ws.beta_y + alpha * ws.d
            @. Xbeta_out = ws.Xbeta_y + alpha * ws.Xd
            (!finite_all(beta_out) || !finite_all(Xbeta_out)) && return iter - 1, eta, true
            tnew = T(0.5) * (one(T) + sqrt(one(T) + T(4) * t^2))
            theta = (t - one(T)) / tnew
            @. ws.beta_y = beta_out + theta * (beta_out - ws.beta_old)
            @. ws.Xbeta_y = Xbeta_out + theta * (Xbeta_out - ws.Xbeta_old)
            (!finite_all(ws.beta_y) || !finite_all(ws.Xbeta_y)) && return iter, eta, true
            @. ws.residual = ws.Xbeta_y - z
            @timeit SMM_TIMER "inner/gradient refresh" begin
                wls_grad_from_resid!(ws.g, ws.Wr, X, w, ws.residual, ws.beta_y; ridge=ridge)
            end
            t = tnew
        else
            # Fast path: update beta, Xbeta, and gradient recursively.
            @. beta_out += alpha * ws.d
            @. Xbeta_out += alpha * ws.Xd
            (!finite_all(beta_out) || !finite_all(Xbeta_out)) && return iter - 1, eta, true
            @timeit SMM_TIMER "inner/H*d" begin
                wls_hess_from_Xv!(ws.Hp, ws.WXp, X, w, ws.d, ws.Xd; ridge=ridge)
            end
            @. ws.g += alpha * ws.Hp
        end
    end

    # Report the forcing statistic at the point actually returned/evaluated.
    if nesterov
        @. ws.residual = Xbeta_out - z
        wls_grad_from_resid!(ws.g, ws.Wr, X, w, ws.residual, beta_out; ridge=ridge)
    end
    eta = norm(ws.g) / g0
    failed = !isfinite(eta)
    return maxiter, eta, failed
end

# PCG inner solver
function solve_wls_spectral_pcg!(
    beta_out::AbstractVector{T}, Xbeta_out::AbstractVector{T},
    beta_start::AbstractVector{T}, Xbeta_start::AbstractVector{T},
    X::AbstractMatrix{T}, z::AbstractVector{T}, w::AbstractVector{T},
    Vk::AbstractMatrix{T}, factor_base::T, factor_coeff::AbstractVector{T},
    ws::InnerWorkspace{T};
    ridge::T=zero(T), maxiter::Int=50, forcing::T=T(0.1), abstol::T=T(1e-10)
) where {T<:Real}

    @. ws.residual = Xbeta_start - z
    wls_grad_from_resid!(ws.g, ws.Wr, X, w, ws.residual, beta_start; ridge=ridge)
    g0 = norm(ws.g)
    g0 <= abstol && (copyto!(beta_out, beta_start); copyto!(Xbeta_out, Xbeta_start); return 0, zero(T), false)
    fill!(ws.delta, zero(T))
    fill!(ws.Xdelta, zero(T))
    @. ws.r_cg = -ws.g
    inv_apply!(ws.z_cg, ws.factor_tmp, ws.r_cg, Vk, factor_base, factor_coeff)
    !finite_all(ws.z_cg) && return 0, one(T), true
    copyto!(ws.p_cg, ws.z_cg)
    rz = dot(ws.r_cg, ws.z_cg)
    (!isfinite(rz) || rz <= zero(T)) && return 0, one(T), true
    eta = one(T)

    for iter in 1:maxiter
        wls_hess_mul!(ws.Hp, ws.Xp, ws.WXp, X, w, ws.p_cg; ridge=ridge)
        pHp = dot(ws.p_cg, ws.Hp)
        (!isfinite(pHp) || pHp <= zero(T)) && return iter - 1, eta, true
        alpha = rz / pHp
        !isfinite(alpha) && return iter - 1, eta, true
        @. ws.delta += alpha * ws.p_cg
        @. ws.r_cg -= alpha * ws.Hp
        @. ws.Xdelta += alpha * ws.Xp
        rnorm = norm(ws.r_cg)
        !isfinite(rnorm) && return iter, eta, true
        eta = rnorm / g0
        if rnorm <= abstol || eta <= forcing
            @. beta_out = beta_start + ws.delta
            @. Xbeta_out = Xbeta_start + ws.Xdelta
            return iter, eta, false
        end
        inv_apply!(ws.z_cg, ws.factor_tmp, ws.r_cg, Vk, factor_base, factor_coeff)
        !finite_all(ws.z_cg) && return iter, eta, true
        rz_new = dot(ws.r_cg, ws.z_cg)
        (!isfinite(rz_new) || rz_new <= zero(T)) && return iter, eta, true
        beta_cg = rz_new / rz
        @. ws.p_cg = ws.z_cg + beta_cg * ws.p_cg
        rz = rz_new
    end
    @. beta_out = beta_start + ws.delta
    @. Xbeta_out = Xbeta_start + ws.Xdelta
    return maxiter, eta, false
end

# Inner-solver dispatch

function solve_inner!(
    beta_out,
    Xbeta_out,
    beta,
    Xbeta,
    X,
    z,
    w,
    Vk,
    factor_base,
    factor_coeff,
    ws::InnerWorkspace{T},
    opts::InnerOptions{T};
    ridge::T,
    forcing::T
) where {T}
    if opts.solver === :psd
        return solve_wls_spectral_psd!(
            beta_out, Xbeta_out, beta, Xbeta, X, z, w, Vk, factor_base, factor_coeff, ws;
            ridge=ridge, maxiter=opts.maxiter, forcing=forcing, abstol=opts.abstol,
            alpha_min=opts.alpha_min, alpha_max=opts.alpha_max, nesterov=opts.nesterov
        )
    elseif opts.solver === :pcg
        return solve_wls_spectral_pcg!(
            beta_out, Xbeta_out, beta, Xbeta, X, z, w, Vk, factor_base, factor_coeff, ws;
            ridge=ridge, maxiter=opts.maxiter, forcing=forcing, abstol=opts.abstol
        )
    else
        throw(ArgumentError("inner solver must be :psd or :pcg"))
    end
end

# Outer helpers
@inline function nesterov_update(t::T) where {T}
    return T(0.5) * (one(T) + sqrt(one(T) + T(4) * t^2))
end

function extrapolate!(
    out::AbstractVector{T},
    beta::AbstractVector{T},
    beta_prev::AbstractVector{T},
    t::T
) where {T}
    theta = (t - one(T)) / (t + one(T))
    @. out = beta + theta * (beta - beta_prev)
    return out
end

function outer_linesearch!(
    beta_new::AbstractVector{T},
    Xbeta_new::AbstractVector{T},
    beta::AbstractVector{T},
    Xbeta::AbstractVector{T},
    direction::AbstractVector{T},
    Xdirection::AbstractVector{T},
    y::AbstractVector{T},
    family::IRLSFamily,
    fref::T;
    ridge::T,
    alpha_min::T,
    shrink::T
) where {T}
    alpha = one(T)
    while alpha >= alpha_min
        @. beta_new = beta + alpha * direction
        @. Xbeta_new = Xbeta + alpha * Xdirection
        if admissible_eta(family, Xbeta_new)
            fnew = logloss_xb(family, Xbeta_new, y, beta_new; ridge=ridge)
            if isfinite(fnew) && fnew <= fref
                return alpha, fnew
            end
        end
        alpha *= shrink
    end
    return zero(T), fref
end

# Spectral restart
function restart_spectrum!(
    sp::SpectralWorkspace{T},
    X,
    w,
    k::Int,
    iter::Int,
    opts::SpectralOptions{T}
) where {T}
    p = size(X, 2)
    r = k + 1
    kd = something(opts.krylovdim, min(p, max(3r + 20, r + 10)))
    maxkd = something(opts.max_krylovdim, p)
    V, XV, lambda, res, _, ok = checked_lz(X, w, r;
        ridge=opts.ridge, krylovdim=kd, max_krylovdim=maxkd, tol=opts.lanczos_tol,
        resid_tol=opts.resid_tol, seed=iter + 100, max_retries=opts.lanczos_retries, verbose=false)
    copyto!(sp.V, V)
    copyto!(sp.XV, XV)
    copyto!(sp.lambda, lambda)
    return res, ok
end

# Projected Hessian change
function projected_hessian_change!(
    sp::SpectralWorkspace{T},
    w_spectral::AbstractVector{T},
    w_new::AbstractVector{T}
) where {T<:Real}
    @inbounds @simd for i in eachindex(sp.dw)
        sp.dw[i] = w_new[i] - w_spectral[i]
    end
    m, r = size(sp.XV)
    @inbounds for j in 1:r
        @simd for i in 1:m
            sp.dWXV[i, j] = sp.dw[i] * sp.XV[i, j]
        end
    end
    mul!(sp.B, transpose(sp.XV), sp.dWXV)
    denom = max(norm(sp.lambda), sqrt(eps(T)))
    return norm(sp.B) / denom
end

# First-order spectral correction
function correct_spectrum!(
    sp::SpectralWorkspace{T},
    X,
    w_old,
    w_new,
    ridge::T
) where {T}
    res = perturb!(
        sp.Vnew, sp.XVnew, sp.lambda_new, sp.dw, sp.dWXV, sp.B, sp.E, sp.A, sp.Vraw, sp.XVraw, sp.HV, sp.WXV,
        sp.Ssmall, X, w_old, w_new, sp.V, sp.XV, sp.lambda; ridge=ridge)
    return res
end

function accept_spectrum!(
    sp::SpectralWorkspace
)
    copyto!(sp.V, sp.Vnew)
    copyto!(sp.XV, sp.XVnew)
    copyto!(sp.lambda, sp.lambda_new)
    return nothing
end

# Diagnostics

function print_iteration(
    iter,
    solver,
    loss,
    relgrad,
    inner_iter,
    inner_stat,
    eigres,
    step,
    restart
)
    tag = restart ? "  restart" : ""
    if solver === :pcg
        @printf("outer=%4d  PCG  loss=%.6e  rel|g|=%.3e  inner=%3d  eta=%.3e  eigres=%.3e  step=%.3g%s\n",
                iter, loss, relgrad, inner_iter, inner_stat, eigres, step, tag)
    else
        @printf("outer=%4d  PSD  loss=%.6e  rel|g|=%.3e  inner=%3d  eta=%.3e  eigres=%.3e  step=%.3g%s\n",
                iter, loss, relgrad, inner_iter, inner_stat, eigres, step, tag)
    end
end

# ================================================================
# Persistent Lanczos restart support
# Requires LanczosRestartWorkspace and checked_lz! to be defined.
# ================================================================

@inline function lanczos_workspace_capacity(
    krylovdim::Int,
    max_krylovdim::Int,
    max_retries::Int
)
    kd = min(krylovdim, max_krylovdim)
    cap = kd
    for _ in 2:max_retries
        kd >= max_krylovdim && break
        kd = min(max_krylovdim, max(kd + 20, Int(ceil(1.5 * kd))))
        cap = max(cap, kd)
    end
    return cap
end

function restart_spectrum!(
    sp::SpectralWorkspace{T},
    lzws::LanczosRestartWorkspace{T},
    X::AbstractMatrix{T},
    Xt,
    w::AbstractVector{T},
    k::Int,
    iter::Int,
    opts::SpectralOptions{T}
) where {T<:Real}

    p = size(X, 2)
    r = k + 1

    kd = something(
        opts.krylovdim,
        min(p, max(3r + 20, r + 10))
    )
    requested_maxkd = something(opts.max_krylovdim, p)

    # Never ask checked_lz! for more storage than the persistent workspace owns.
    maxkd = min(requested_maxkd, size(lzws.Q, 2))

    V, XV, lambda, res, _, ok = checked_lz!(lzws, X, Xt, w, r; ridge=opts.ridge, krylovdim=kd, max_krylovdim=maxkd,
        tol=opts.lanczos_tol, resid_tol=opts.resid_tol, seed=iter + 100, max_retries=opts.lanczos_retries, verbose=false
    )

    copyto!(sp.V, V)
    copyto!(sp.XV, XV)
    copyto!(sp.lambda, lambda)

    return res, ok
end


# Main solver
function spectral_mm(
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    family::IRLSFamily=BernoulliLogit(),
    beta0::Union{Nothing,AbstractVector{T}}=nothing,
    spectral::SpectralOptions{T}=SpectralOptions{T}(),
    inner::InnerOptions{T}=InnerOptions{T}(),
    outer::OuterOptions{T}=OuterOptions{T}()
) where {T<:Real}
    
    spectral.correction_tol >= zero(T) || throw(ArgumentError("correction_tol must be nonnegative"))
    spectral.restart_every >= 0 || throw(ArgumentError("restart_every must be >= 0"))

    m, p = size(X)
    k = spectral.k
    length(y) == m || throw(DimensionMismatch("length(y) must equal size(X,1)"))
    1 <= k < p || throw(ArgumentError("k must satisfy 1 <= k < p"))
    inner.solver in (:psd, :pcg) || throw(ArgumentError("solver must be :psd or :pcg"))
    zero(T) < inner.shrink < one(T) || throw(ArgumentError("shrink must lie in (0,1)"))
    zero(T) < inner.eta_max < one(T) || throw(ArgumentError("eta_max must lie in (0,1)"))
    inner.forcing_c > zero(T) || throw(ArgumentError("forcing_c must be positive"))
    inner.forcing_alpha >= zero(T) || throw(ArgumentError("forcing_alpha must be nonnegative"))
    zero(T) < inner.alpha_min <= inner.alpha_max || throw(ArgumentError("require 0 < alpha_min <= alpha_max"))
    check_y(family, y)

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------
    local beta, beta_prev, ws, iw, sp, lzws, Xt, fbase
    @timeit SMM_TIMER "initialization" begin
        beta = beta0 === nothing ? zeros(T, p) : Vector{T}(beta0)
        beta_prev = copy(beta)
        ws = OuterWorkspace(T, m, p)
        iw = InnerWorkspace(T, m, p, k)
        sp = SpectralWorkspace(T, m, p, k)

        # Cache the lazy transpose once. This does not copy X.
        Xt = transpose(X)

        # Persistent Lanczos workspace: allocate only the largest Krylov
        # dimension that can actually be reached within lanczos_retries.
        r_lz = k + 1
        kd0_lz = something(spectral.krylovdim, min(p, max(3r_lz + 20, r_lz + 10)))
        maxkd_requested = something(spectral.max_krylovdim, p)
        maxkd_lz = lanczos_workspace_capacity(kd0_lz, min(maxkd_requested, p), spectral.lanczos_retries)
        lzws = LanczosRestartWorkspace(X, r_lz, maxkd_lz)
        mul!(ws.Xbeta_base, X, beta)
        fbase = logloss_xb(family, ws.Xbeta_base, y, beta; ridge=spectral.ridge)
    end

    # ------------------------------------------------------------
    # History
    # ------------------------------------------------------------
    losses = T[]
    gradnorms = T[]
    relgradnorms = T[]
    eigresiduals = T[]
    inner_iters = Int[]
    inner_stats = T[]
    outer_steps = T[]
    restart_flags = Bool[]
    correction_flags = Bool[]

    # ------------------------------------------------------------
    # Algorithm state
    # ------------------------------------------------------------
    grad_scale = one(T)
    grad_ref = T(NaN)
    t_outer = one(T)
    need_restart = true
    last_eigres = T(NaN)
    restarts = 0

    # ============================================================
    # Outer loop
    # ============================================================
    for iter in 1:outer.maxiter
        
        # --------------------------------------------------------
        # Evaluation point
        # --------------------------------------------------------
        if outer.nesterov && iter > 1
            extrapolate!(ws.beta_eval, beta, beta_prev, t_outer)
        else
            copyto!(ws.beta_eval, beta)
        end
        mul!(ws.Xbeta, X, ws.beta_eval)
        f0 = logloss_xb(family, ws.Xbeta, y, ws.beta_eval; ridge=spectral.ridge)
        
        # Monotone Nesterov restart.
        if outer.nesterov && iter > 1 && (!isfinite(f0) || f0 > fbase)
            copyto!(ws.beta_eval, beta)
            copyto!(ws.Xbeta, ws.Xbeta_base)
            f0 = fbase
            t_outer = one(T)
        end

        # --------------------------------------------------------
        # Gradient and IRLS weights
        # --------------------------------------------------------
        local gnorm, relgrad
        @timeit SMM_TIMER "gradient + weights" begin
            grad_weights_xb!(family, ws.g, ws.mu, ws.w, ws.grad_resid, X, ws.Xbeta, y, ws.beta_eval;
                            ridge=spectral.ridge, w_floor=outer.w_floor)
            gnorm = norm(ws.g)
            if iter == 1
                grad_ref = gnorm
                grad_scale = one(T) + gnorm
            end
            relgrad = gnorm / grad_scale
        end
        push!(losses, fbase)
        push!(gradnorms, gnorm)
        push!(relgradnorms, relgrad)

        # --------------------------------------------------------
        # Convergence
        # --------------------------------------------------------
        if gnorm <= outer.gtol || relgrad <= outer.relgtol
            outer.verbose && @printf("outer=%4d  solver=%s  loss=%.6e  |g|=%.3e  rel|g|=%.3e  converged=true\n",
                                iter, uppercase(String(inner.solver)), fbase, gnorm, relgrad)
            return MMResult(beta, losses, gradnorms, relgradnorms, eigresiduals, inner_iters, inner_stats,
                outer_steps, restart_flags, correction_flags, restarts, iter - 1, true)
        end

        # --------------------------------------------------------
        # Spectral state
        # --------------------------------------------------------
        periodic_restart = spectral.restart_every > 0 && iter > 1 && (iter - 1) % spectral.restart_every == 0
        restarted = need_restart || periodic_restart
        if restarted
            @timeit SMM_TIMER "spectral restart (Lanczos)" begin
                last_eigres, _ = restart_spectrum!(sp, lzws, X, Xt, ws.w, k, iter, spectral)
            end
            # The stored spectral state now corresponds to the current IRLS weights.
            copyto!(ws.w_spectral, ws.w)
            restarts += 1
            need_restart = false
        end
        eigres = last_eigres
        push!(eigresiduals, eigres)
        push!(restart_flags, restarted)

        # --------------------------------------------------------
        # Spectral majorizer
        #
        # M = gamma I + Vk(Dk - gamma I)Vk'
        # --------------------------------------------------------
        local Vk, gamma, factor_base
        @timeit SMM_TIMER "majorizer setup" begin
            Vk, sp.d_eigs, gamma = eig_factor!(sp.d_eigs, sp.V, sp.lambda, k)
            factor_base, sp.factor_coeff = inv_cache!(sp.factor_coeff, sp.d_eigs, gamma, spectral.rho)
        end

        # --------------------------------------------------------
        # Working response
        # --------------------------------------------------------
        @timeit SMM_TIMER "working response" begin
            work_y!(family, ws.z, ws.Xbeta, y, ws.mu, ws.w; w_floor=outer.w_floor)
        end

        # --------------------------------------------------------
        # Inner forcing: eta_m = min(eta_max, forcing_c * grad_ratio^forcing_alpha)
        # --------------------------------------------------------
        grad_ratio = gnorm / max(grad_ref, sqrt(eps(T)))
        eta_m = min(inner.eta_max, inner.forcing_c * grad_ratio^inner.forcing_alpha)

        # --------------------------------------------------------
        # Fixed-WLS inner solve
        # --------------------------------------------------------
        local nit, stat, failed
        @timeit SMM_TIMER "inner solve" begin
            nit, stat, failed = solve_inner!(
                iw.beta_inner, iw.Xbeta_inner, ws.beta_eval, ws.Xbeta, X, ws.z, ws.w,
                Vk, factor_base, sp.factor_coeff, iw, inner;
                ridge=spectral.ridge, forcing=eta_m
            )
        end
        push!(inner_iters, nit)
        push!(inner_stats, stat)
        @. ws.direction = iw.beta_inner - ws.beta_eval
        @. ws.Xdirection = iw.Xbeta_inner - ws.Xbeta
        dnorm = norm(ws.direction)

        # --------------------------------------------------------
        # Inner failure
        # --------------------------------------------------------
        if failed && (nit == 0 || !isfinite(dnorm))
            need_restart = true
            t_outer = one(T)
            push!(outer_steps, zero(T))
            push!(correction_flags, true)
            outer.verbose && @printf("outer=%4d  inner solver failed; restarting spectrum\n", iter)
            continue
        end

        # --------------------------------------------------------
        # Negligible step
        # --------------------------------------------------------
        negligible = dnorm <= T(1e-14) * (one(T) + norm(beta))
        if negligible
            outer.verbose && @printf("outer=%4d  terminated: negligible step\n", iter)
            return MMResult(beta, losses, gradnorms, relgradnorms, eigresiduals, inner_iters, inner_stats,
                outer_steps, restart_flags, correction_flags, restarts, iter, true)
        end

        # --------------------------------------------------------
        # Globalization
        # --------------------------------------------------------
        local alpha, fnew
        @timeit SMM_TIMER "line search / globalization" begin
            if outer.safeguard
                alpha, fnew =
                    outer_linesearch!(ws.beta_new, ws.Xbeta_new, ws.beta_eval, ws.Xbeta, ws.direction, ws.Xdirection,
                        y, family, fbase; ridge=spectral.ridge, alpha_min=inner.alpha_min, shrink=inner.shrink)
            else
                alpha = one(T)
                @. ws.beta_new = ws.beta_eval + ws.direction
                @. ws.Xbeta_new = ws.Xbeta + ws.Xdirection
                fnew = logloss_xb(family, ws.Xbeta_new, y, ws.beta_new; ridge=spectral.ridge)
            end
        end
        push!(outer_steps, alpha)

        # --------------------------------------------------------
        # Failed globalization
        # --------------------------------------------------------
        if alpha == zero(T)
            need_restart = true
            t_outer = one(T)
            push!(correction_flags, true)
            outer.verbose && @printf("outer=%4d  safeguard failed; restarting spectrum\n", iter)
            continue
        end
        stepnorm = alpha * dnorm

        # --------------------------------------------------------
        # Stalled convergence
        # --------------------------------------------------------
        if outer.accept_stalled && relgrad <= outer.stalled_relgtol && stepnorm <= outer.step_reltol * (one(T) + norm(beta))
            copyto!(beta, ws.beta_new)
            copyto!(ws.Xbeta_base, ws.Xbeta_new)
            outer.verbose && @printf("outer=%4d  terminated: stalled near tolerance\n", iter)
            return MMResult(beta, losses, gradnorms, relgradnorms, eigresiduals, inner_iters, inner_stats,
                outer_steps, restart_flags, correction_flags, restarts, iter, true)
        end

        # --------------------------------------------------------
        # New IRLS weights
        # --------------------------------------------------------
        @timeit SMM_TIMER "new weights" begin
            copyto!(ws.w_old, ws.w)
            weights_xb!(family, ws.mu_new, ws.w_new, ws.Xbeta_new, y; w_floor=outer.w_floor)
        end

        
        # --------------------------------------------------------
        # Adaptive spectral correction based on projected Hessian change
        #
        # B = V' (H_new - H_spectral) V = (XV)' Diagonal(w_new - w_spectral) (XV).
        #
        # Correct only when ||B||_F / ||Lambda||_F exceeds correction_tol.
        # --------------------------------------------------------
        local eigres_new, correction_failed, proj_change

        @timeit SMM_TIMER "projected change" begin
            proj_change = projected_hessian_change!(sp, ws.w_spectral, ws.w_new)
        end

        do_correction = !isfinite(proj_change) || proj_change > spectral.correction_tol

        if do_correction
            @timeit SMM_TIMER "spectral correction" begin
                eigres_new = correct_spectrum!(sp, X, ws.w_spectral, ws.w_new, spectral.ridge)

                correction_failed = !isfinite(eigres_new) || eigres_new > spectral.resid_tol ||
                    !finite_all(sp.Vnew) || !finite_all(sp.XVnew) || !finite_all(sp.lambda_new)

                if correction_failed
                    # Keep the previous spectral state and restart Lanczos next iteration.
                    need_restart = true
                else
                    accept_spectrum!(sp)
                    copyto!(ws.w_spectral, ws.w_new)
                    last_eigres = eigres_new
                    need_restart = false
                end
            end
        else
            # Reuse the current spectral state. It still corresponds to w_spectral.
            eigres_new = last_eigres
            correction_failed = false
        end

        push!(correction_flags, correction_failed)

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------
        outer.verbose && print_iteration(iter, inner.solver, fnew, relgrad, nit, stat, eigres_new, alpha, correction_failed)

        # --------------------------------------------------------
        # Accept iterate
        # --------------------------------------------------------
        copyto!(beta_prev, beta)
        copyto!(beta, ws.beta_new)
        copyto!(ws.Xbeta_base, ws.Xbeta_new)
        fbase = fnew

        # --------------------------------------------------------
        # Outer acceleration
        # --------------------------------------------------------
        if outer.nesterov
            if correction_failed && outer.restart_on_correction_failure
                t_outer = one(T)
            else
                t_outer = nesterov_update(t_outer)
            end
        end
    end

    return MMResult(
        beta,
        losses,
        gradnorms,
        relgradnorms,
        eigresiduals,
        inner_iters,
        inner_stats,
        outer_steps,
        restart_flags,
        correction_flags,
        restarts,
        outer.maxiter,
        false
    )
end


# ------------------------------------------------------------------
# Timing helpers
# ------------------------------------------------------------------
"""
    reset_smm_timer!()

Reset all accumulated Spectral-MM timing statistics.
Call this after compilation/warm-up and before a measured run.
"""
reset_smm_timer!() = nothing

"""
    show_smm_timer()

Print the accumulated timing breakdown.
"""
show_smm_timer() = show(SMM_TIMER)
