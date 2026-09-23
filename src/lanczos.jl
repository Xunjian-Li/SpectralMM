# Hessian products for H = X'WX + ridge I.
# Optimized restart path: all large Lanczos/Ritz/residual arrays are persistent.
mutable struct LanczosRestartWorkspace{T}
    # H*v workspace
    Xv::Vector{T}
    WXv::Vector{T}

    # Lanczos recurrence (allocated to maximum Krylov dimension)
    Q::Matrix{T}
    alpha::Vector{T}
    beta::Vector{T}
    q::Vector{T}
    qprev::Vector{T}
    z::Vector{T}

    # Ritz extraction
    Z::Matrix{T}
    V::Matrix{T}
    lambda::Vector{T}
    XV::Matrix{T}

    # Best result across retries
    best_V::Matrix{T}
    best_XV::Matrix{T}
    best_lambda::Vector{T}

    # Residual check
    HV::Matrix{T}
    WXV::Matrix{T}
end

function LanczosRestartWorkspace(
    X::AbstractMatrix{T},
    r::Int,
    max_krylovdim::Int = size(X, 2)
) where {T <: Real}
    m, p = size(X)
    maxkd = min(max_krylovdim, p)
    @assert 1 <= r <= maxkd
    return LanczosRestartWorkspace{T}(
        zeros(T, m),                    # Xv
        zeros(T, m),                    # WXv
        zeros(T, p, maxkd),             # Q
        zeros(T, maxkd),                # alpha
        zeros(T, max(maxkd - 1, 1)),    # beta
        zeros(T, p),                    # q
        zeros(T, p),                    # qprev
        zeros(T, p),                    # z
        zeros(T, maxkd, r),             # Z
        zeros(T, p, r),                 # V
        zeros(T, r),                    # lambda
        zeros(T, m, r),                 # XV
        zeros(T, p, r),                 # best_V
        zeros(T, m, r),                 # best_XV
        zeros(T, r),                    # best_lambda
        zeros(T, p, r),                 # HV
        zeros(T, m, r),                 # WXV
    )
end

@inline function hess_mv!(
    Hv::AbstractVector{T},
    ws::LanczosRestartWorkspace{T},
    X::AbstractMatrix{T},
    Xt,
    w::AbstractVector{T},
    v::AbstractVector{T};
    ridge::T = zero(T)
) where {T <: Real}
    mul!(ws.Xv, X, v)
    @inbounds @simd for i in eachindex(ws.Xv)
        ws.WXv[i] = w[i] * ws.Xv[i]
    end
    mul!(Hv, Xt, ws.WXv)
    if ridge > zero(T)
        @inbounds @simd for i in eachindex(Hv)
            Hv[i] += ridge * v[i]
        end
    end
    return Hv
end

# Compatibility allocating interface. The optimized restart path does not use it.
function hess_mv(
    X::AbstractMatrix{T},
    w::AbstractVector{T},
    v::AbstractVector{T};
    ridge::T = zero(T)
) where {T <: Real}
    Xv = X * v
    WXv = similar(Xv)
    @inbounds @simd for i in eachindex(Xv)
        WXv[i] = w[i] * Xv[i]
    end
    Hv = transpose(X) * WXv
    if ridge > zero(T)
        @inbounds @simd for i in eachindex(Hv)
            Hv[i] += ridge * v[i]
        end
    end
    return Hv
end

function weight_rows!(
    Y::AbstractMatrix{T},
    w::AbstractVector{T},
    Xmat::AbstractMatrix{T}
) where {T <: Real}
    @assert size(Y) == size(Xmat)
    @assert length(w) == size(Xmat, 1)
    @inbounds for j in axes(Xmat, 2)
        @simd for i in axes(Xmat, 1)
            Y[i, j] = w[i] * Xmat[i, j]
        end
    end
    return Y
end

function hess_xv!(
    HV::AbstractMatrix{T},
    WXV::AbstractMatrix{T},
    X::AbstractMatrix{T},
    Xt,
    w::AbstractVector{T},
    V::AbstractMatrix{T},
    XV::AbstractMatrix{T};
    ridge::T = zero(T)
) where {T <: Real}
    weight_rows!(WXV, w, XV)
    mul!(HV, Xt, WXV)
    if ridge > zero(T)
        @inbounds for j in axes(HV, 2)
            @simd for i in axes(HV, 1)
                HV[i, j] += ridge * V[i, j]
            end
        end
    end
    return HV
end

# Compatibility method.
function hess_xv!(
    HV::AbstractMatrix{T}, WXV::AbstractMatrix{T}, X::AbstractMatrix{T},
    w::AbstractVector{T}, V::AbstractMatrix{T}, XV::AbstractMatrix{T};
    ridge::T = zero(T)
) where {T <: Real}
    return hess_xv!(HV, WXV, X, transpose(X), w, V, XV; ridge=ridge)
end

function eig_resid_hv!(
    HV::AbstractMatrix{T},
    V::AbstractMatrix{T},
    lambda::AbstractVector{T}
) where {T <: Real}
    hv_norm = norm(HV)
    @inbounds for j in axes(HV, 2)
        lj = lambda[j]
        @simd for i in axes(HV, 1)
            HV[i, j] -= lj * V[i, j]
        end
    end
    return norm(HV) / max(hv_norm, eps(T))
end

function eig_resid_xv!(
    HV::AbstractMatrix{T},
    WXV::AbstractMatrix{T},
    X::AbstractMatrix{T},
    Xt,
    w::AbstractVector{T},
    V::AbstractMatrix{T},
    XV::AbstractMatrix{T},
    lambda::AbstractVector{T};
    ridge::T = zero(T)
) where {T <: Real}
    hess_xv!(HV, WXV, X, Xt, w, V, XV; ridge=ridge)
    return eig_resid_hv!(HV, V, lambda)
end

# Allocation-light Lanczos + Ritz extraction.
# Large Q, q, qprev, z, Z, V and lambda arrays are reused from ws.
function top_eigs!(
    ws::LanczosRestartWorkspace{T},
    X::AbstractMatrix{T},
    Xt,
    w::AbstractVector{T},
    r::Int;
    ridge::T = zero(T),
    krylovdim::Int = min(size(X, 2), max(3r + 20, r + 10)),
    tol::T = T(1e-10),
    seed::Int = 1,
    reorthogonalize::Bool = true
) where {T <: Real}
    p = size(X, 2)
    @assert 1 <= r <= p
    @assert r <= krylovdim <= min(p, size(ws.Q, 2))

    Q = ws.Q
    alpha = ws.alpha
    beta = ws.beta
    q = ws.q
    qprev = ws.qprev
    z = ws.z

    rng = MersenneTwister(seed)
    randn!(rng, q)
    nq = norm(q)
    invnq = inv(nq)
    @inbounds @simd for i in eachindex(q)
        q[i] *= invnq
        qprev[i] = zero(T)
    end

    actual_dim = krylovdim

    @timeit SMM_TIMER "restart/Lanczos iterations" begin
        for j in 1:krylovdim
            copyto!(@view(Q[:, j]), q)

            @timeit SMM_TIMER "restart/matvec" begin
                hess_mv!(z, ws, X, Xt, w, q; ridge=ridge)
            end

            if j > 1
                bj = beta[j - 1]
                @inbounds @simd for i in eachindex(z)
                    z[i] -= bj * qprev[i]
                end
            end

            aj = dot(q, z)
            alpha[j] = aj
            @inbounds @simd for i in eachindex(z)
                z[i] -= aj * q[i]
            end

            if reorthogonalize
                @timeit SMM_TIMER "restart/reorthogonalization" begin
                    for s in 1:j
                        qs = @view Q[:, s]
                        c = dot(qs, z)
                        @inbounds @simd for i in eachindex(z)
                            z[i] -= c * qs[i]
                        end
                    end
                end
            end

            if j < krylovdim
                bj = norm(z)
                beta[j] = bj
                if bj <= tol
                    actual_dim = j
                    break
                end
                copyto!(qprev, q)
                invbj = inv(bj)
                @inbounds @simd for i in eachindex(q)
                    q[i] = z[i] * invbj
                end
            end
        end
    end

    # This small eigensolve is intentionally left to LAPACK/LinearAlgebra.
    # Views avoid copies of alpha and beta; eigen itself still allocates O(kd^2).
    eigT = @timeit SMM_TIMER "restart/tridiagonal eigen" begin
        av = @view alpha[1:actual_dim]
        bv = @view beta[1:max(actual_dim - 1, 0)]
        eigen(SymTridiagonal(av, bv))
    end

    # eigen(SymTridiagonal) returns ascending eigenvalues. Copy the largest r
    # in reverse order into persistent lambda/Z, avoiding sortperm and slicing.
    @timeit SMM_TIMER "restart/Ritz select" begin
        @inbounds for j in 1:r
            src = actual_dim - j + 1
            ws.lambda[j] = T(eigT.values[src])
            @simd for i in 1:actual_dim
                ws.Z[i, j] = T(eigT.vectors[i, src])
            end
        end
    end

    @timeit SMM_TIMER "restart/Ritz vectors" begin
        Qv = @view Q[:, 1:actual_dim]
        Zv = @view ws.Z[1:actual_dim, 1:r]
        mul!(ws.V, Qv, Zv)
    end

    @timeit SMM_TIMER "restart/Ritz normalize" begin
        @inbounds for j in 1:r
            vj = @view ws.V[:, j]
            nv = norm(vj)
            invnv = inv(nv)
            @simd for i in eachindex(vj)
                vj[i] *= invnv
            end
        end
    end

    return ws.V, ws.lambda, actual_dim
end

# Fully optimized checked restart. ws MUST be created once outside the outer loop.
function checked_lz!(
    ws::LanczosRestartWorkspace{T},
    X::AbstractMatrix{T},
    Xt,
    w::AbstractVector{T},
    r::Int;
    ridge::T = zero(T),
    krylovdim::Int = min(size(X, 2), max(3r + 20, r + 10)),
    max_krylovdim::Int = size(X, 2),
    tol::T = T(1e-8),
    resid_tol::T = T(5e-2),
    seed::Int = 1,
    max_retries::Int = 3,
    verbose::Bool = false
) where {T <: Real}
    p = size(X, 2)
    kd = min(krylovdim, p, size(ws.Q, 2))
    max_kd = min(max_krylovdim, p, size(ws.Q, 2))
    best_res = T(Inf)
    best_kd = kd

    for attempt in 1:max_retries
        V, lambda, _ = @timeit SMM_TIMER "restart/lz eig total" begin
            top_eigs!(ws, X, Xt, w, r;
                ridge=ridge,
                krylovdim=kd,
                tol=tol,
                seed=seed + 7919 * attempt,
                reorthogonalize=true)
        end

        @timeit SMM_TIMER "restart/XV" begin
            mul!(ws.XV, X, V)
        end

        res = @timeit SMM_TIMER "restart/eigen residual" begin
            eig_resid_xv!(ws.HV, ws.WXV, X, Xt, w, V, ws.XV, lambda; ridge=ridge)
        end

        if res < best_res
            @timeit SMM_TIMER "restart/best copy" begin
                best_res = res
                copyto!(ws.best_V, V)
                copyto!(ws.best_XV, ws.XV)
                copyto!(ws.best_lambda, lambda)
                best_kd = kd
            end
        end

        if verbose
            @printf("          Lanczos attempt=%d  kd=%d  eigres=%.3e\n", attempt, kd, res)
        end

        if res <= resid_tol
            # Current V/XV/lambda live in persistent workspace.
            return V, ws.XV, lambda, res, kd, true
        end

        if kd >= max_kd
            break
        end
        kd = min(max_kd, max(kd + 20, Int(ceil(1.5 * kd))))
    end

    return ws.best_V, ws.best_XV, ws.best_lambda, best_res, best_kd, false
end

# Compatibility wrapper. Correct but allocates a workspace on every call.
# Performance-critical code should call checked_lz!(ws, X, Xt, ...).
function checked_lz(
    X::AbstractMatrix{T},
    w::AbstractVector{T},
    r::Int;
    Xt = transpose(X),
    ridge::T = zero(T),
    krylovdim::Int = min(size(X, 2), max(3r + 20, r + 10)),
    max_krylovdim::Int = size(X, 2),
    tol::T = T(1e-8),
    resid_tol::T = T(5e-2),
    seed::Int = 1,
    max_retries::Int = 3,
    verbose::Bool = false
) where {T <: Real}
    ws = LanczosRestartWorkspace(X, r, max_krylovdim)
    return checked_lz!(ws, X, Xt, w, r;
        ridge=ridge,
        krylovdim=krylovdim,
        max_krylovdim=max_krylovdim,
        tol=tol,
        resid_tol=resid_tol,
        seed=seed,
        max_retries=max_retries,
        verbose=verbose)
end

# Low-rank + γI majorizer: keep the top-k directions and use λ[k+1] as the tail bound.
function eig_factor!(
    d::AbstractVector{T},
    Vr::AbstractMatrix{T},
    λr::AbstractVector{T},
    k::Int
) where {T <: Real}

    @assert length(λr) >= k + 1
    @assert length(d) == k

    Vk = @view Vr[:, 1:k]

    @inbounds for i in 1:k
        d[i] = max(λr[i], zero(T))
    end

    γ = max(λr[k + 1], zero(T))

    return Vk, d, γ
end

function inv_cache!(
    coeff::AbstractVector{T},
    d::AbstractVector{T},
    γ::T,
    ρ::T
) where {T <: Real}

    base = inv(γ + ρ)

    @inbounds for i in eachindex(d)
        coeff[i] = inv(d[i] + ρ) - base
    end

    return base, coeff
end

function inv_apply!(
    out::AbstractVector{T},
    tmp::AbstractVector{T},
    g::AbstractVector{T},
    V::AbstractMatrix{T},
    base::T,
    coeff::AbstractVector{T}
) where {T <: Real}

    mul!(tmp, transpose(V), g)

    @inbounds @simd for i in eachindex(tmp)
        tmp[i] *= coeff[i]
    end

    out .= base .* g
    mul!(out, V, tmp, one(T), one(T))

    return out
end

# Skinny block Hessian action used by spectral correction.
# For very small r, repeated GEMV is typically faster than a skinny GEMM.
function hess_xv_skinny!(
    HV::AbstractMatrix{T}, WXV::AbstractMatrix{T},
    X::AbstractMatrix{T}, w::AbstractVector{T},
    V::AbstractMatrix{T}, XV::AbstractMatrix{T};
    ridge::T=zero(T)
) where {T <: Real}
    weight_rows!(WXV, w, XV)
    @inbounds for j in axes(HV, 2)
        mul!(@view(HV[:,j]), transpose(X), @view(WXV[:,j]))
    end
    if ridge != zero(T)
        @inbounds for j in axes(HV, 2), i in axes(HV, 1)
            HV[i,j] += ridge * V[i,j]
        end
    end
    return HV
end

# First-order eigenspace perturbation. Cached XV avoids recomputing X * V.
# Since V_old'V_old = I and V_raw = V_old*A, orthogonalization can be
# performed entirely on the small r×r matrix A: if A = Q_A R_A, then
# V_old*A = (V_old*Q_A)R_A and X(V_old*Q_A) = XV_old*Q_A.
function perturb!(
    V_corr::AbstractMatrix{T}, XV_corr::AbstractMatrix{T}, λ_corr::AbstractVector{T},
    Δw::AbstractVector{T}, ΔWXV_old::AbstractMatrix{T}, Bmat::AbstractMatrix{T},
    E::AbstractMatrix{T}, A::AbstractMatrix{T}, V_raw::AbstractMatrix{T},
    XV_raw::AbstractMatrix{T}, HV_workspace::AbstractMatrix{T},
    WXV_workspace::AbstractMatrix{T}, Ssmall_mat::AbstractMatrix{T},
    X::AbstractMatrix{T}, w_old::AbstractVector{T}, w_new::AbstractVector{T},
    V_old::AbstractMatrix{T}, XV_old::AbstractMatrix{T}, λ_old::AbstractVector{T};
    ridge::T=zero(T), gap_tol::T=T(1e-8)
) where {T <: Real}
    r = size(V_old, 2)

    @inbounds @simd for i in eachindex(Δw)
        Δw[i] = w_new[i] - w_old[i]
    end
    (!finite_all(Δw) || !finite_all(V_old) || !finite_all(XV_old) || !finite_all(λ_old)) && return T(Inf)

    # B = V' ΔH V = (XV)' diag(Δw) XV.
    weight_rows!(ΔWXV_old, Δw, XV_old)
    mul!(Bmat, transpose(XV_old), ΔWXV_old)
    !finite_all(Bmat) && return T(Inf)

    # First-order retained-subspace rotation A = I + E.
    fill!(E, zero(T))
    @inbounds for i in 1:r, j in 1:r
        if i != j
            gap = λ_old[i] - λ_old[j]
            scale = max(one(T), abs(λ_old[i]), abs(λ_old[j]))
            abs(gap) > gap_tol * scale && (E[j,i] = Bmat[j,i] / gap)
        end
    end
    copyto!(A, E)
    @inbounds for i in 1:r
        A[i,i] += one(T)
    end

    # Small-r QR only. qr!(A) overwrites A; materialize Q_A into E.
    F = qr!(A)
    fill!(E, zero(T))
    @inbounds for i in 1:r
        E[i,i] = one(T)
    end
    lmul!(F.Q, E)

    # V_tilde = V_old*Q_A and XV_tilde = XV_old*Q_A.
    mul!(V_corr, V_old, E)
    mul!(XV_corr, XV_old, E)
    (!finite_all(V_corr) || !finite_all(XV_corr)) && return T(Inf)

    # Rayleigh-Ritz refinement. For small r use repeated GEMV rather than
    # a p×m by m×r skinny GEMM.
    hess_xv_skinny!(HV_workspace, WXV_workspace, X, w_new,
                    V_corr, XV_corr; ridge=ridge)
    mul!(Ssmall_mat, transpose(V_corr), HV_workspace)
    !finite_all(Ssmall_mat) && return T(Inf)

    @inbounds for j in 1:r, i in 1:j-1
        z = T(0.5) * (Ssmall_mat[i,j] + Ssmall_mat[j,i])
        Ssmall_mat[i,j] = z
        Ssmall_mat[j,i] = z
    end
    es = eigen!(Symmetric(Ssmall_mat, :U))

    # eigen! returns ascending eigenvalues. Reuse Bmat for descending eigenvectors.
    @inbounds for j in 1:r
        jj = r-j+1
        λ_corr[j] = es.values[jj]
        for i in 1:r
            Bmat[i,j] = es.vectors[i,jj]
        end
    end

    # Rotate corrected basis, cached XV, and H*V_tilde.
    mul!(V_raw, V_corr, Bmat)
    mul!(XV_raw, XV_corr, Bmat)
    copyto!(V_corr, V_raw)
    copyto!(XV_corr, XV_raw)
    mul!(V_raw, HV_workspace, Bmat)

    return eig_resid_hv!(V_raw, V_corr, λ_corr)
end

# Result container.

