
mutable struct IRLSCholeskyResult{T}
    beta::Vector{T}
    losses::Vector{T}
    gradnorms::Vector{T}
    relgradnorms::Vector{T}
    stepsizes::Vector{T}
    iters::Int
    converged::Bool
end

function irls_cholesky(
    X::AbstractMatrix{T}, y::AbstractVector{T};
    family::IRLSFamily=BernoulliLogit(), ridge::T=T(1e-6),
    beta0::Union{Nothing,AbstractVector{T}}=nothing,
    maxiter::Int=100, gtol::T=T(1e-7), relgtol::T=T(1e-8), w_floor::T=T(1e-12),
    chol_jitter::T=T(1e-10), max_chol_tries::Int=8,
    line_search::Bool=true, alpha_min::T=T(1e-10), backtrack_factor::T=T(0.5),
    c_armijo::T=T(1e-4), penalize_intercept::Bool=true,
    outer_nesterov::Bool=false, restart_outer_on_line_search_failed::Bool=true,
    reuse_factorization::Bool=true, weight_reuse_tol::T=T(100)*eps(T),
    verbose::Bool=true
) where {T<:LinearAlgebra.BlasReal}

    m,p=size(X)
    length(y)==m || throw(DimensionMismatch("length(y) must equal size(X,1)"))
    check_y(family,y)
    beta0v=beta0===nothing ? zeros(T,p) : Vector{T}(beta0)
    length(beta0v)==p || throw(DimensionMismatch("length(beta0) must equal size(X,2)"))

    beta_base,beta_prev=copy(beta0v),copy(beta0v)
    beta_eval,beta_new,beta_trial,s=similar(beta0v),similar(beta0v),similar(beta0v),similar(beta0v)
    Xbeta_base,Xbeta_prev,Xbeta=zeros(T,m),zeros(T,m),zeros(T,m)
    Xbeta_new,Xs=zeros(T,m),zeros(T,m)
    mul!(Xbeta_base,X,beta_base); copyto!(Xbeta_prev,Xbeta_base)

    mu,w,w_factor=zeros(T,m),zeros(T,m),zeros(T,m)
    z,wz,grad_resid=zeros(T,m),zeros(T,m),zeros(T,m)
    g,b=zeros(T,p),zeros(T,p)
    A,Atry=zeros(T,p,p),zeros(T,p,p)
    sqrtWX=Matrix{T}(undef,m,p)

    losses,gradnorms=Vector{T}(undef,maxiter),Vector{T}(undef,maxiter)
    relgradnorms,stepsizes=Vector{T}(undef,maxiter),Vector{T}(undef,maxiter)
    converged,grad_scale,t_outer,base_loss=false,one(T),one(T),T(Inf)
    factor_valid,F=false,nothing

    function weights_unchanged()
        factor_valid || return false
        scale,diff=zero(T),zero(T)
        @inbounds @simd for i in 1:m
            scale=max(scale,abs(w[i])); diff=max(diff,abs(w[i]-w_factor[i]))
        end
        return diff<=weight_reuse_tol*max(one(T),scale)
    end

    function weighted_gram!()
        @inbounds for j in 1:p
            @simd for i in 1:m
                sqrtWX[i,j]=sqrt(max(w[i],zero(T)))*X[i,j]
            end
        end
        BLAS.syrk!('U','T',one(T),sqrtWX,zero(T),A)
        if ridge>zero(T)
            j0=penalize_intercept ? 1 : 2
            @inbounds for j in j0:p; A[j,j]+=ridge; end
        end
        return nothing
    end

    function factorize_gram!()
        weighted_gram!()
        for attempt in 1:max_chol_tries
            copyto!(Atry,A)
            jitter=attempt==1 ? zero(T) : chol_jitter*T(10)^(attempt-2)
            jitter>zero(T) && (@inbounds for j in 1:p; Atry[j,j]+=jitter; end)
            Ftry=cholesky!(Symmetric(Atry,:U);check=false)
            if issuccess(Ftry)
                verbose && attempt>1 && @printf("          Cholesky succeeded with jitter %.3e\n",jitter)
                return Ftry,true
            end
        end
        return nothing,false
    end

    function finish(iter,nsteps,ok)
        resize!(losses,iter); resize!(gradnorms,iter); resize!(relgradnorms,iter); resize!(stepsizes,nsteps)
        return IRLSCholeskyResult(copy(beta_base),losses,gradnorms,relgradnorms,stepsizes,nsteps,ok)
    end

    for iter in 1:maxiter
        outer_reset=false
        if outer_nesterov && iter>1
            theta=(t_outer-one(T))/(t_outer+one(T))
            @. beta_eval=beta_base+theta*(beta_base-beta_prev)
            @. Xbeta=Xbeta_base+theta*(Xbeta_base-Xbeta_prev)
        else
            copyto!(beta_eval,beta_base); copyto!(Xbeta,Xbeta_base)
        end

        f0=logloss_xb(family,Xbeta,y,beta_eval;ridge=ridge)
        if iter==1
            base_loss=f0
        elseif outer_nesterov && f0>base_loss
            copyto!(beta_eval,beta_base); copyto!(Xbeta,Xbeta_base)
            f0,t_outer,outer_reset=base_loss,one(T),true
        end

        grad_weights_xb!(family,g,mu,w,grad_resid,X,Xbeta,y,beta_eval;ridge=ridge,w_floor=w_floor)
        gnorm=norm(g)
        iter==1 && (grad_scale=one(T)+gnorm)
        relgnorm=gnorm/grad_scale
        losses[iter],gradnorms[iter],relgradnorms[iter]=f0,gnorm,relgnorm

        if gnorm<=gtol || relgnorm<=relgtol
            copyto!(beta_base,beta_eval); copyto!(Xbeta_base,Xbeta)
            converged=true
            return finish(iter,iter-1,true)
        end

        work_y!(family,z,Xbeta,y,mu,w;w_floor=w_floor)

        reuse=reuse_factorization && weights_unchanged()
        if !reuse
            Fnew,solved=factorize_gram!()
            solved || return finish(iter,iter-1,false)
            F=Fnew; copyto!(w_factor,w); factor_valid=true
        end

        @inbounds @simd for i in 1:m; wz[i]=w[i]*z[i]; end
        mul!(b,transpose(X),wz)
        if ridge>zero(T) && !penalize_intercept
            # Normal equations contain no ridge contribution in the RHS.
            nothing
        end
        copyto!(beta_new,b); ldiv!(F,beta_new)
        finite_all(beta_new) || return finish(iter,iter-1,false)

        @. s=beta_new-beta_eval
        mul!(Xs,X,s)
        gd,alpha,fnew=dot(g,s),one(T),T(Inf)

        if line_search
            while alpha>=alpha_min
                @. beta_trial=beta_eval+alpha*s
                @. Xbeta_new=Xbeta+alpha*Xs
                if !admissible_eta(family,Xbeta_new)
                    alpha*=backtrack_factor; continue
                end
                fnew=logloss_xb(family,Xbeta_new,y,beta_trial;ridge=ridge)
                if !isfinite(fnew)
                    alpha*=backtrack_factor; continue
                end
                ((gd<zero(T) && fnew<=f0+c_armijo*alpha*gd) || (gd>=zero(T) && fnew<=f0)) && break
                alpha*=backtrack_factor
            end
            if alpha<alpha_min
                if outer_nesterov && restart_outer_on_line_search_failed && iter>1
                    t_outer=one(T); continue
                end
                return finish(iter,iter-1,false)
            end
        else
            copyto!(beta_trial,beta_new); @. Xbeta_new=Xbeta+Xs
            fnew=logloss_xb(family,Xbeta_new,y,beta_trial;ridge=ridge)
        end

        stepsizes[iter]=alpha
        step_norm=abs(alpha)*norm(s)
        if verbose
            @printf("outer=%4d  loss=%.6e  |g|=%.3e  rel|g|=%.3e",iter,f0,gnorm,relgnorm)
            reuse && @printf("  chol=reused")
            outer_reset && @printf("  reset=true")
            println()
        end

        if step_norm<=T(1e-12)*(one(T)+norm(beta_eval))
            copyto!(beta_base,beta_trial); copyto!(Xbeta_base,Xbeta_new)
            return finish(iter,iter,true)
        end

        if outer_nesterov
            copyto!(beta_prev,beta_base); copyto!(Xbeta_prev,Xbeta_base)
            copyto!(beta_base,beta_trial); copyto!(Xbeta_base,Xbeta_new)
            base_loss=fnew
            t_outer=T(0.5)*(one(T)+sqrt(one(T)+T(4)*t_outer^2))
        else
            copyto!(beta_base,beta_trial); copyto!(Xbeta_base,Xbeta_new); base_loss=fnew
        end
    end
    return finish(maxiter,maxiter,converged)
end
