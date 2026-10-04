# Shared English iteration-table formatting.
function print_trace_header(solver, rank)
    println()
    @printf("SpectralMM  Solver: %s   Rank: %s\n\n", uppercase(String(solver)), rank===nothing ? "-" : string(rank))
    @printf("%4s  %12s  %10s  %5s  %10s  %10s  %6s  %s\n",
            "Iter", "Loss", "RelGrad", "Inner", "InnerRes", "EigRes", "Step", "Spectrum")
end

function print_iteration(iter, loss, relgrad, inner_iter, inner_stat, eigres, step, status::Symbol)
    eigstr = isfinite(eigres) ? @sprintf("%.2e", eigres) : "-"
    statusstr =
        status === :none            ? "-" :
        status === :initial         ? "initial" :
        status === :init            ? "initial" :
        status === :restart         ? "restart" :
        status === :correct         ? "correct" :
        status === :correct_fail    ? "correct-fail" :
        status === :restart_correct ? "restart+correct" :
        status === :restart_fail    ? "restart+fail" :
                                      "reuse"

    innerstr = inner_iter < 0 ? "-" : string(inner_iter)
    statstr = isfinite(inner_stat) ? @sprintf("%.2e", inner_stat) : "-"
    stepstr = isfinite(step) ? @sprintf("%.2f", step) : "-"
    @printf("%4d  %12.4e  %10.3e  %5s  %10s  %10s  %6s  %s\n",
            iter, loss, relgrad, innerstr, statstr, eigstr, stepstr, statusstr)
end

function print_initial_iteration(loss, relgrad, eigres)
    eigstr = isfinite(eigres) ? @sprintf("%.2e", eigres) : "-"
    @printf("%4d  %12.4e  %10.3e  %5s  %10s  %10s  %6s  %s\n",
            0, loss, relgrad, "-", "-", eigstr, "-", "initial")
end

function print_failure_iteration(iter, inner_iter, status)
    @printf("Attempt %d: %s (inner iterations: %d; no accepted update)\n",
            iter, status, inner_iter)
end

function print_convergence(iters, loss, relgrad)
    println()
    @printf("Converged after %d iterations\n", iters)
    @printf("Final loss:      %.6e\n", loss)
    @printf("Relative grad.:  %.3e\n", relgrad)
end

function print_termination(reason, iters, loss, relgrad)
    println()
    @printf("Terminated after %d iterations (%s)\n", iters, reason)
    @printf("Final loss:      %.6e\n", loss)
    @printf("Relative grad.:  %.3e\n", relgrad)
end


# Optional buffered logging for the original Julia non-spectral solvers.
# Scratch arrays are separate from solver workspaces; enabling output cannot
# change the numerical trajectory. No logger is allocated when verbose=false.
mutable struct IterationLog{TX,TY,TF,T}
    X::TX
    y::TY
    family::TF
    ridge::T
    penalize_intercept::Bool
    floor::T
    solver::Symbol
    scale::T
    g::Vector{T}
    mu::Vector{T}
    w::Vector{T}
    score::Vector{T}
    rows::Vector{NamedTuple}
end
function IterationLog(X::AbstractMatrix{T},y,family,ridge,penalty,floor,solver) where {T}
    n,p=size(X)
    IterationLog(X,y,family,ridge,penalty,floor,solver,one(T),zeros(T,p),
        zeros(T,n),zeros(T,n),zeros(T,n),NamedTuple[])
end
function log_point!(log::IterationLog, iter, beta, eta; inner=-1, residual=NaN, step=NaN)
    grad_weights_xb!(log.family,log.g,log.mu,log.w,log.score,log.X,eta,log.y,beta;
        ridge=log.ridge,penalize_intercept=log.penalize_intercept,w_floor=log.floor)
    grad=norm(log.g)
    isempty(log.rows) && (log.scale=1+grad)
    loss=logloss_xb(log.family,eta,log.y,beta;ridge=log.ridge,penalize_intercept=log.penalize_intercept)
    row=(;iteration=iter,loss,gradnorm=grad,relgradnorm=grad/log.scale,inner,
        inner_residual=residual,step)
    if !isempty(log.rows) && last(log.rows).iteration==iter
        log.rows[end]=row
    else
        push!(log.rows,row)
    end
    nothing
end
function finish_log!(log::IterationLog,iter,beta,eta,gtol,relgtol,reason,converged)
    lastrow=last(log.rows)
    same=lastrow.iteration==iter
    log_point!(log,iter,beta,eta;inner=same ? lastrow.inner : 0,
        residual=same ? lastrow.inner_residual : NaN,step=same ? lastrow.step : 0.)
    print_trace_header(log.solver,nothing)
    for r in log.rows
        print_iteration(r.iteration,r.loss,r.relgradnorm,r.inner,r.inner_residual,NaN,r.step,:none)
    end
    r=last(log.rows)
    if converged && (r.gradnorm<=gtol || r.relgradnorm<=relgtol)
        print_convergence(iter,r.loss,r.relgradnorm)
    else
        print_termination(reason,iter,r.loss,r.relgradnorm)
    end
    nothing
end
