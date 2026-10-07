# Public presentation metadata; the numerical family structs retain their ABI.
function _display_options(name,options)
    defaults=Dict(:negative_binomial=>(theta=1.,),:binomial=>(trials=1.,),
        :tweedie=>(power=1.5,),:expectile=>(q=.5,),:smooth_quantile=>(q=.5,smoothing=.1),
        :pseudo_huber=>(delta=1.,),:student_t=>(nu=4.,sigma=1.))
    d=Dict{Symbol,Any}(pairs(get(defaults,Symbol(name),NamedTuple())))
    for (k,v) in pairs(options); d[k===:tau ? :q : k]=v; end
    return (; (k=>d[k] for k in sort!(collect(keys(d));by=string))...)
end
function _print_family(io,name,options;link=nothing)
    aliases=Dict(:bernoulli=>(:binomial,:logit),:probit=>(:binomial,:probit),
        :gaussian=>(:gaussian,:identity),:gaussian_log=>(:gaussian,:log),
        :gamma_inverse=>(:gamma,:inverse),:gamma=>(:gamma,:log),:poisson=>(:poisson,:log),
        :negative_binomial=>(:negative_binomial,:log),:binomial=>(:binomial,:logit),:tweedie=>(:tweedie,:log))
    family,default=get(aliases,Symbol(name),(name,"not applicable"))
    println(io,"Family: ",family,"   Link: ",something(link,default))
    if !isempty(options)
        format(v)=v isa Real ? @sprintf("%g",v) : length(v)==1 ? @sprintf("%g",only(v)) : "<$(length(v)) values>"
        println(io,"Family options: ",join(("$k=$(format(v))" for (k,v) in pairs(options)),", "))
    end
end
function _print_model_header(io,name,options,n,p,intercept,backend;link=nothing)
    println(io,"SpectralMM Regression Model")
    _print_family(io,name,options;link)
    println(io,"Observations: ",n,"   Parameters: ",p)
    println(io,"Intercept: ",intercept ? "yes" : "no","   Backend: ",backend===:julia ? "Julia" : "C++")
end

# Shared English iteration-table formatting. Internal diagnostic fields are retained.
function print_trace_header(solver, rank; metric="Objective")
    println("Solver: ",uppercase(String(solver)),rank===nothing ? "" : "   Rank: $rank"); println()
    @printf("%4s  %12s  %10s  %10s  %5s  %8s  %s\n","Iter",metric,"GradNorm","RelGrad","Inner","Stepsize","Spectrum")
end
function print_iteration(iter, loss, relgrad, inner_iter, inner_stat, eigres, step, status::Symbol;
                         gradnorm=NaN,value=loss)
    states=Dict(:none=>"-",:initial=>"initial",:init=>"initial",:restart=>"restart",
        :correct=>"correct",:correct_fail=>"correct-fail",:restart_correct=>"restart+correct",:restart_fail=>"restart+fail")
    valstr=isnan(value) ? "-" : @sprintf("%.4e",value)
    stepstr=isfinite(step) ? @sprintf("%.2f",step) : "-"
    @printf("%4d  %12s  %10.3e  %10.3e  %5s  %8s  %s\n",iter,valstr,gradnorm,relgrad,
        inner_iter<0 ? "-" : string(inner_iter),stepstr,get(states,status,"reuse"))
end
function print_initial_iteration(loss,relgrad,eigres;gradnorm=NaN,value=loss)
    print_iteration(0,loss,relgrad,-1,NaN,eigres,NaN,:initial;gradnorm,value)
end
function print_failure_iteration(iter,inner_iter,status)
    @printf("Attempt %d: %s (inner iterations: %d; no accepted update)\n",iter,status,inner_iter)
end
function print_final_metric(value,relgrad,metric,gradnorm)
    @printf("Final %s: %.6e\nGradient norm:  %.3e\nRelative gradient: %.3e\n",metric,value,gradnorm,relgrad)
    println(metric=="LogLik" ? "LogLik includes distribution constants and excludes ridge; gradients refer to the optimization objective." : "Objective is the summed model loss plus ridge penalty.")
end
function print_convergence(iters,loss,relgrad;metric="Objective",gradnorm=NaN,total_inner=nothing,criterion="gradient tolerance")
    println("\nStatus: converged\nStopping criterion: ",criterion)
    println("Outer iterations: ",iters,"   Total inner iterations: ",something(total_inner,"not recorded")); println()
    print_final_metric(loss,relgrad,metric,gradnorm)
end
function print_termination(reason,iters,loss,relgrad;metric="Objective",gradnorm=NaN,total_inner=nothing)
    println("\nStatus: terminated\nStopping criterion: ",reason)
    println("Outer iterations: ",iters,"   Total inner iterations: ",something(total_inner,"not recorded")); println()
    print_final_metric(loss,relgrad,metric,gradnorm)
end

# Only called by pure Julia solvers. Late-bound statistical helpers avoid changing
# the optimizer or allocating full covariance matrices for display.
function iteration_metric(f,y,p,dispersion)
    name=nameof(typeof(f))
    name in (:GaussianIdentity,:GaussianLog,:BernoulliLogit,:BernoulliProbit,:BinomialLogit,:PoissonLog,:NegativeBinomialLog,:GammaLog,:GammaInverse,:StudentT) || return "Objective"
    name in (:BernoulliLogit,:BernoulliProbit,:BinomialLogit,:PoissonLog,:NegativeBinomialLog) && !all(isinteger,y) && return "Objective"
    if name===:BinomialLogit
        n=f.n isa Real ? (f.n,) : f.n
        all(isinteger,n) || return "Objective"
    end
    name in (:GammaLog,:GammaInverse) && dispersion===nothing && length(y)<=p && return "Objective"
    return "LogLik"
end
function iteration_value(f,y,eta,p,dispersion,objective)
    iteration_metric(f,y,p,dispersion)=="Objective" && return objective
    names=Dict(:GaussianIdentity=>:gaussian,:GaussianLog=>:gaussian_log,
        :BernoulliLogit=>:bernoulli,:BernoulliProbit=>:probit,:BinomialLogit=>:binomial,
        :PoissonLog=>:poisson,:NegativeBinomialLog=>:negative_binomial,
        :GammaLog=>:gamma,:GammaInverse=>:gamma_inverse,:StudentT=>:student_t)
    name=names[nameof(typeof(f))]
    options=name===:binomial ? (trials=f.n,) : name===:negative_binomial ? (theta=f.theta,) :
        name===:student_t ? (nu=f.nu,sigma=f.sigma) : NamedTuple()
    mu=_response_prediction(f,eta)
    _,ll=_fit_statistics(name,y,mu,eta,p,options,dispersion)
    return ll===nothing ? NaN : ll
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
    dispersion::Union{Nothing,Float64}
    rows::Vector{NamedTuple}
end
function IterationLog(X::AbstractMatrix{T},y,family,ridge,penalty,floor,solver;dispersion=nothing) where {T}
    n,p=size(X)
    IterationLog(X,y,family,ridge,penalty,floor,solver,one(T),zeros(T,p),
        zeros(T,n),zeros(T,n),zeros(T,n),dispersion,NamedTuple[])
end
function log_point!(log::IterationLog, iter, beta, eta; inner=-1, residual=NaN, step=NaN)
    grad_weights_xb!(log.family,log.g,log.mu,log.w,log.score,log.X,eta,log.y,beta;
        ridge=log.ridge,penalize_intercept=log.penalize_intercept,w_floor=log.floor)
    grad=norm(log.g)
    isempty(log.rows) && (log.scale=1+grad)
    loss=logloss_xb(log.family,eta,log.y,beta;ridge=log.ridge,penalize_intercept=log.penalize_intercept)
    value=iteration_value(log.family,log.y,eta,length(beta),log.dispersion,loss)
    row=(;iteration=iter,loss,value,gradnorm=grad,relgradnorm=grad/log.scale,inner,
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
    metric=iteration_metric(log.family,log.y,length(beta),log.dispersion)
    print_trace_header(log.solver,nothing;metric)
    for r in log.rows
        print_iteration(r.iteration,r.loss,r.relgradnorm,r.inner,r.inner_residual,NaN,r.step,:none;gradnorm=r.gradnorm,value=r.value)
    end
    r=last(log.rows)
    if converged && (r.gradnorm<=gtol || r.relgradnorm<=relgtol)
        print_convergence(iter,r.value,r.relgradnorm;metric,gradnorm=r.gradnorm,total_inner=sum(max(row.inner,0) for row in log.rows),criterion=r.gradnorm<=gtol ? "absolute gradient tolerance" : "relative gradient tolerance")
    else
        print_termination(reason,iter,r.value,r.relgradnorm;metric,gradnorm=r.gradnorm,total_inner=sum(max(row.inner,0) for row in log.rows))
    end
    nothing
end
