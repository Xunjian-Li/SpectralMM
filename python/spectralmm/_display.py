"""Shared public model and inference formatting (no numerical computation)."""
import numpy as np

def options_for(family, options):
    defaults={'negative_binomial':dict(theta=1.),'binomial':dict(trials=1.),
        'tweedie':dict(power=1.5),'expectile':dict(q=.5),
        'smooth_quantile':dict(q=.5,smoothing=.1),'pseudo_huber':dict(delta=1.),
        'student_t':dict(nu=4.,sigma=1.)}
    result=defaults.get(family,{}).copy()
    for k,v in options.items(): result['q' if k=='tau' else k]=v
    return result

def family_lines(family, options, link=None):
    pairs={'gaussian':('gaussian','identity'),'gaussian_log':('gaussian','log'),
        'bernoulli':('binomial','logit'),'probit':('binomial','probit'),
        'binomial':('binomial','logit'),'poisson':('poisson','log'),
        'gamma':('gamma','log'),'gamma_inverse':('gamma','inverse'),
        'negative_binomial':('negative_binomial','log'),'tweedie':('tweedie','log')}
    name,default=pairs.get(family,(family,'not applicable'))
    lines=[f'Family: {name}   Link: {link or default}']
    opts=options_for(family,options)
    if opts:
        def fmt(v):
            a=np.asarray(v)
            return f'{float(a.reshape(-1)[0]):g}' if a.size==1 else f'<{a.size} values>'
        lines.append('Family options: '+', '.join(f'{k}={fmt(v)}' for k,v in opts.items()))
    return lines

def model_lines(family,options,n,p,intercept,backend,solver,rank,link=None):
    solver={'spectral':'pcg','cholesky':'cho'}.get(solver,solver)
    return ['SpectralMM Regression Model',*family_lines(family,options,link),
        f'Observations: {n}   Parameters: {p}',
        f'Intercept: {"yes" if intercept else "no"}   Backend: {backend}',
        f'Solver: {solver.upper()}'+(f'   Rank: {rank}' if solver in ('pcg','mm') else '')]

def stopping_reason(info,gtol,relgtol):
    if info['gradnorm']<=gtol: return 'absolute gradient tolerance'
    if info['relgradnorm']<=relgtol: return 'relative gradient tolerance'
    return info.get('termination_reason','not recorded').replace('_',' ')

def terminal_lines(info,value,metric,reason):
    status='converged' if info.get('gradient_converged',False) else 'terminated'
    return [f'Status: {status}',f'Stopping criterion: {reason}',
        f'Outer iterations: {info["iterations"]}   Total inner iterations: {info["inner_iterations"]}',
        '',f'Final {metric}: {value:.6e}',f'Gradient norm: {info["gradnorm"]:.3e}',
        f'Relative gradient: {info["relgradnorm"]:.3e}']

def inference_lines(family,options,link,inf,names,coef,max_rows=20):
    lines=['Inference:',*family_lines(family,options,link)]
    ok=inf['status']=='ok'
    if ok:
        ref=f't (df={inf["df_resid"]:g})' if inf['statistic_type']=='t' else 'standard normal'
        cov='sandwich HC1' if inf['cov_type']=='sandwich' else 'model-based'
        lines.append(f'Covariance: {cov}   Reference distribution: {ref}')
    else: lines.extend([f'Status: {inf["status"]}',f'Reason: {inf["reason"]}'])
    lines.extend(['','Coefficients:',f'{"Term":<24} {"Estimate":>12}'+(f' {"Std. Error":>12} {inf["statistic_type"]+" statistic":>12} {"p-value":>10}' if ok else '')])
    for i in range(min(max_rows,len(coef))):
        line=f'{names[i]:<24} {coef[i]:12.6f}'
        if ok:
            pv='<0.0001' if inf['p_value'][i]<.0001 else f'{inf["p_value"][i]:.4f}'
            line+=f' {inf["std_error"][i]:12.6f} {inf["statistic"][i]:12.4f} {pv:>10}'
        lines.append(line)
    if len(coef)>max_rows: lines.append('Additional coefficients are available in params.')
    return lines
