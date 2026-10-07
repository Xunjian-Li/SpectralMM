"""Statistical interfaces layered over the unchanged numerical binding."""
from dataclasses import dataclass
from types import MappingProxyType
import inspect
import numpy as np
from scipy.special import xlogy, gammaln, log_ndtr
from scipy.stats import norm, t
from . import _api
from .families import _normalize_family

_GLM = {('gaussian','identity'):'gaussian', ('gaussian','log'):'gaussian_log',
        ('binomial','logit'):'bernoulli', ('binomial','probit'):'probit',
        ('poisson','log'):'poisson', ('gamma','log'):'gamma',
        ('gamma','inverse'):'gamma_inverse', ('negative_binomial','log'):'negative_binomial'}
_DEFAULT = dict(gaussian='identity', binomial='logit', poisson='log', gamma='log', negative_binomial='log')
_LEGACY = {v:k for k,v in _GLM.items()}
_LEGACY['binomial'] = ('binomial','logit')
_NUMERIC = {'maxiter','inner_maxiter','krylovdim','ridge','floor','gtol','relgtol','eta_max',
            'correction_tol','resid_tol','nesterov','preconditioner','krylov_rtol','krylov_atol',
            'accept_negligible','accept_stalled','negligible_step_tol','step_reltol','stalled_relgtol'}

@dataclass(frozen=True, init=False)
class Control:
    """Advanced numerical overrides; omitted values retain the solver defaults."""
    options: object
    def __init__(self, **options):
        unknown = set(options)-_NUMERIC
        if unknown: raise TypeError(f'unknown control options: {sorted(unknown)}')
        object.__setattr__(self, 'options', MappingProxyType(dict(options)))


def _options(control, kwargs):
    if control is not None and not isinstance(control, Control): raise TypeError('control must be Control')
    opts = dict(control.options) if control is not None else {}
    duplicates = set(opts)&set(kwargs)
    if duplicates: raise TypeError(f'options supplied both directly and in control: {sorted(duplicates)}')
    opts.update(kwargs)
    return opts


def _statistics(family, y, mu, eta, p, options, dispersion):
    """Unscaled deviance and normalized density; no covariance allocation."""
    if family in ('gaussian','gaussian_log'):
        dev = np.sum((y-mu)**2)
        scale = dispersion if dispersion is not None else dev/len(y)
        ll = np.inf if scale == 0 else -.5*(len(y)*np.log(2*np.pi*scale)+dev/scale)
    elif family in ('bernoulli','probit','binomial'):
        trials = np.asarray(options.get('trials',1.))
        lp = log_ndtr(eta) if family=='probit' else -np.logaddexp(0,-eta)
        lq = log_ndtr(-eta) if family=='probit' else -np.logaddexp(0,eta)
        kernel = np.sum(y*lp+(trials-y)*lq)
        sat = np.sum(xlogy(y,y/trials)+xlogy(trials-y,(trials-y)/trials))
        dev = 2*(sat-kernel)
        ll = kernel+np.sum(gammaln(trials+1)-gammaln(y+1)-gammaln(trials-y+1))
    elif family=='poisson':
        dev = 2*np.sum(xlogy(y,y/mu)-y+mu)
        ll = np.sum(y*eta-mu-gammaln(y+1))
    elif family in ('gamma','gamma_inverse'):
        dev = 2*np.sum((y-mu)/mu-np.log(y/mu))
        scale = dispersion if dispersion is not None else (np.sum(((y-mu)/mu)**2)/(len(y)-p) if len(y)>p else np.nan)
        if not np.isfinite(scale) or scale<=0: ll=np.nan
        else:
            shape=1/scale
            ll=np.sum((shape-1)*np.log(y)-y/(mu*scale)-gammaln(shape)-shape*np.log(mu*scale))
    elif family=='negative_binomial':
        theta=options.get('theta',1.)
        dev=2*np.sum(xlogy(y,y/mu)-(y+theta)*np.log((y+theta)/(mu+theta)))
        ll=np.sum(gammaln(y+theta)-gammaln(theta)-gammaln(y+1)+theta*np.log(theta/(theta+mu))+xlogy(y,mu/(theta+mu)))
    elif family=="student_t":
        nu=options.get("nu",4.); sigma=options.get("sigma",1.)
        return None, float(np.sum(t.logpdf((y-eta)/sigma,df=nu)-np.log(sigma)))
    else: return None, None
    if family in ('bernoulli','probit','binomial','poisson','negative_binomial') and not np.all(np.asarray(y)==np.floor(y)): ll=np.nan
    if family=='binomial' and not np.all(np.asarray(options.get('trials',1.))==np.floor(options.get('trials',1.))): ll=np.nan
    return float(dev), float(ll)


class Results:
    """GLM-style statistical result, with legacy numerical access delegated."""
    def __init__(self, raw, X, y, names, family, link, formula, design_info, intercept, settings, dispersion):
        self._raw=raw
        self.family=family; self.link=link
        self.formula=formula; self.design_info=design_info
        self.param_names=tuple(names); self.nobs=len(y); self.nparams=len(raw.coef)
        self.fit_intercept=intercept
        self.linear_predictor=np.asarray(X @ raw.coef_).reshape(-1)+raw.intercept_
        self.fittedvalues=raw.predict(X)
        self.resid_response=np.asarray(y)-self.fittedvalues
        self._deviance,self._llf=_statistics(raw.family,np.asarray(y),self.fittedvalues,self.linear_predictor,
                                           self.nparams,raw.family_options,dispersion)
        self.objective=raw.info['loss']
        self.diagnostics=dict(raw.info, solver=settings.get('solver','pcg'),
                              rank=settings['rank'], outer_iterations=raw.info['iterations'],
                              gradient_norm=raw.info['gradnorm'], objective=self.objective,
                              backend='cpp', control=dict(settings))
        self.diagnostics['krylov_dimension']=settings.get('krylovdim') or min(self.nparams,max(3*(settings['rank']+1)+20,settings['rank']+11))
        self.diagnostics['preconditioner']=settings.get('preconditioner','jacobi')
    def __getattr__(self, name): return getattr(self._raw,name)
    @property
    def params(self): return self._raw.coef
    @property
    def deviance(self):
        if self._deviance is None: raise NotImplementedError('deviance is not defined for this residual loss')
        return self._deviance
    @property
    def llf(self):
        if self._llf is None: raise NotImplementedError('a normalized likelihood is not implemented for this loss')
        return self._llf
    def conf_int(self, alpha=.05):
        if not 0<alpha<1: raise ValueError('alpha must lie in (0,1)')
        inf=self._raw._inference_result()
        q=t.ppf(1-alpha/2, inf['df_resid']) if inf['statistic_type']=='t' else norm.ppf(1-alpha/2)
        return np.column_stack((self.params-q*self.bse,self.params+q*self.bse))
    def predict(self, X=None, *, which='mean', transform=True, trials=None, offset=None):
        """Predict on training data, a new feature matrix, or a formula-compatible table.

        For non-GLM losses, ``which='mean'`` returns the fitted location: an
        expectile, smoothed quantile, or robust location, depending on the loss.
        It equals ``which='linear'`` for these identity-link models and is not
        a prediction interval. Formula transformations and the fitted intercept
        are applied automatically.
        """
        if offset is not None: raise NotImplementedError('offset is not implemented by the numerical core')
        if which not in ('mean','linear'): raise ValueError("which must be 'mean' or 'linear'")
        if X is None: return (self.fittedvalues if which=='mean' else self.linear_predictor).copy()
        if self.design_info is not None and transform:
            from patsy import build_design_matrices
            X=np.asarray(build_design_matrices([self.design_info],X,NA_action='raise')[0])
            if self.fit_intercept: X=X[:,1:]
        elif not hasattr(X,'shape'): X=np.asarray(X)
        if len(X.shape)!=2 or X.shape[1]!=len(self._raw.coef_): raise ValueError('prediction feature count differs from training')
        if which=='linear': return np.asarray(X @ self._raw.coef_).reshape(-1)+self._raw.intercept_
        return self._raw.predict(X,trials=trials)
    def _inference_lines(self,max_rows=20):
        from ._display import inference_lines
        return inference_lines(self._raw.family,self._raw.family_options,self.link,self.inference,self.param_names,self.params,max_rows)
    def summary(self, max_rows=20):
        if not isinstance(max_rows,int) or isinstance(max_rows,bool) or max_rows<1: raise ValueError('max_rows must be a positive integer')
        from ._display import model_lines,terminal_lines,stopping_reason
        d=self.diagnostics
        metric='LogLik' if self._llf is not None and not np.isnan(self._llf) else 'Objective'
        lines=model_lines(self._raw.family,self._raw.family_options,self.nobs,self.nparams,self.fit_intercept,'C++',d['solver'],d['rank'],self.link)
        lines+=['']+terminal_lines(self.info,self._llf if metric=='LogLik' else self.objective,metric,stopping_reason(self.info,d['control']['gtol'],d['control']['relgtol']))
        lines+=['']+self._inference_lines(max_rows)
        return '\n'.join(lines)
    def __repr__(self): return self.summary()


def fit(X, y=None, family='gaussian', *, data=None, start=None, beta0=None,
        fit_intercept=None, control=None, rank=None, solver='pcg',
        weights=None, freq_weights=None, var_weights=None, offset=None, **kwargs):
    """Generic GLM/residual fit. Formula parsing and matrices share _api.fit."""
    if any(v is not None for v in (weights,freq_weights,var_weights,offset)):
        raise NotImplementedError('observation weights and offset are not implemented by the numerical core')
    if start is not None and beta0 is not None: raise TypeError('use start or beta0, not both')
    opts=_options(control,kwargs)
    family,opts=_normalize_family(family,opts)
    design=None; formula=None
    if isinstance(X,str):
        if y is not None or data is None: raise TypeError('formula input requires data= and no separate y')
        if fit_intercept is not None: raise TypeError('the formula controls the intercept')
        import re
        if re.search(r'\boffset\s*\(', X): raise NotImplementedError('formula offsets are not implemented')
        from patsy import dmatrices
        response,design_matrix=dmatrices(X,data,NA_action='raise',return_type='matrix')
        if response.shape[1]!=1: raise ValueError('formula response must be a single numeric column')
        formula=X; design=design_matrix.design_info; names=list(design.column_names)
        fit_intercept='Intercept' in names
        y=np.asarray(response).ravel(); X=np.asarray(design_matrix)
        if fit_intercept: X=X[:,1:]
    else:
        if data is not None: raise TypeError('data= is only valid with a formula')
        fit_intercept=True if fit_intercept is None else fit_intercept
        if not hasattr(X,'shape'): X=np.asarray(X)
        columns=getattr(X,'columns',None)
        names=(['Intercept'] if fit_intercept else [])+(list(map(str,columns)) if columns is not None else [f'x{i+1}' for i in range(X.shape[1])])
    p=X.shape[1]+int(fit_intercept)
    if rank is not None: opts['rank']=rank
    raw=_api.fit(X,y,family=family,beta0=start if start is not None else beta0,fit_intercept=fit_intercept,solver=solver,**opts)
    canonical,link=_LEGACY.get(family,(family,None))
    settings={k:v.default for k,v in inspect.signature(_api.fit).parameters.items() if k in _NUMERIC and v.default is not inspect.Parameter.empty}
    defaults=_api._Options(); _api._library().smm_default_options(_api.C.byref(defaults))
    settings.update({k:getattr(defaults,k) for k,_ in defaults._fields_ if k in _NUMERIC})
    settings.update({k:v for k,v in opts.items() if k in _NUMERIC})
    settings.update(solver={'spectral':'pcg','cholesky':'cho'}.get(solver,solver),rank=rank if rank not in (None,0) else (p-1 if p<20 else 10))
    result=Results(raw,X,y,names,canonical,link,formula,design,fit_intercept,settings,opts.get('dispersion'))
    if opts.get("verbose",False): print("\n"+"\n".join(result._inference_lines()))
    return result


def glm(X, y=None, *, data=None, family='gaussian', link=None, **kwargs):
    """Conventional GLMs. Binary responses are 0/1; grouped binomial is not implemented here."""
    aliases={'bernoulli':('binomial','logit'),'probit':('binomial','probit'),
             'gaussian_log':('gaussian','log'),'gamma_inverse':('gamma','inverse')}
    if family in aliases:
        family,oldlink=aliases[family]
        if link is not None and link!=oldlink: raise ValueError('legacy family alias conflicts with link')
        link=oldlink
    link=_DEFAULT.get(family) if link is None else link
    if (family,link) not in _GLM: raise ValueError('unsupported GLM family/link; use fit for residual models')
    if family=='binomial' and 'trials' in (kwargs.get('family_options') or {}):
        raise NotImplementedError('glm currently supports binary responses only; grouped counts remain available through legacy fit')
    return fit(X,y,data=data,family=_GLM[family,link],**kwargs)
