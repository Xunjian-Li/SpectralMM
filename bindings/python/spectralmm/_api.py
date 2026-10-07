"""SpectralMM statistical estimation with the shared C++ core."""
from __future__ import annotations
import ctypes as C
from dataclasses import dataclass, field
from functools import lru_cache
import operator
import os
from pathlib import Path
import sys
import warnings
import numpy as np

_D = C.POINTER(C.c_double)
_I = C.POINTER(C.c_int64)

class _Matrix(C.Structure):
    _fields_ = [(k, C.c_int64) for k in ('n', 'p', 'nnz')] + [
        ('storage', C.c_int32), ('values', _D), ('indptr', _I), ('indices', _I)]

class _Options(C.Structure):
    _fields_ = [('family', C.c_int32), ('nesterov', C.c_int32)] + [
        (k, C.c_int64) for k in ('rank', 'maxiter', 'inner_maxiter', 'krylovdim')] + [
        (k, C.c_double) for k in ('ridge', 'floor', 'gtol', 'relgtol', 'eta_max', 'correction_tol', 'resid_tol')]

class _Info(C.Structure):
    _fields_ = [(k, C.c_int64) for k in ('iterations', 'inner_iterations', 'restarts', 'corrections')] + [
        ('converged', C.c_int32), ('termination', C.c_int32)] + [
        (k, C.c_double) for k in ('loss', 'gradnorm', 'relgradnorm', 'eigresidual')]

class _Krylov(C.Structure):
    _fields_ = [('solver', C.c_int32), ('jacobi', C.c_int32),
                ('rtol', C.c_double), ('atol', C.c_double)]

class _Stop(C.Structure):
    _fields_ = [('accept_negligible', C.c_int32), ('accept_stalled', C.c_int32)] + [
        (k, C.c_double) for k in ('negligible_step_tol', 'step_reltol', 'stalled_relgtol')]

class _Trace(C.Structure):
    _fields_ = [(k, C.c_int64) for k in ('iteration', 'inner_iterations', 'restarts', 'corrections', 'inner')] + [
        (k, C.c_double) for k in ('loss', 'gradnorm', 'relgradnorm', 'inner_residual', 'eigresidual', 'step')] + [
        ('spectrum', C.c_int32)]

_SPECTRUM = ('-', 'initial', 'reuse', 'correct', 'restart', 'correct-fail', 'restart+correct', 'restart+fail')

def _print_trace(history, info, solver, rank, family, options, n, p, intercept, gtol, relgtol):
    from ._display import model_lines, terminal_lines, stopping_reason
    label = 'PCG' if solver == 'spectral' else solver.upper()
    metric = 'LogLik' if not np.isnan(history[-1]['loglikelihood']) else 'Objective'
    key = 'loglikelihood' if metric == 'LogLik' else 'loss'
    print("\n"+"\n".join(model_lines(family,options,n,p,intercept,"C++",solver,rank))+"\n")
    print(f"{'Iter':>4}  {metric:>12}  {'GradNorm':>10}  {'RelGrad':>10}  {'Inner':>5}  {'Stepsize':>8}  Spectrum")
    for r in history:
        inner = '-' if r['inner'] < 0 else str(r['inner'])
        step = f"{r['step']:.2f}" if np.isfinite(r['step']) else '-'
        value = '-' if np.isnan(r[key]) else f"{r[key]:.4e}"
        print(f"{r['iteration']:4d}  {value:>12}  {r['gradnorm']:10.3e}  {r['relgradnorm']:10.3e}  {inner:>5}  {step:>8}  {r['spectrum']}")
    print("\n"+"\n".join(terminal_lines(info,history[-1][key],metric,stopping_reason(info,gtol,relgtol))))
    print('LogLik includes distribution constants and excludes ridge; gradients refer to the optimization objective.' if metric == 'LogLik' else 'Objective is the summed model loss plus ridge penalty.')

_REASON_LABELS = ('gradient', 'maximum iterations reached', 'line search failed', 'inner breakdown', 'negligible step', 'stalled near tolerance')
_REASONS = ('gradient', 'maxiter', 'line_search_failed', 'inner_breakdown', 'negligible_step', 'stalled_step')
_KRYLOV = {'cg': 0, 'cgls': 1, 'crls': 2, 'lsqr': 3, 'lsmr': 4, 'cho': 5, 'mm': 6}
_FAMILIES = {name: i for i, name in enumerate((
    'gaussian', 'bernoulli', 'probit', 'poisson', 'gamma', 'negative_binomial',
    'smooth_quantile', 'expectile', 'pseudo_huber', 'student_t', 'gaussian_log',
    'gamma_inverse', 'tweedie', 'binomial'))}
_FAMILY_PARAMS = {'negative_binomial': {'theta'}, 'smooth_quantile': {'tau', 'smoothing'},
    'expectile': {'tau'}, 'pseudo_huber': {'delta'}, 'student_t': {'nu', 'sigma'},
    'tweedie': {'power'}, 'binomial': {'trials'}}
class _Family(C.Structure):
    _fields_ = [(k, C.c_double) for k in ('theta', 'tau', 'smoothing', 'delta', 'nu', 'sigma', 'power')] + [
        ('trials', _D), ('trials_count', C.c_int64)]


class _InferenceInfo(C.Structure):
    _fields_ = [('cov_type', C.c_int32), ('estimated_dispersion', C.c_int32)] + [
        (k, C.c_double) for k in ('df_resid', 'dispersion', 'rcond')]


def _inference_options(inference, max_p, cov_type, level, use_t, dispersion):
    if not (isinstance(inference, (bool, np.bool_)) or inference == 'auto'):
        raise ValueError("inference must be 'auto', True, or False")
    if isinstance(max_p, bool) or operator.index(max_p) < 1:
        raise ValueError('inference_max_p must be a positive integer')
    if cov_type not in ('auto', 'model', 'sandwich'):
        raise ValueError('cov_type must be auto, model, or sandwich')
    if not 0 < level < 1:
        raise ValueError('level must be in (0,1)')
    if use_t is not None and not isinstance(use_t, (bool, np.bool_)):
        raise ValueError('use_t must be None or boolean')
    if dispersion is not None and not (np.isfinite(dispersion) and dispersion > 0):
        raise ValueError('dispersion must be positive and finite')


def _postfit_inference(lib, desc, y, coef, opts, fp, intercept, info,
                       inference, max_p, cov_type, level, use_t, dispersion):
    def unavailable(reason, status='unavailable'):
        if isinstance(inference, (bool, np.bool_)) and inference:
            warnings.warn('Inference unavailable: '+reason, UserWarning, stacklevel=3)
        return dict(status=status, reason=reason)
    if isinstance(inference, (bool, np.bool_)) and not inference:
        return dict(status='disabled', reason='inference=False')
    if inference == 'auto' and coef.size > max_p:
        return dict(status='skipped', reason=f'parameter count {coef.size} exceeds inference_max_p={max_p}')
    if opts.ridge != 0:
        return unavailable('ordinary Wald inference is unavailable for ridge-penalized fits')
    if not info['gradient_converged']:
        return unavailable('fit did not satisfy the gradient convergence criterion')
    if desc.n <= coef.size:
        return unavailable('inference requires n > number of parameters')
    try:
        from scipy import stats
    except ImportError:
        return unavailable('SciPy is required for inference; install scipy or set inference=False')
    covariance = np.empty((coef.size, coef.size), dtype=np.float64, order='F')
    meta, error = _InferenceInfo(), C.create_string_buffer(1024)
    code = lib.smm_infer(C.byref(desc), y.ctypes.data_as(_D), coef.ctypes.data_as(_D),
        C.byref(opts), C.byref(fp), int(intercept), ('auto','model','sandwich').index(cov_type),
        np.nan if dispersion is None else dispersion, covariance.ctypes.data_as(_D), C.byref(meta), error, len(error))
    if code:
        return unavailable(error.value.decode())
    student = bool(meta.estimated_dispersion) if use_t is None else bool(use_t)
    dist = stats.t(meta.df_resid) if student else stats.norm
    se = np.sqrt(covariance.diagonal())
    statistic = coef/se
    critical = dist.isf((1-level)/2)
    return dict(status='ok', reason='', covariance=covariance, std_error=se,
        statistic=statistic, statistic_type='t' if student else 'z',
        wald_chisq=statistic**2, wald_p_value=stats.chi2.sf(statistic**2,1),
        p_value=2*dist.sf(np.abs(statistic)),
        conf_int=np.column_stack((coef-critical*se, coef+critical*se)), level=level,
        df_resid=meta.df_resid, dispersion=meta.dispersion, rcond=meta.rcond,
        cov_type='model' if meta.cov_type==1 else 'sandwich',
        estimated_dispersion=bool(meta.estimated_dispersion))


_LIBRARY_SUFFIX = '.dylib' if sys.platform == 'darwin' else ('.dll' if sys.platform == 'win32' else '.so')
_LIBRARY_NAME = ('spectralmm' if sys.platform == 'win32' else 'libspectralmm') + _LIBRARY_SUFFIX
_DEFAULT_LIBRARY = str(Path(__file__).resolve().parent / '_lib' / _LIBRARY_NAME)



def _library():
    path = os.environ.get('SPECTRALMM_LIBRARY', _DEFAULT_LIBRARY)
    if not Path(path).is_file():
        raise RuntimeError('SpectralMM library is missing; reinstall the package with pip or check SPECTRALMM_LIBRARY.')
    return _load_library(os.path.abspath(path))


@lru_cache(maxsize=None)
def _load_library(path):
    # Keep the library and configured function interfaces alive across fits.
    lib = C.CDLL(path)
    lib.smm_abi_version.restype = C.c_int32
    if lib.smm_abi_version() != 1:
        raise RuntimeError('SpectralMM native ABI mismatch (expected 1)')
    lib.smm_default_options.argtypes = [C.POINTER(_Options)]
    lib.smm_default_options.restype = None
    lib.smm_fit.argtypes = [C.POINTER(_Matrix), _D, _D, C.POINTER(_Options), _D,
                            C.POINTER(_Info), C.POINTER(C.c_char), C.c_size_t]
    lib.smm_fit.restype = C.c_int32
    lib.smm_default_family_options.argtypes = [C.POINTER(_Family)]
    lib.smm_default_family_options.restype = None
    lib.smm_fit_logged_stats.argtypes = [C.POINTER(_Matrix), _D, _D, C.POINTER(_Options), C.POINTER(_Family),
        C.POINTER(_Krylov), C.POINTER(_Stop), _D, C.POINTER(_Info), C.POINTER(_Trace),
        C.c_int64, _I, C.POINTER(C.c_char), C.c_size_t, C.c_int32, C.c_int32, _D, C.c_double]
    lib.smm_fit_logged_stats.restype = C.c_int32
    lib.smm_infer.argtypes = [C.POINTER(_Matrix), _D, _D, C.POINTER(_Options), C.POINTER(_Family),
        C.c_int32, C.c_int32, C.c_double, _D, C.POINTER(_InferenceInfo), C.POINTER(C.c_char), C.c_size_t]
    lib.smm_infer.restype = C.c_int32
    return lib

@dataclass
class Model:
    coef: np.ndarray
    family: str
    info: dict
    trace: list = None
    family_options: dict = field(default_factory=dict)

    fit_intercept: bool = False
    inference: dict = field(default_factory=lambda: dict(status='disabled', reason='not requested'))

    def _inference_result(self):
        if self.inference['status'] != 'ok':
            raise ValueError('Inference unavailable: '+self.inference['reason'])
        return self.inference

    @property
    def bse(self): return self._inference_result()['std_error']

    @property
    def pvalues(self): return self._inference_result()['p_value']

    @property
    def tvalues(self): return self._inference_result()['statistic']

    def cov_params(self): return self._inference_result()['covariance'].copy()

    def conf_int(self): return self._inference_result()['conf_int'].copy()

    def __repr__(self):
        return self.summary()

    def summary(self, max_rows=20):
        if isinstance(max_rows, bool) or operator.index(max_rows) < 1:
            raise ValueError('max_rows must be a positive integer')
        result=self.inference
        if result['status'] != 'ok':
            return f"SpectralMM ({self.family}), {self.coef.size} parameters\nInference {result['status']}: {result['reason']}"
        lines=[f"SpectralMM ({self.family}); covariance={result['cov_type']}; reference={result['statistic_type']}",
               f'Term               Estimate    Std. Error   {result["statistic_type"]} statistic      p-value        lower        upper']
        names=(['(Intercept)'] if self.fit_intercept else [])+[f'x{i+1}' for i in range(self.coef_.size)]
        for i in range(min(max_rows,self.coef.size)):
            vals=[self.coef[i],result['std_error'][i],result['statistic'][i],result['p_value'][i],*result['conf_int'][i]]
            lines.append(f'{names[i]:<16}'+''.join(f'{x:13.6g}' for x in vals))
        if self.coef.size > max_rows: lines.append(f'... {self.coef.size-max_rows} more coefficients; full arrays are in model.inference')
        return '\n'.join(lines)

    @property
    def intercept_(self):
        return float(self.coef[0]) if self.fit_intercept else 0.0

    @property
    def coef_(self):
        return self.coef[1:] if self.fit_intercept else self.coef

    def predict(self, X, *, trials=None):
        """Response prediction; residual models return fitted location/quantile/expectile.

        Binomial returns expected counts. Supply trials for new rows when training
        used a vector of trial counts.
        """
        if not hasattr(X, 'shape'):
            X = np.asarray(X)
        if len(X.shape) != 2 or X.shape[1] != self.coef_.size:
            raise ValueError('prediction X must have the same feature columns as training X')
        eta = np.asarray(X @ self.coef_).reshape(-1) + self.intercept_
        if self.family in ('gaussian', 'smooth_quantile', 'expectile', 'pseudo_huber', 'student_t'):
            return eta
        if self.family == 'probit':
            from scipy.special import ndtr
            return ndtr(eta)
        if self.family == 'gamma_inverse':
            return 1 / np.maximum(eta, np.sqrt(np.finfo(float).eps))
        if self.family not in ('bernoulli', 'binomial'):
            return np.exp(np.minimum(eta, 700))
        result = np.empty_like(eta)
        pos = eta >= 0
        result[pos] = 1 / (1 + np.exp(-eta[pos]))
        z = np.exp(eta[~pos]); result[~pos] = z / (1 + z)
        if self.family == 'binomial':
            n = np.asarray(self.family_options.get('trials', 1.) if trials is None else trials, dtype=float)
            if n.ndim > 1 or n.size not in (1, eta.size) or not np.all(np.isfinite(n) & (n > 0)):
                raise ValueError('prediction trials must be positive scalar or length n')
            result *= n.reshape(-1)
        return result


def fit(X, y, family='gaussian', *, family_options=None, beta0=None, fit_intercept=True,
        penalize_intercept=False, inference='auto', inference_max_p=50,
        cov_type='auto', level=.95, use_t=None, dispersion=None, solver='spectral',
        krylov_rtol=1e-4, krylov_atol=1e-8, preconditioner='jacobi', accept_negligible=True, accept_stalled=True,
        negligible_step_tol=1e-14, step_reltol=np.sqrt(np.finfo(float).eps),
        stalled_relgtol=1e-4, trace=False, verbose=False, **options):
    """Fit float64 dense or SciPy sparse features; add an intercept by default.

    Solvers: spectral/pcg, mm, cho/cholesky, cg, cgls, crls, lsqr, lsmr.
    Cholesky forms a full curvature matrix; the other methods are matrix-free.

    Use fit_intercept=False for a complete, already augmented design matrix.
    beta0 and model.coef include the intercept first; coef_ excludes it and
    intercept_ returns it separately. Ridge excludes an automatically added
    intercept unless penalize_intercept=True. Formula input is not supported.

    inference='auto' computes post-fit inference for at most inference_max_p
    parameters (including intercept). True bypasses the size cutoff; False
    disables it. Requires SciPy, an unpenalized gradient-converged fit, n>p and
    identifiable statistical curvature. See docs/README.md for assumptions.

    See cpp/README.md for options and convergence semantics. Nonconvergence is
    returned in model.info; invalid inputs raise ValueError. A conversion copy
    may be needed for dtype/layout or CSC normalization.
    """
    if not isinstance(trace, (bool, np.bool_)) or not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("trace and verbose must be boolean")
    _inference_options(inference, inference_max_p, cov_type, level, use_t, dispersion)
    if family not in _FAMILIES:
        raise ValueError('unknown family; choose from ' + ', '.join(_FAMILIES))
    solver = {'pcg': 'spectral', 'cholesky': 'cho'}.get(solver, solver)
    if solver != 'spectral' and solver not in _KRYLOV:
        raise ValueError('unknown solver')
    if preconditioner not in ('jacobi', 'none'):
        raise ValueError('preconditioner must be jacobi or none')
    if not isinstance(fit_intercept, (bool, np.bool_)) or not isinstance(penalize_intercept, (bool, np.bool_)):
        raise ValueError('intercept flags must be boolean')
    lib = _library()
    opts = _Options()
    lib.smm_default_options(C.byref(opts))
    opts.family = _FAMILIES[family]
    family_options = dict(family_options or {})
    if "q" in family_options:
        if "tau" in family_options: raise TypeError("supply only q, not both q and tau")
        family_options["tau"] = family_options.pop("q")
    unknown = set(family_options) - _FAMILY_PARAMS.get(family, set())
    if unknown:
        raise TypeError(f'unknown parameters for {family}: {sorted(unknown)}')
    fp = _Family()
    lib.smm_default_family_options(C.byref(fp))
    trial_counts = None
    for key, value in family_options.items():
        if key == 'trials':
            if np.iscomplexobj(value): raise ValueError('trials must be real')
            trial_counts = np.asarray(value, dtype=np.float64)
            if trial_counts.ndim > 1: raise ValueError('trials must be scalar or vector')
            trial_counts = np.ascontiguousarray(trial_counts.reshape(-1))
            fp.trials = trial_counts.ctypes.data_as(_D)
            fp.trials_count = trial_counts.size
            family_options[key] = trial_counts.copy()
        else:
            setattr(fp, key, float(value))
    fields = dict(_Options._fields_)
    for key, value in options.items():
        if key not in fields or key == 'family':
            raise TypeError(f'unknown option: {key}')
        if fields[key] is not C.c_double:
            value = operator.index(value)
            bits = C.sizeof(fields[key]) * 8
            if not -(2**(bits-1)) <= value < 2**(bits-1):
                raise ValueError(f'{key} is outside integer range')
        setattr(opts, key, value)
    try:
        from scipy.sparse import issparse
    except ImportError:
        issparse = lambda _: False
    if issparse(X):
        if np.iscomplexobj(X.data):
            raise ValueError('X must be real')
        X = X.tocsc(copy=False).astype(np.float64, copy=False)
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
            X.sort_indices()
        values = np.ascontiguousarray(X.data)
        indptr = np.ascontiguousarray(X.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(X.indices, dtype=np.int64)
        desc = _Matrix(*X.shape, X.nnz, 1, values.ctypes.data_as(_D),
                       indptr.ctypes.data_as(_I), indices.ctypes.data_as(_I))
    else:
        if np.iscomplexobj(X):
            raise ValueError('X must be real')
        X = np.asarray(X, dtype=np.float64, order='F')
        if X.ndim != 2:
            raise ValueError('X must be two-dimensional')
        desc = _Matrix(*X.shape, 0, 0, X.ctypes.data_as(_D), None, None)
    if fit_intercept:
        if issparse(X):
            constant = any(X.indptr[j+1]-X.indptr[j] == X.shape[0] and X.shape[0] > 0
                and X.data[X.indptr[j]] != 0
                and np.all(X.data[X.indptr[j]:X.indptr[j+1]] == X.data[X.indptr[j]])
                for j in range(X.shape[1]))
        else:
            constant = X.shape[0] > 0 and any(X[0,j] != 0 and (X.shape[0] == 1 or X[1,j] == X[0,j])
                and np.all(X[:,j] == X[0,j])
                for j in range(X.shape[1]))
        if constant:
            warnings.warn('X contains a nonzero constant column; remove it or use fit_intercept=False '
                          'to avoid a redundant intercept.', UserWarning, stacklevel=2)
    p = X.shape[1] + int(fit_intercept)
    if np.iscomplexobj(y):
        raise ValueError('y must be real')
    y = np.ascontiguousarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size != X.shape[0]:
        raise ValueError('y must be a vector of length n')
    bptr = None
    if beta0 is not None:
        if np.iscomplexobj(beta0):
            raise ValueError('beta0 must be real')
        beta0 = np.ascontiguousarray(beta0, dtype=np.float64)
        if beta0.ndim != 1 or beta0.size != p:
            raise ValueError('beta0 must have one value per parameter, including the intercept first')
        bptr = beta0.ctypes.data_as(_D)
    coef = np.empty(p, dtype=np.float64)
    info, error = _Info(), C.create_string_buffer(1024)
    for flag in (accept_negligible, accept_stalled):
        if operator.index(flag) not in (0, 1):
            raise ValueError('stop flags must be boolean')
    stops = _Stop(accept_negligible, accept_stalled, negligible_step_tol, step_reltol, stalled_relgtol)
    settings = None if solver == 'spectral' else _Krylov(
        _KRYLOV[solver], preconditioner == 'jacobi', krylov_rtol, krylov_atol)
    if opts.maxiter <= 0:
        raise ValueError('maxiter must be positive')
    capture = trace or verbose
    history = (_Trace * (opts.maxiter + 1))() if capture else None
    likelihood = (C.c_double * (opts.maxiter + 1))() if capture else None
    size = C.c_int64()
    code = lib.smm_fit_logged_stats(C.byref(desc), y.ctypes.data_as(_D), bptr, C.byref(opts), C.byref(fp),
        C.byref(settings) if settings is not None else None, C.byref(stops),
        coef.ctypes.data_as(_D), C.byref(info), history, len(history) if capture else 0,
        C.byref(size), error, len(error), int(fit_intercept), int(penalize_intercept), likelihood, np.nan if dispersion is None else dispersion)
    if code:
        raise ValueError(error.value.decode('utf-8'))
    result = {k: getattr(info, k) for k, _ in _Info._fields_}
    result['converged'] = bool(result['converged'])
    result['gradient_converged'] = info.gradnorm <= opts.gtol or info.relgradnorm <= opts.relgtol
    result['termination_reason'] = _REASONS[info.termination]
    history = [{k: getattr(history[i], k) for k, _ in _Trace._fields_} for i in range(size.value)] if capture else None
    if capture:
        for i, row in enumerate(history):
            row["loglikelihood"] = likelihood[i]
            row["spectrum"] = _SPECTRUM[row["spectrum"]]
    if verbose:
        _print_trace(history, result, solver, opts.rank or (p-1 if p<20 else 10),family,family_options,X.shape[0],p,fit_intercept,opts.gtol,opts.relgtol)
    inference_result = _postfit_inference(lib, desc, y, coef, opts, fp, fit_intercept, result,
        inference, inference_max_p, cov_type, level, use_t, dispersion)
    if "tau" in family_options: family_options["q"] = family_options.pop("tau")
    return Model(coef, family, result, history if trace else None, family_options, bool(fit_intercept), inference_result)
