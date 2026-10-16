#!/usr/bin/env python3
"""Shared benchmark imports, binary input loader, and result writer."""
import os
# Set before importing numeric runtimes; threadpoolctl also enforces Python BLAS.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]=os.environ.get('SPECTRALMM_BENCH_THREADS','12')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
import numpy as np
from scipy import sparse, special
import scipy
import statsmodels.api as sm
import statsmodels
from threadpoolctl import threadpool_limits, threadpool_info
ROOT=Path(__file__).resolve().parents[2]
from spectralmm import fit
from spectralmm import _api as spectralmm_api
METHODS=['Spectral-MM (PCG)','Spectral-MM (MM)','CG','CGLS','CRLS','LSQR','LSMR','GLM']
SOLVERS=dict(zip(METHODS[:-1],['pcg','mm','cg','cgls','crls','lsqr','lsmr']))
FIELDS=['language','family','storage','n','p','p_total','method','median_ms','min_ms','max_ms',
        'samples','batch','converged','quality_pass','iterations','inner_iterations','loss',
        'gradnorm','relgrad','coef_norm','relative_coef_error','glm_input','warnings','error']



def load_case(out,case):
    base=out/'data'/case['id']; n=case['n']; p=case['p_total']
    y=np.fromfile(str(base)+'_y.bin',dtype='<f8')
    if case['storage']=='dense':
        X=np.fromfile(str(base)+'_X.bin',dtype='<f8').reshape((n,p),order='F')
    else:
        X=sparse.csc_matrix((np.fromfile(str(base)+'_values.bin',dtype='<f8'),
                            np.fromfile(str(base)+'_indices.bin',dtype='<i4'),
                            np.fromfile(str(base)+'_indptr.bin',dtype='<i4')),shape=(n,p))
    return X,y



def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)


def native_options(config):
    """Identical native settings for initial and repeat runs."""
    keys=('rank','krylovdim','floor','maxiter','inner_maxiter','eta_max',
          'correction_tol','resid_tol','gtol','relgtol','krylov_rtol','krylov_atol',
          'accept_negligible','accept_stalled')
    result={key:config[key] for key in keys}
    result.update(ridge=0.,nesterov=True,fit_intercept=False,inference=False)
    return result

PARAMS={'negative_binomial': {'theta':4.}, 'smooth_quantile':{'tau':.25,'smoothing':.1},
        'expectile':{'tau':.25},'pseudo_huber':{'delta':1.},'student_t':{'nu':4.,'sigma':1.}}
FAMILIES=['gaussian','bernoulli','probit','poisson','gamma','negative_binomial',
          'smooth_quantile','expectile','pseudo_huber','student_t']
GLM_FAMILIES=set(FAMILIES[:6])
