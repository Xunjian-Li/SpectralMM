#!/usr/bin/env python3
"""Shared benchmark imports, binary input loader, and result writer."""
import os
# Set before importing numeric runtimes; threadpoolctl also enforces Python BLAS.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
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
METHODS=['Spectral-MM','CG','CGLS','CRLS','LSQR','LSMR','GLM']
SOLVERS=dict(zip(METHODS[:-1],['spectral','cg','cgls','crls','lsqr','lsmr']))
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
