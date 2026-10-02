#!/usr/bin/env python3
"""Benchmark article-style Julia-generated inputs, all article families."""
from common import *
import shutil
PARAMS={'negative_binomial': {'theta':4.}, 'smooth_quantile':{'tau':.25,'smoothing':.1},
        'expectile':{'tau':.25},'pseudo_huber':{'delta':1.},'student_t':{'nu':4.,'sigma':1.}}
FAMILIES=['gaussian','bernoulli','probit','poisson','gamma','negative_binomial',
          'smooth_quantile','expectile','pseudo_huber','student_t']
GLM_FAMILIES=set(FAMILIES[:6])
FIELDS=FIELDS+['termination']
def glm_family(family):
    return {'gaussian':lambda:sm.families.Gaussian(), 'bernoulli':lambda:sm.families.Binomial(),
        'probit':lambda:sm.families.Binomial(link=sm.families.links.Probit()),
        'poisson':lambda:sm.families.Poisson(), 'gamma':lambda:sm.families.Gamma(link=sm.families.links.Log()),
        'negative_binomial':lambda:sm.families.NegativeBinomial(alpha=.25)}[family]()
def evaluate(X,y,b,family,scale):
    # Scoring is outside timing; this native score was checked against Julia.
    info=fit(X,y,family,family_options=PARAMS.get(family,{}),beta0=b,gtol=1e100, fit_intercept=False, inference=False).info
    return info['loss'],info['gradnorm'],info['gradnorm']/scale
def benchmark_python(out,config):
    rows=[]; timings=[]
    for case in config['cases']:
        X,y=load_case(out,case); family=case['family']
        b0=np.fromfile(out/'data'/f"{case['id']}_beta0.bin",dtype='<f8')
        scale=1+evaluate(X,y,b0,family,1)[1]
        coefs={}; case_rows=[]
        # Fixed, reproducible shuffled order reduces systematic ordering bias.
        order=np.random.default_rng(case['seed']+17).permutation(METHODS if family in GLM_FAMILIES else METHODS[:-1])
        for method in order:
            row={k:case[k] for k in ('family','storage','n','p','p_total')}
            row.update(language='Python',method=method,glm_input='dense conversion included' if method=='GLM' and sparse.issparse(X) else case['storage'],error='',warnings='')
            if method=='GLM':
                def run():
                    XX=X.toarray(order='F') if sparse.issparse(X) else X
                    fam=glm_family(family)
                    result=sm.GLM(y,XX,family=fam).fit(start_params=b0,method='IRLS',
                        maxiter=config['maxiter'],tol=config['glm_tolerance'],atol=config['glm_tolerance'],rtol=0.)
                    return result.params,bool(result.converged),result.fit_history.get('iteration',-1),-1,'GLM'
            else:
                def run():
                    result=fit(X,y,family,family_options=PARAMS.get(family,{}),solver=SOLVERS[method],beta0=b0,rank=config['rank'],
                        floor=config.get('floor',1e-6),krylovdim=config['krylovdim'],maxiter=config['maxiter'],inner_maxiter=config['inner_maxiter'],
                        gtol=config['gtol'],relgtol=config['relgtol'],eta_max=config['eta_max'],
                        correction_tol=config['correction_tol'],resid_tol=config['resid_tol'],
                        krylov_rtol=config['krylov_rtol'],krylov_atol=config['krylov_atol'],ridge=0.,nesterov=True, fit_intercept=False, inference=False)
                    i=result.info
                    return result.coef,i['converged'],i['iterations'],i['inner_iterations'],i['termination_reason']
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    start=time.perf_counter(); result=run(); warm=time.perf_counter()-start
                notes=sorted(set(str(w.message) for w in caught))
                # Any model warnings remain in the CSV and can disqualify a row.
                row['warnings']=' | '.join(notes)
                batch=max(1,min(config['max_batch'],int(np.ceil(config['min_sample_seconds']/max(warm,1e-6)))))
                # Calibrate batches outside the reported samples. R's elapsed
                # timer has millisecond resolution, so both languages use >=50ms
                # batches where practical instead of timing tiny single calls.
                while warm<config['min_sample_seconds']:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        start=time.perf_counter()
                        for _ in range(batch): result=run()
                        duration=time.perf_counter()-start
                    if duration>=config['min_sample_seconds'] or batch>=config['max_batch']: break
                    batch=min(config['max_batch'],max(batch+1,int(np.ceil(batch*config['min_sample_seconds']/max(duration,1e-6)))))
                times=[]
                for sample in range(config['samples']):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore') # warning formatting is not part of fit timing
                        start=time.perf_counter()
                        for _ in range(batch): result=run()
                        elapsed=(time.perf_counter()-start)*1000/batch
                    times.append(elapsed)
                    timings.append(dict(language='Python',case=case['id'],method=method,sample=sample+1,batch=batch,time_ms=elapsed))
                b,converged,outer,inner,termination=result
                loss,gn,rel=evaluate(X,y,b,family,scale)
                quality=bool(np.all(np.isfinite(b)) and (gn<=config['gtol'] or rel<=config['relgtol']))
                if any('separation' in n.lower() for n in notes): quality=False
                row.update(median_ms=float(np.median(times)),min_ms=min(times),max_ms=max(times),samples=len(times),batch=batch,
                    converged=converged,quality_pass=quality,termination=termination,iterations=outer,inner_iterations=inner,
                    loss=loss,gradnorm=gn,relgrad=rel,coef_norm=float(np.linalg.norm(b)))
                coefs[method]=b
            except Exception as e:
                row.update(error=repr(e),quality_pass=False,converged=False)
            case_rows.append(row)
            print(f"Python {case['id']} {method}: {row.get('median_ms',float('nan')):.3f} ms, quality={row['quality_pass']}",flush=True)
        ref=np.fromfile(out/'data'/f"{case['id']}_julia_coef.bin",dtype='<f8')
        for row in case_rows:
            if row['method'] in coefs:
                row['relative_coef_error']=float(np.linalg.norm(coefs[row['method']]-ref)/max(1.,np.linalg.norm(ref)))
        rows.extend(case_rows)
        write_csv(out/'python.csv',rows,FIELDS)
        write_csv(out/'python_samples.csv',timings,list(timings[0]))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'benchmark/native/results/family-performance')
    parser.add_argument('--language',choices=['python','R','both'],default='both')
    parser.add_argument('--size',type=int,default=1000)
    args=parser.parse_args(); out=args.output.resolve(); p=args.size
    cases=[]
    for family in FAMILIES:
        for storage in ('dense','sparse'):
            id=f'{family}_{storage}_p{p}'
            nnz=(out/'data'/f'{id}_values.bin').stat().st_size//8 if storage=='sparse' else 2*p*(p+1)
            cases.append(dict(id=id,family=family,storage=storage,n=2*p,p=p,p_total=p+1,nnz=nnz,seed=3124))
    config=dict(cases=cases,samples=3,min_sample_seconds=.05,max_batch=1000,maxiter=1000,
        inner_maxiter=200,rank=5,krylovdim=12,floor=1e-6,gtol=1e-6,relgtol=1e-8,
        eta_max=.6,correction_tol=.05,resid_tol=.5,krylov_rtol=1e-4,krylov_atol=1e-8,
        glm_tolerance=1e-12,threads=1,methods=METHODS)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    with threadpool_limits(limits=1):
        metadata=dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
            statsmodels=statsmodels.__version__,threadpools=threadpool_info(),
            native_sha256=hashlib.sha256(Path(spectralmm_api._library()._name).read_bytes()).hexdigest(),
            timing='warm end-to-end fit; excludes data generation/imports/compilation/scoring; sparse GLM conversion included')
        (out/'environment_python.json').write_text(json.dumps(metadata,indent=2))
        if args.language in ('both','python'): benchmark_python(out,config)
    if args.language in ('both','R'):
        subprocess.run([shutil.which('Rscript') or 'Rscript',str(ROOT/'benchmark/native/family_performance.R'),str(out),str(ROOT)],check=True,env=os.environ)
