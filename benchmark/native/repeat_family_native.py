"""Round-robin native timings after the full family/GLM run."""
from family_performance import *
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',type=Path,default=ROOT/'benchmark/native/results/family-performance')
 parser.add_argument('--language',choices=['both','python','R'],default='both')
 args=parser.parse_args(); out=args.output.resolve()
 config=json.loads((out/'config.json').read_text()); rows=[]; samples=[]
 with threadpool_limits(limits=1):
  for case in (config['cases'] if args.language in ('both','python') else []):
   X,y=load_case(out,case); b0=np.fromfile(out/'data'/f"{case['id']}_beta0.bin",dtype='<f8')
   def run(method):
    return fit(X,y,case['family'],family_options=PARAMS.get(case['family'],{}),beta0=b0,solver=SOLVERS[method],
      rank=5,krylovdim=12,floor=1e-6,ridge=0.,maxiter=1000,inner_maxiter=200,eta_max=.6,
      correction_tol=.05,resid_tol=.5,gtol=1e-6,relgtol=1e-8,nesterov=True, fit_intercept=False, inference=False)
   methods=METHODS[:-1]; batches={}; results={}; times={m:[] for m in methods}
   for method in methods:
    start=time.perf_counter(); run(method); warm=time.perf_counter()-start
    batches[method]=max(1,min(50,int(np.ceil(.05/max(warm,1e-6)))))
   rng=np.random.default_rng(718)
   for sample in range(5):
    for method in rng.permutation(methods):
     start=time.perf_counter()
     for _ in range(batches[method]): results[method]=run(method)
     ms=(time.perf_counter()-start)*1000/batches[method]; times[method].append(ms)
     samples.append(dict(language='Python',case=case['id'],method=method,sample=sample+1,batch=batches[method],time_ms=ms))
   for method in methods:
    info=results[method].info
    rows.append(dict(language='Python',family=case['family'],storage=case['storage'],method=method,
      median_ms=float(np.median(times[method])),min_ms=min(times[method]),max_ms=max(times[method]),
      quality_pass=info['gradient_converged'],converged=info['converged'],termination=info['termination_reason'],
      iterations=info['iterations'],inner_iterations=info['inner_iterations'],loss=info['loss'],gradnorm=info['gradnorm'],relgrad=info['relgradnorm']))
   write_csv(out/'python_repeat.csv',rows,list(rows[0])); write_csv(out/'python_repeat_samples.csv',samples,list(samples[0]))
   print(case['id'], 'completed',flush=True)
 if args.language in ('both','R'): subprocess.run([shutil.which('Rscript') or 'Rscript',str(ROOT/'benchmark/native/repeat_family_native.R'),str(out),str(ROOT)],check=True,env=os.environ)
