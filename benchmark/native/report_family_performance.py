"""Render completed family benchmark, retaining first-run and repeat evidence."""
import csv, json, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'benchmark/native/results/family-performance')
out=parser.parse_args().output.resolve()
config=json.loads((out/'config.json').read_text())
n=config['cases'][0]['n']; p_total=config['cases'][0]['p_total']
models=['gaussian','bernoulli','probit','poisson','gamma','negative_binomial','smooth_quantile','expectile','pseudo_huber','student_t']
names=dict(zip(models,['Gaussian','Logistic','Probit','Poisson','Gamma','Negative-binomial','SmoothQuantile','Expectile','PseudoHuber','Student-t']))
methods=['Spectral-MM','CG','CGLS','CRLS','LSQR','LSMR']
def read(name): return list(csv.DictReader(open(out/name)))
def good(r): return r['quality_pass'].lower()=='true'
def lookup(rows,f,s,m): return next(r for r in rows if (r['family'],r['storage'],r['method'])==(f,s,m))
def cell(r): return f"{float(r['median_ms']):.2f}"+('' if good(r) else '*')
py,rr=read('python_repeat.csv'),read('R_repeat.csv')
firstpy,firstr=read('python.csv'),read('R.csv'); julia=read('Julia.csv')
assert len(py)==len(rr)==120 and len(firstpy)==len(firstr)==132 and len(julia)==20
lines=['# Performance comparison of ten models from the article (2026-09-29)','',
f'This run covers ten article benchmark models with n={n}, p={p_total} (including the intercept), rather than the final article scale of n=10000, p=5001. Gaussian-log, Gamma-inverse, Tweedie and Binomial were not timed in this run.',
'Data generation and random seeds follow benchmark/scaling.jl and scaling_sparse.jl. Julia, Python and R share identical binary inputs and initial coefficients for each model.',
'', '## Timing and accuracy','',
'- rho=1e-6, ridge=0, rank=5, krylovdim=12, at most 1000 outer and 200 inner iterations, eta_max=0.6.',
'- Gradient criterion: absolute gradient norm <=1e-6 or relative gradient <=1e-8. Default step-based stopping remains enabled; solver-declared convergence does not imply gradient convergence.',
'- Initial run: warm up each method and report the median of three timings. Native repeat: five rounds per scenario, shuffling the six methods each round; short calls are timed in batches.',
'- Main tables use native repeat results; GLM and Julia use initial-run results. Raw records from both runs are retained.',
'- Timing includes wrapper conversion and model construction, but excludes loading, compilation, data generation and scoring. Numerical libraries are configured for one thread.',
'- Python GLM uses statsmodels.GLM; R uses glm.fit. Negative-binomial theta is fixed at 4, with Python alpha=0.25.',
'- GLM tolerance is 1e-12, but all methods are assessed using the same final gradient criterion. The four non-GLM models have no GLM comparator.',
'- Sparse inputs are converted to dense for GLM, with conversion included in timing. This is therefore not a comparison of sparse GLM implementations.',
'- `*` denotes failure of an accuracy/validity check and excludes a result from equal-accuracy speed rankings. All times are in milliseconds.',
'', '## Spectral-MM versus other native solvers','',
'Each ratio is the runtime of the fastest gradient-converged CG/CGLS/CRLS/LSQR/LSMR method divided by the Spectral-MM runtime. Values above 1 favor Spectral-MM. No ratio is reported when Spectral-MM fails the gradient criterion.',
'', '| Model | Python dense | Python sparse | R dense | R sparse |','|---|---:|---:|---:|---:|']
wins={}
for f in models:
 cells=[]
 for rows in (py,rr):
  for st in ('dense','sparse'):
   sp=lookup(rows,f,st,'Spectral-MM'); peers=[lookup(rows,f,st,m) for m in methods[1:]]; peers=[r for r in peers if good(r)]
   if not good(sp): cells.append('Gradient criterion not met'); continue
   best=min(peers,key=lambda r:float(r['median_ms']))
   ratio=float(best['median_ms'])/float(sp['median_ms']); cells.append(f'{ratio:.2f}x ({best["method"]})')
 lines.append('| '+names[f]+' | '+' | '.join(cells)+' |')
for label,rows in [('Python',py),('R',rr)]:
 lines += ['',f'## {label}: complete native repeat timings','', '| Model | Matrix | Spectral-MM | CG | CGLS | CRLS | LSQR | LSMR |','|---|---|---:|---:|---:|---:|---:|---:|']
 for f in models:
  for st in ('dense','sparse'):
   lines.append('| '+names[f]+' | '+st+' | '+' | '.join(cell(lookup(rows,f,st,m)) for m in methods)+' |')
lines += ['', '## GLM comparisons (initial run)','', '| Model | Matrix | Python GLM | R GLM |','|---|---|---:|---:|']
for f in models[:6]:
 for st in ('dense','sparse'): lines.append('| '+names[f]+' | '+st+' | '+cell(lookup(firstpy,f,st,'GLM'))+' | '+cell(lookup(firstr,f,st,'GLM'))+' |')
lines += ['', '## Julia Spectral-MM baseline (initial run)','', '| Model | Dense | Sparse |','|---|---:|---:|']
for f in models: lines.append('| '+names[f]+' | '+cell(lookup(julia,f,'dense','Spectral-MM'))+' | '+cell(lookup(julia,f,'sparse','Spectral-MM'))+' |')
lines += ['', '## Interpretation and limitations','',
'For Logistic/Probit, statsmodels detected complete separation or perfect prediction, while R reported fitted probabilities of 0/1. Even a small gradient does not establish a unique finite maximum-likelihood estimate. These timings measure attainment of numerical stopping criteria.',
'', 'Both Gamma GLM implementations exhausted 1000 iterations without meeting the criterion. Their long runtimes cannot be used as successful-solve speedup denominators. Some negative-binomial GLM results also failed the common gradient threshold.',
'', 'Several initial timings varied by multiples between minimum and maximum. Repeats alternate execution order to reduce order effects. Python and R share the same C++ library, so differences between measurement periods cannot be attributed directly to language performance. Julia was also timed in a different period.',
'', 'Fixed parameters for the four non-GLM models: SmoothQuantile(tau=.25, smoothing=.1), Expectile(tau=.25), PseudoHuber(delta=1), Student-t(nu=4, sigma=1). Student-t uses MM surrogate weights.',
'', '## Spectral-MM results failing the gradient criterion','', '| Interface | Model | Matrix | Termination reason | Relative gradient |','|---|---|---|---|---:|']
for label,rows in [('Python',py),('R',rr)]:
 for r in rows:
  if r['method']=='Spectral-MM' and not good(r): lines.append(f"| {label} | {names[r['family']]} | {r['storage']} | {r['termination']} | {float(r['relgrad']):.3g} |")
lines += ['', '## Raw records and reproduction','',
'- `python.csv` / `R.csv` / `Julia.csv`: initial-run summaries.',
'- `python_repeat.csv` / `R_repeat.csv`: native repeat summaries.',
'- `*_samples.csv`: individual timings, including batch counts.',
'- `config.json` / `environment_*`: configuration and runtime environment; the Python record includes the shared-library SHA256.',
'- Binary inputs can be regenerated. The cleaned archive retains results and excludes `data/`.','',
'```sh',
'BENCH_OUT=build/family-benchmark',
'julia --compiled-modules=existing --project=. benchmark/native/family_performance.jl 1000 "$BENCH_OUT"',
'python3 benchmark/native/family_performance.py --size 1000 --output "$BENCH_OUT" --language both',
'python3 benchmark/native/repeat_family_native.py --output "$BENCH_OUT" --language both',
'python3 benchmark/native/report_family_performance.py --output "$BENCH_OUT"',
'python3 benchmark/native/export_latex.py --results "$BENCH_OUT"','```']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[20:36]))
