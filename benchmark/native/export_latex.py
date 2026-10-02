"""Export article-style booktabs tables from recorded benchmark data."""
import csv, math, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=ROOT/'benchmark/native/results/family-performance')
DATA=parser.parse_args().results.resolve()
config=json.loads((DATA/'config.json').read_text())
n=config['cases'][0]['n']; p_total=config['cases'][0]['p_total']
latex_n=f'{n:,}'.replace(',', '{,}'); latex_p=f'{p_total:,}'.replace(',', '{,}')
OUT = DATA / 'latex'
OUT.mkdir(exist_ok=True)
GLM = ['gaussian','bernoulli','probit','poisson','gamma','negative_binomial']
NONGLM = ['smooth_quantile','expectile','pseudo_huber','student_t']
NAMES = dict(zip(GLM+NONGLM, ['Gaussian-identity','Bernoulli-logit','Bernoulli-probit','Poisson-log',
 'Gamma-log','Negative-binomial-log','Smooth quantile','Expectile','Pseudo-Huber','Student-$t$']))

def read(file): return list(csv.DictReader((DATA/file).open()))
def yes(value): return value.lower() == 'true'
def gradpass(r): return float(r['gradnorm']) <= 1e-6 or float(r['relgrad']) <= 1e-8

def scientific(value):
    x = float(value)
    if not math.isfinite(x): raise ValueError('nonfinite table metric')
    if x == 0: return '$0$'
    m,e = f'{x:.2e}'.split('e')
    return rf'${m}\times10^{{{int(e)}}}$'

def make_table(language, storage, group, rows, baseline=False):
    families = GLM if group=='glm' else NONGLM
    key = language.lower()
    title = 'generalized linear models' if group=='glm' else 'non-GLM models'
    impl = 'the original Julia implementation' if baseline else language
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}', r'\caption{',
      f'Computational comparison for {title} using {impl} with a {storage} design matrix,',
      f'$n={latex_n}$ and $p={latex_p}$ (including the intercept).']
    if storage=='sparse':
        lines += [rf'The non-intercept design has approximately 20 nonzeros per row (about ${2000/(p_total-1):.2g}\%$ density).']
    lines += [r'Runtime is reported in seconds. ``Iteration\textquotedblright{} denotes the number of outer iterations,',
      r'``Rel.\ grad.\textquotedblright{} denotes the relative gradient norm at termination, and',
      r'``Converged\textquotedblright{} reports whether the solver declared convergence.}',
      rf'\label{{tab:{key}-{group}-comparison-{storage}}}', r'\begin{tabular}{llrrrc}', r'\toprule',
      r'Model & Method & Time (s) & Iteration & Rel.\ grad. & Converged \\', r'\midrule']
    for index,family in enumerate(families):
        selected = [r for r in rows if r['family']==family and r['storage']==storage]
        assert selected
        selected.sort(key=lambda r: (0 if r['method']=='Spectral-MM' else 2 if r['method']=='GLM' else 1, float(r['median_ms'])))
        model = NAMES[family] + (r'$^{\ddagger}$' if family in ('bernoulli','probit') else '')
        lines += ['', model]
        for r in selected:
            method = r['method']
            if method=='GLM': method = r'\texttt{statsmodels.GLM}' if language=='Python' else r'\texttt{glm.fit}'
            status = 'Yes' if yes(r['converged']) else 'No'
            if yes(r['converged']) and not gradpass(r): status += r'$^{\dagger}$'
            lines.append(f"& {method} & {float(r['median_ms'])/1000:.3f} & {int(float(r['iterations']))} & {scientific(r['relgrad'])} & {status} " + r'\\')
        if index != len(families)-1: lines.append(r'\midrule')
    lines += ['',r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{\linewidth}',r'\footnotesize']
    lines += [r'$^{\dagger}$The solver declared convergence, but neither $\|g\|\leq 10^{-6}$ nor',
              r'$\|g\|/(1+\|g_0\|)\leq 10^{-8}$ was satisfied.']
    if group=='glm':
        lines += [r'$^{\ddagger}$These data exhibit separation/perfect prediction; a small gradient does not establish a finite maximum-likelihood estimate.']
    if baseline:
        lines += ['Timings are medians of three measured runs after warm-up.']
    else:
        lines += ['Spectral-MM and the Krylov methods use the shared C++ core.', 'Native timings are medians of five interleaved rounds after warm-up.']
        if group=='glm':
            lines += ['GLM timings are medians of three runs from an earlier timing session;',
                      'cross-session timing variation was observed. Sparse GLM inputs were converted to dense, with conversion included in timing.']
    if group=='nonglm':
        lines += [r'Parameters: smooth quantile $(\tau,\epsilon)=(0.25,0.1)$; expectile $\tau=0.25$;',
                  r'pseudo-Huber $\delta=1$; Student-$t$ $(\nu,\sigma)=(4,1)$.']
    else:
        lines += [r'The negative-binomial shape parameter is fixed at $\theta=4$.']
    lines += [r'All native/Julia fits use $\rho=10^{-6}$ and no ridge penalty.',r'\end{minipage}',r'\end{table}','']
    return '\n'.join(lines)

row_count=0
for language,file in [('Python','python'),('R','R')]:
    rows=read(file+'_repeat.csv') + [r for r in read(file+'.csv') if r['method']=='GLM']
    assert len(rows)==132
    parts=[]
    for group in ('glm','nonglm'):
        for storage in ('dense','sparse'):
            table=make_table(language,storage,group,rows)
            parts.append(table); row_count+=sum(line.startswith('& ') for line in table.splitlines())
    aggregate='% Requires \\usepackage{booktabs}. Times and stopping flags are from recorded benchmark files.\n\n'+'\n\\clearpage\n\n'.join(parts)
    (OUT/f'{file}_tables.tex').write_text(aggregate)
assert row_count==264
# Original Julia timings are a separate supplement rather than a different method
# silently inserted into the Python/R tables.
parts=[make_table('Julia',storage,group,read('Julia.csv'),baseline=True)
       for group in ('glm','nonglm') for storage in ('dense','sparse')]
(OUT/'julia_baseline_tables.tex').write_text('% Requires \\usepackage{booktabs}.\n\n'+'\n\\clearpage\n\n'.join(parts))
(OUT/'performance_tables.tex').write_text(r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{booktabs}
\begin{document}
\input{python_tables.tex}
\clearpage
\input{R_tables.tex}
\end{document}
''')
print(f'Wrote eight Python/R tables ({row_count} rows) and four Julia baseline tables to {OUT}')
