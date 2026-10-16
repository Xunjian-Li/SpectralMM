"""Export benchmark CSVs without silently mixing timing sessions."""
import argparse
import csv
import json
import math
import hashlib
from pathlib import Path

GROUPS={'glm':['gaussian','bernoulli','probit','poisson','gamma','negative_binomial'],
        'nonglm':['smooth_quantile','expectile','pseudo_huber','student_t']}
NAMES=dict(zip(GROUPS['glm']+GROUPS['nonglm'],['Gaussian-identity','Bernoulli-logit','Bernoulli-probit','Poisson-log','Gamma-log','Negative-binomial-log','Smooth quantile','Expectile','Pseudo-Huber','Student-$t$']))


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def number(value):
    try:return float(value)
    except (ValueError,TypeError):return math.nan


def sci(value):
    x=number(value)
    if not math.isfinite(x):return '--'
    if x==0:return '$0$'
    a,b=f'{x:.2e}'.split('e');return rf'${a}\times10^{{{int(b)}}}$'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--source',choices=('initial','repeat'),default='initial')
    args=parser.parse_args(); data=args.results.resolve()
    config=json.loads((data/'config.json').read_text())
    if args.source=='repeat':
        records=[]
        for stage in ('initial','repeat'):
            path=data/f'provenance_{stage}.json'
            if not path.exists():raise ValueError('Repeat export requires matched provenance from run_experiment.py; use --source initial for legacy runs')
            records.append(json.loads(path.read_text()))
        for record in records:
            if not record.get('result_sha256'):raise ValueError('Missing measured-result hashes')
            for filename, expected in record['result_sha256'].items():
                if hashlib.sha256((data/filename).read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'Result changed after recorded timing: {filename}')
        for key in ('settings','sources','binaries','input_sha256'):
            if records[0][key]!=records[1][key]:raise ValueError(f'Cannot mix initial/repeat timings: {key} differs')
    out=data/('latex' if args.source=='initial' else 'latex_repeat');out.mkdir(exist_ok=True)
    gn_tol=config['gtol']; rel_tol=config['relgtol']
    def good(r):return number(r['gradnorm'])<=gn_tol or number(r['relgrad'])<=rel_tol
    parts=[];total=0
    markdown=['# Python/R benchmark results','',f'Timing source: {args.source}. Times are seconds. Gradient convergence is assessed using the common gradient thresholds; warnings remain separate.','']
    for language,prefix in [('Python','python'),('R','R')]:
        initial=read(data/(prefix+'.csv'))
        rows=initial if args.source=='initial' else read(data/(prefix+'_repeat.csv'))+[r for r in initial if r['method']=='GLM']
        # Warnings from the initial run remain relevant to repeated native fits.
        warnings={(r['family'],r['storage'],r['method']):r.get('warnings','') for r in initial}
        langparts=[]
        for storage in ('dense','sparse'):
            for group,families in GROUPS.items():
                relevant=[r for r in rows if r['storage']==storage and r['family'] in families]
                if not relevant:continue
                dims={(c['n'],c['p_total']) for c in config['cases'] if c['storage']==storage and c['family'] in families}
                if len(dims)!=1:raise ValueError('Each table requires one data size')
                n,p=dims.pop();title='generalized linear models' if group=='glm' else 'non-GLM regression models'
                lines=[r'\begingroup',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{llrrrc}',
                       rf'\caption{{Computational comparison for {title} in {language} with a {storage} design matrix: $n={n}$, $p={p}$ (including the intercept). Times are seconds; {config["threads"]} threads were requested.}}',
                       rf'\label{{tab:{prefix}-{storage}-{group}-{args.source}}}\\',r'\toprule',
                       r'Model & Method & Time (s) & Iteration & Rel. Grad. & \shortstack{Gradient\\converged} \\',r'\midrule',r'\endfirsthead',r'\toprule',
                       r'Model & Method & Time (s) & Iteration & Rel. Grad. & \shortstack{Gradient\\converged} \\',r'\midrule',r'\endhead',
                       r'\midrule',r'\multicolumn{6}{r}{Continued on next page}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot']
                markdown += [f'## {language}: {storage}, {group}','', '| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |','|---|---|---:|---:|---:|---|']
                for index,family in enumerate(families):
                    selected=[r for r in relevant if r['family']==family]
                    if not selected:raise ValueError(f'Missing model: {family}')
                    expected=config.get('methods',[])
                    expected=[m for m in expected if group=='glm' or m!='GLM']
                    methods=[r['method'] for r in selected]
                    if len(set(methods))!=len(methods) or (expected and set(methods)!=set(expected)):raise ValueError(f'Incomplete/duplicate methods: {family}/{storage}')
                    selected.sort(key=lambda r:({'Spectral-MM':0,'Spectral-MM (PCG)':0,'Spectral-MM (MM)':1,'GLM':3}.get(r['method'],2), number(r['median_ms'])))
                    for j,r in enumerate(selected):
                        flag=bool(warnings.get((family,storage,r['method']))) or bool(r.get('error'))
                        label=r['method']
                        if label=='GLM':label=r'\texttt{statsmodels.GLM}' if language=='Python' else r'\texttt{glm.fit}'
                        elapsed=number(r['median_ms']);time=f'{elapsed/1000:.3f}' if math.isfinite(elapsed) else '--'
                        iteration=number(r.get('iterations'));iteration=str(int(iteration)) if math.isfinite(iteration) and iteration>=0 else '--'
                        status='Yes' if good(r) else 'No'
                        lines.append(f"{NAMES[family] if j==0 else ''} & {label} & {time} & {iteration} & {sci(r['relgrad'])} & {status}"+(r'$^{\ddagger}$' if flag else '')+r' \\')
                        markdown.append(f"| {family} | {r['method']} | {time} | {iteration} | {r['relgrad']} | {status}{'*' if flag else ''} |")
                        total+=1
                    if index<len(families)-1:lines.append(r'\midrule')
                samples=config['samples'];repeat=config.get('repeat_samples',5)
                timing=(f'All timings are medians of {samples} initial-run samples.' if args.source=='initial' else f'Native timings are medians of {repeat} interleaved rounds; GLM timings are medians of {samples} initial-run samples with matching source, binary, settings and input records.')
                strict=not config.get('accept_negligible',True) and not config.get('accept_stalled',True)
                # Imported legacy config may omit flags; do not assert strict stopping without evidence.
                lines += [r'\end{longtable}',r'\noindent\textit{Notes.} '+timing,
                  rf'Gradient convergence means $\|g\|_2\leq {sci(gn_tol).strip("$")}$ or $\|g\|_2/(1+\|g_0\|_2)\leq {sci(rel_tol).strip("$")}$.',
                  r'Relative gradients are evaluated using a common definition after fitting. GLM implementations use deviance-based stopping; runtimes do not necessarily correspond to identical final accuracy.',
                  rf'Python GLM uses absolute deviance tolerance {sci(config["glm_tolerance"])} and zero relative tolerance; R GLM uses relative deviance tolerance {sci(config["glm_tolerance"])}.',
                  r'$^{\ddagger}$A warning or error was recorded; see the CSV warnings/error fields. A small gradient alone does not establish a finite MLE under separation.']
                if strict:lines.append('Native negligible-step and stalled-step acceptance are disabled.')
                if storage=='sparse':lines.append(r'Sparse GLM inputs are converted to dense within the timed call.')
                if group=='nonglm':lines.append(r'Parameters: smooth quantile $(q,\epsilon)=(0.25,0.1)$; expectile $q=0.25$; pseudo-Huber $\delta=1$; Student-$t$ $(\nu,\sigma)=(4,1)$.')
                lines += [r'\par',r'\endgroup']
                text='\n'.join(lines)+'\n';parts.append(text);langparts.append(text)
                (out/f'{prefix}_{storage}_{group}.tex').write_text(text)
        (out/f'{prefix}_tables.tex').write_text('\n\\clearpage\n'.join(langparts))
    header=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{booktabs,longtable,amsmath}
\begin{document}
'''
    (out/'performance_tables.tex').write_text(header+'\n\\clearpage\n'.join(parts)+'\n\\end{document}\n')
    markdown+=['','* indicates a warning or error; inspect the raw CSV. Actual BLAS parallelism depends on the linked provider.']
    (out/'report.md').write_text('\n'.join(markdown)+'\n')
    print(f'Exported {len(parts)} tables, {total} rows: {out}')


if __name__=='__main__':main()
