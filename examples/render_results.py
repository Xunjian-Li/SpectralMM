"""Render the saved example outputs as English Markdown and standalone LaTeX."""
import csv
from pathlib import Path
from run_python import CASES, ROOT

HERE = ROOT / 'examples'

def read(name):
    with (HERE / 'results' / name).open() as f:
        return list(csv.DictReader(f))

def number(value):
    return f'{float(value):.6g}' if value else '--'

def tex(s):
    return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

results = {lang: read(lang + '.csv') for lang in ('python', 'R', 'Julia')}
for lang, rows in results.items():
    assert len(rows) == 56, (lang, len(rows))

intro = '''These examples use the same saved data in all three languages: 100 observations and three feature columns. The default intercept gives four coefficients, with the intercept first. The data generator uses NumPy seed 20260930, independent normal features with standard deviation 0.5, and predictor 0.2 + 0.4 x1 - 0.3 x2 + 0.2 x3, except for the positive Gamma-inverse predictor described below.

All models use PCG, rank 3, at most 500 outer iterations, absolute gradient tolerance 1e-6 and relative tolerance 1e-8. Stalled-step acceptance is disabled. Initial coefficients are zero, except Gamma inverse, which starts at (2, 0, 0, 0). Rank 3 spans all but one of the four coefficient directions; these small examples illustrate usage and results, not large-scale performance.

Python and R call the shared C++ core. Julia uses the original Julia implementation. Negative binomial, Tweedie and grouped binomial use Julia's low-level API with an explicit intercept column; this API does not attach inference. Their inference is available in the Python/R results. All other examples use high-level APIs with automatic inference (the default cutoff is 50 parameters including the intercept).

On this saved data, Python and R meet the gradient criterion for all 14 models. Original Julia Probit reaches the 500-iteration limit (relative gradient approximately 1.51e-7), so its inference is unavailable at these tolerances. This exception is retained rather than classified as convergence.

The displayed coefficient tables are the Python results. The CSV files contain full-precision results for each language, including Wald chi-square statistics and their p-values. Standard errors, reference statistics and confidence intervals follow the model's inference convention. A t-reference coefficient p-value can differ from the asymptotic chi-square Wald p-value. Estimated coefficients need not equal the generating coefficients in a sample of 100 observations. Loss values should not be compared across different model families.'''
commands = '''# From the repository root; build the native library first (cpp/README.md).
python3 examples/run_python.py
Rscript examples/run_R.R
julia --project=. examples/run_julia.jl
python3 examples/render_results.py'''
md = ['# Small-data model examples', intro, '## Reproduce', '```sh\n' + commands + '\n```',
      'Install the standard R package and the Python/Julia dependencies described in the repository guides first. The Python runner creates the common CSV and the Python/solver outputs; the other runners read that CSV. The renderer runs only after all three languages finish.',
      'Complete iteration logs and fitted summaries: [Python](results/python.txt), [R](results/R.txt), [Julia](results/Julia.txt), [eight solvers](results/solvers.txt). Standalone Overleaf source: [Small_Model_Examples.tex](../docs/Small_Model_Examples.tex).',
      '## Cross-language results', '| Model | Language | Iterations | Relative gradient | Max coefficient difference from Python | Converged | Inference |', '|---|---|---:|---:|---:|---|---|']
latex = [r'\documentclass[11pt]{article}', r'\usepackage[margin=0.8in]{geometry}', r'\usepackage{booktabs,longtable,hyperref,listings}',
         r'\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible}',
         r'\title{SpectralMM: Small-Data Model Examples}', r'\author{}', r'\date{}', r'\begin{document}\maketitle', tex(intro),
         r'\section{Reproduce}', r'\begin{lstlisting}', commands, r'\end{lstlisting}',
         'Build the native library and install the dependencies using the repository guides. Run Python first to generate the shared data. This document is self-contained and can be uploaded directly to Overleaf.',
         r'\section{Cross-language results}',r'\scriptsize',r'\begin{longtable}{llrrrll}',
         r'\toprule Model & Language & Iter. & Rel. grad. & Max diff. & Conv. & Inference \\ \midrule\endhead']
for family, _, title, _ in CASES:
    ref = [r for r in results['python'] if r['family'] == family]
    for lang, rows in results.items():
        rr = [r for r in rows if r['family'] == family]
        diff = max(abs(float(a['estimate'])-float(b['estimate'])) for a,b in zip(ref,rr))
        r = rr[0]
        cells = [title,lang,r['iterations'],number(r['relgradnorm']),f'{diff:.2e}',r['converged'].lower(),r['inference_status']]
        md.append('| ' + ' | '.join(cells) + ' |')
        latex.append(' & '.join(tex(c) for c in cells) + r' \\')
latex += [r'\bottomrule\end{longtable}\normalsize']
for family, params, title, description in CASES:
    rr = [r for r in results['python'] if r['family'] == family]
    r = rr[0]
    description += ' Family identifier: ' + family + '. Fixed options: ' + (str(params) if params else 'defaults') + '.'
    status = f"Converged in {r['iterations']} iterations; loss {number(r['loss'])}; relative gradient {number(r['relgradnorm'])}; inference {r['inference_status']}."
    md += ['## ' + title, description, status,
           '| Term | Estimate | SE | Statistic | p-value | 95% CI |', '|---|---:|---:|---:|---:|---|']
    latex += [r'\section{' + tex(title) + '}',tex(description), '\n'+tex(status),
              r'\begin{center}\small\begin{tabular}{lrrrrl}\toprule',
              r'Term & Estimate & SE & Statistic & p-value & 95\% CI \\ \midrule']
    for r in rr:
        cells = [r['term']] + [number(r[k]) for k in ('estimate','std_error','statistic','p_value')]
        cells += [f"[{number(r['lower'])}, {number(r['upper'])}]"]
        md.append('| ' + ' | '.join(cells) + ' |')
        latex.append(' & '.join(tex(c) for c in cells) + r' \\')
    latex += [r'\bottomrule\end{tabular}\end{center}']
solvers = read('solvers.csv')
md += ['## Eight solvers on the same Logistic data',
       'All solvers start from zero on the same 100 x 3 input. Krylov relative/absolute tolerances are 1e-8/1e-12. These are numerical demonstrations, not timing comparisons.',
       '| Solver | Iterations | Loss | Relative gradient |', '|---|---:|---:|---:|']
latex += [r'\section{Eight solvers on the same Logistic data}',
          'All solvers use the same input and zero initial coefficients. Krylov relative/absolute tolerances are 1e-8/1e-12. These are numerical demonstrations, not timing comparisons.',
          r'\begin{center}\begin{tabular}{lrrr}\toprule Solver & Iterations & Loss & Rel. gradient \\ \midrule']
for r in solvers:
    cells = [r['solver'],r['iterations'],number(r['loss']),number(r['relgradnorm'])]
    md.append('| ' + ' | '.join(cells) + ' |')
    latex.append(' & '.join(tex(c) for c in cells) + r' \\')
latex += [r'\bottomrule\end{tabular}\end{center}', r'\appendix\section{Complete Python iteration logs and summaries}',
          'R and Julia logs are saved separately in examples/results. Row 0 is the initial point; subsequent rows report accepted iterates. The final relative gradient matches the last accepted iterate.',
          r'\begin{lstlisting}',(HERE/'results/python.txt').read_text(),r'\end{lstlisting}',r'\end{document}']
(HERE/'README.md').write_text('\n\n'.join(md).replace('|\n\n|', '|\n|')+'\n')
(ROOT/'docs/Small_Model_Examples.tex').write_text('\n'.join(latex)+'\n')
print('Rendered all 14 examples and the eight-solver comparison.')
