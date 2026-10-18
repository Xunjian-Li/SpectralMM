# Benchmarks

There are two experiments. Run all commands from the repository root.
Both include **Spectral-MM (PCG)** and **Spectral-MM (MM)**, dense and sparse
inputs, six GLM families and four non-GLM losses.

| Experiment | Entry point | Parameters | Default size |
|---|---|---|---|
| Pure Julia | `run_julia.jl` | [julia/julia_settings.jl](julia/julia_settings.jl) | p = 5000 for dense and sparse |
| Python and R, C++ backend | `run_python_r.py` | [python_r/settings.json](python_r/settings.json) | p = 1000 for dense and sparse |

Here p excludes the intercept; n = 2p, and the fitted model has p + 1
parameters. These retain the existing experiments' solver settings and stopping
rules; their timings are not a controlled backend-only comparison.

## 1. Julia

Install the benchmark dependencies once, then run:

```sh
julia --project=benchmark/julia benchmark/julia/setup.jl
julia --threads=12 --project=benchmark/julia benchmark/run_julia.jl \
  --output benchmark/runs/julia-new
```

Use `--storage dense` or `--storage sparse` to run only one design type.
Edit `julia/julia_settings.jl` to change `dense_sizes`, `sparse_sizes`, `samples`,
BLAS `threads`, or solver tolerances. A custom settings file can be supplied
with `--settings path/to/settings.jl`.

Outputs: `dense.csv`, `sparse.csv` and `settings.txt` in the chosen directory.
Comparators are Cholesky, CGLS, CRLS, LSQR, LSMR and GLM.jl where applicable.
The Cholesky comparator and reference use dense workspaces even for sparse
inputs; take memory requirements into account before increasing p.

## 2. Python and R

Requirements: Python with NumPy, SciPy, pandas, statsmodels, patsy and
threadpoolctl; R with Rcpp, RcppEigen, Matrix, jsonlite and MASS; CMake and a
C++17 compiler. Julia generates new inputs; prepare it once:

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python benchmark/run_python_r.py --output benchmark/runs/python-r-new
```

The runner builds the current C++ source automatically, installs R in an isolated
local library, generates inputs, runs Python and R sequentially, and exports
LaTeX tables. Both languages use identical input files and initial coefficients.
Comparators are CG, CGLS, CRLS, LSQR, LSMR and applicable statsmodels/R GLM fits.

To reuse the retained local p = 1000 data instead of generating it:

```sh
python benchmark/run_python_r.py \
  --inputs benchmark/data/p1000 --output benchmark/runs/python-r-reused
```

Edit `python_r/settings.json`, or copy it and pass `--settings path/to/settings.json`.
Change `size` for p, `threads` for BLAS threads, `samples`/`repeat_samples` for
timing repetitions, and `gtol`/`relgtol` for native gradient tolerances.
When changing `size`, omit `--inputs` or supply matching data.
See the [detailed parameter and stopping-rule notes](python_r/README.md).

Outputs include `python.csv`, `R.csv`, raw samples, configuration and provenance.
`latex/` contains initial measurements; `latex_repeat/` contains verified native
repeat measurements with matching GLM baselines. GLM baselines keep their own
deviance-based stopping rules with the configured tolerance; they do not use
the native solvers' gradient stopping rule.

## Data, results and threads

Always choose a **new output directory** for a new experiment.
Both experiments request 12 BLAS threads by default. Set the settings file's
`threads` to `"max"` for all logical CPUs. For Julia worker threads as well,
use `julia --threads=auto`; worker and BLAS thread counts are separate.
`SPECTRALMM_BENCH_THREADS` overrides the Julia BLAS setting.

| Location | Contents |
|---|---|
| `data/p1000/` | Retained local binary inputs; ignored by Git, reproducible through the runner |
| `build/` | Automatic native builds and isolated R library; ignored by Git |
| `runs/` | New experiment outputs; ignored by Git |
| `python_r/results/` | Archived Python/R measurements, provenance and tables |
| `julia/results/` | Saved Julia results |

[Latest Python/R tables](python_r/results/initial-strict-12-2026-10-15/latex/performance_tables.tex)
and [their provenance notes](python_r/results/initial-strict-12-2026-10-15/README.md)
remain unchanged. Do not mix archived timings from different runs or code versions.

Implementation helpers are grouped under `julia/` and `python_r/`.
Use the two entry points above for experiments. Older wrapper-overhead diagnostics
are retained separately in [archive/](archive/README.md); they are not part of
these experiments. Historical reports retain their original commands for provenance.
