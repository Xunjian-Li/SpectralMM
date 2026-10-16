# Benchmarks

Run all commands below from the **repository root**, not from `benchmark/`.
There are two separate experiment suites: Python/R comparisons using the C++
core, and the original pure-Julia comparisons. They have different entry points
and parameter settings.

## Python and R: current workflow

Use a Python environment containing NumPy, SciPy, pandas, statsmodels, patsy and
threadpoolctl. The local environment used for these experiments is
`/Users/caleblee/miniconda3/bin/python`; use that executable instead of `python`
if your shell's Python does not have these dependencies. Native builds also
require CMake, a C++17 compiler, R, and the R dependencies listed in the
[detailed instructions](native/README.md#requirements).

On this machine, reuse the retained input data:

```sh
python benchmark/native/run_experiment.py \
  --settings benchmark/native/settings.json \
  --inputs benchmark/data/p1000 \
  --output benchmark/runs/p1000-threads12-new
```

Choose a fresh output directory for every experiment. The runner automatically
builds the current C++ source, installs R into an isolated local library, runs
both interfaces sequentially, and exports tables. It records source, binary,
input and result hashes to prevent mixing incompatible timing sessions.

The binary inputs are **local and ignored by Git**. On a fresh clone, first
prepare Julia and then omit `--inputs` to generate them:

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
python benchmark/native/run_experiment.py \
  --settings benchmark/native/settings.json \
  --output benchmark/runs/generated-p1000
```

Default settings are in [native/settings.json](native/settings.json): 1000
predictors, 2000 observations, one additional intercept, 12 requested threads,
three initial timing samples and five native repeat rounds. Both PCG and MM are
included, along with CG/CGLS/CRLS/LSQR/LSMR and applicable official GLM comparators.
Cholesky is not included in this Python/R suite.

To customize a run without editing the shared defaults:

```sh
cp benchmark/native/settings.json benchmark/settings.local.json
# Edit benchmark/settings.local.json, then:
python benchmark/native/run_experiment.py \
  --settings benchmark/settings.local.json \
  --inputs benchmark/data/p1000 \
  --output benchmark/runs/custom-run
```

Change `threads` to a positive integer or `"max"`. If `size` changes, omit
`--inputs` or provide data generated for that exact size. See the
[parameter reference](native/README.md#change-experiment-parameters) for all
settings. This JSON does not control the original Julia scripts below.

## Saved results and LaTeX export

- [Latest imported initial measurements](native/results/initial-strict-12-2026-10-15/README.md)
- [Latest tables and report](native/results/initial-strict-12-2026-10-15/latex/report.md)
- [Standalone Overleaf source](native/results/initial-strict-12-2026-10-15/latex/performance_tables.tex)

Export those saved measurements without fitting or compiling native code:

```sh
python benchmark/native/export_latex.py \
  --results benchmark/native/results/initial-strict-12-2026-10-15 \
  --source initial
```

For a new complete runner experiment, `latex/` contains initial-only results and
`latex_repeat/` contains verified native repeat timings plus the matching initial
GLM timings. Historical initial and repeat results must not be mixed across
versions; the latest imported measurements have the provenance limitations
stated in their README.

## Original pure-Julia comparisons

These scripts use the local Julia package and a separate benchmark environment:

```sh
julia --project=benchmark benchmark/setup.jl
julia --threads=12 --project=benchmark benchmark/scaling.jl
# Run separately when the dense experiment has finished:
julia --threads=12 --project=benchmark benchmark/scaling_sparse.jl
```

Both scripts currently set `ps = [5000]`, `n = 2p`, and add an intercept,
so there are 10,000 observations and 5,001 fitted parameters. Edit `ps` and the
option constructors in the relevant script to change this suite's settings.
The sparse script uses approximately 20 nonzeros per row. Its Cholesky reference
and Cholesky comparator allocate dense workspaces, so this is not an entirely
matrix-free sparse experiment; increasing p substantially can exhaust memory.

Each script **overwrites** its CSV only after the entire run finishes:

- `benchmark/scaling_results.csv`
- `benchmark/scaling_sparse_results.csv`

Save a copy before rerunning if you need to retain the previous measurements.
These scripts do not use the Python/R runner's fresh-output protection.

For a notebook, restart the kernel before switching package environments, set
the working directory to the repository root, then run:

```julia
include("benchmark/setup.jl")
ENV["SPECTRALMM_BENCH_THREADS"] = "12" # Or "max".
include("benchmark/scaling.jl")
```

`include("benchmark/common.jl")` only activates the environment, configures BLAS
and defines helpers; it does not execute the benchmark or update a CSV.
The Julia scripts default to 12 BLAS threads. For all logical CPUs on macOS:

```sh
SPECTRALMM_BENCH_THREADS=max julia --threads="$(sysctl -n hw.logicalcpu)" \
  --project=benchmark benchmark/scaling.jl
```

Julia worker threads and BLAS threads are separate settings. Loading SpectralMM
does not change either. A requested thread count does not guarantee that every
operation uses all threads, or that more threads are faster.

## Directory layout

| Path within `benchmark/` | Purpose |
|---|---|
| `native/` | Python/R runners, shared settings, exporters and instructions |
| `native/results/` | Curated result archives, raw timing records and tables |
| `data/p1000/` | Local retained fixtures; ignored by Git |
| `build/` | Automatically created C++ build and isolated R installation; ignored |
| `runs/` | New experiment outputs; ignored |
| `scaling*.jl`, `common.jl` | Original Julia suite |
| `scaling*_results.csv`, `*_glm.tex`, `*_nonglm.tex` | Saved Julia results and table fragments |

The root-level `build/` is no longer used by this benchmark workflow.
Native solvers and official GLM implementations retain different stopping rules;
final gradients are evaluated using a common definition. See the
[interpretation notes](native/README.md#interpretation) before reporting speedups.
