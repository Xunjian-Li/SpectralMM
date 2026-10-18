# Python and R benchmarks

Run commands from the repository root. These experiments compare
Spectral-MM (PCG), Spectral-MM (MM), CG, CGLS, CRLS, LSQR and LSMR, plus
statsmodels.GLM / R glm.fit for the six applicable GLM families. Ten models
are tested with both dense and sparse designs. Cholesky is not included.

## Saved results

- [Latest initial-run tables and report](results/initial-strict-12-2026-10-15/latex/report.md)
- [Standalone Overleaf document](results/initial-strict-12-2026-10-15/latex/performance_tables.tex)
- [Latest result provenance and limitations](results/initial-strict-12-2026-10-15/README.md)
- `results/all-strict-12-2026-10-15/`: earlier strict-stopping measurements and tables, before the latest MM rerun. Do not combine with the latest initial results.
- `results/pcg-mm-2026-10-15/`: earlier two-method run with step stopping enabled.
- `results/family-performance/`: original historical archive.

The latest imported initial results have 152 rows per language (304 combined).
They contain three timing samples per method. The temporary directory's five-round
CSV files were older and are intentionally not copied into the latest directory.
No historical numbers were remeasured or silently replaced during consolidation.

## Requirements

Use a Python environment with NumPy, SciPy, pandas, statsmodels, patsy and
threadpoolctl. R needs Rcpp, RcppEigen, Matrix, jsonlite and MASS. CMake and a C++17
compiler are needed for the local builds. Eigen is detected at common system
locations or fetched by CMake; `EIGEN3_INCLUDE_DIR` can specify local headers.
Julia is needed only when generating new input data. Prepare its environment once:

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

Use the same Python executable for setup and experiments. On the current machine,
`/Users/caleblee/miniconda3/bin/python` has the required Python dependencies.
Changing code does not update an already loaded notebook; use fresh processes.

## Reproduce the current 1000-predictor experiment

The retained input binaries are local, ignored build artifacts at
`benchmark/data/p1000/data/`. They were copied from the completed experiment.
They are not shipped in Git. Run:

```sh
python benchmark/run_python_r.py \
  --settings benchmark/python_r/settings.json \
  --inputs benchmark/data/p1000 \
  --output benchmark/runs/p1000-threads12-new
```

The output must be new. The runner performs these steps in order:

1. Synchronize the shared C++ source into the three language package trees.
2. Incrementally build a release C++ library and install R into an isolated local library under `benchmark/build/`. R installation is skipped when its source fingerprint is unchanged.
3. Load Python from this checkout and R from that isolated library, ignoring old temporary library paths.
4. Use the existing binary fixtures, including identical initial coefficients for every method.
5. Run Python and R sequentially: three initial timing samples for all methods, then five interleaved rounds for native methods.
6. Save source/binary/input hashes, settings and environment records; export initial-only tables and a separate verified repeat-based set.

No changes are made to the user's global Python/R installation. The runner may
synchronize package source copies and writes build products under ignored `benchmark/build/`.
The Julia fixture-generation baseline is not included in the Python/R tables.

On another machine, or for a different size, omit `--inputs` to regenerate data:

```sh
python benchmark/run_python_r.py \
  --settings benchmark/python_r/settings.json \
  --output benchmark/runs/regenerated-p1000
```

## Change experiment parameters

Copy and edit the JSON settings; do not edit separate Python and R constants:

```sh
cp benchmark/python_r/settings.json benchmark/settings.local.json
```

| Setting | Meaning | Default |
|---|---|---:|
| `size` | Predictors excluding intercept; n=2*size, total parameters=size+1 | 1000 |
| `threads` | Requested BLAS and Julia threads; integer or `"max"` | 12 |
| `samples` | Initial measured samples, including GLM | 3 |
| `repeat_samples` | Interleaved native timing rounds | 5 |
| `rank`, `krylovdim` | Spectral rank and initial Lanczos dimension | 5, 12 |
| `maxiter`, `inner_maxiter` | Outer and inner iteration limits | 1000, 200 |
| `gtol`, `relgtol` | Absolute and relative gradient thresholds | 1e-6, 1e-8 |
| `eta_max` | Inner forcing tolerance | 0.6 |
| `floor` | Spectral regularization rho | 1e-6 |
| `correction_tol`, `resid_tol` | Spectral update tolerances | 0.05, 0.5 |
| `krylov_rtol`, `krylov_atol` | Other Krylov methods' inner tolerances | 1e-4, 1e-8 |
| `glm_tolerance` | Deviance tolerance passed to official GLM implementations | 1e-12 |
| `accept_negligible`, `accept_stalled` | Native step-based acceptance | false, false |
| `min_sample_seconds`, `max_batch` | Timing batch calibration controls | 0.05, 1000 |

For example, change `threads` to 1 and keep all other fields unchanged to compare
thread counts. Use a different `--output` for each run. If `size` changes, omit
`--inputs` or supply data generated for that exact size. Model families, fixed
family parameters, and seeds are defined in `family_performance.jl` and the
Python/R family parameter mappings; changing them requires updating all mappings
and generating new data. They are not configurable through this JSON.

The native ridge penalty remains zero, intercept columns are already present,
and inference is disabled. MM inner Nesterov is now enabled by the C++ core;
`nesterov=True` controls outer acceleration. Threads are requested limits, not a
guarantee that each BLAS provider actually uses that many workers.

## Run in stages and export

```sh
python benchmark/run_python_r.py --output benchmark/runs/new-run \
  --inputs benchmark/data/p1000 --stage initial
python benchmark/run_python_r.py --output benchmark/runs/new-run \
  --stage repeat
```

Use the same settings file for both stages. The repeat stage rejects changed
sources, inputs, settings or binaries rather than mixing measurements.

Export without computing any fits or compiling native code:

```sh
python benchmark/python_r/export_latex.py \
  --results benchmark/python_r/results/initial-strict-12-2026-10-15 \
  --source initial
```

Output: `latex/performance_tables.tex` (standalone, eight tables), language-level
and individual fragments, and `latex/report.md`. Upload the standalone file to
Overleaf; no additional inputs are required. For inclusion in a paper, use the
fragments with `booktabs`, `longtable` and `amsmath`.

`--source repeat` uses native repeat timings and initial GLM timings only when
both stages have matching provenance created by the runner. It writes separately
to `latex_repeat/`. Older archives without this evidence cannot use that mode.

## Interpretation

Each GLM implementation retains its own deviance-based stopping rule. Python uses
absolute deviance difference with zero relative tolerance; R uses relative
deviance change. These thresholds are not gradient tolerances. Final gradients
are scored consistently against the summed model loss, without ridge:
`norm(g) / (1 + norm(g_initial))`. The common criterion is absolute gradient <=
`gtol` OR relative gradient <= `relgtol`.

Tables label this column **Gradient converged**, separate from solver-declared
convergence. Warnings (including separation) are marked separately and retained
in raw CSVs. A small gradient under separation does not prove a finite MLE.
Consequently these are software comparisons under specified stopping rules, not
identical-final-accuracy algorithm timings. In particular, do not interpret a
speed ratio as a same-accuracy advantage when a comparator did not meet the
common gradient threshold or has a model-validity warning.

## Archived investigations

Older Julia/C++ wrapper and small-data diagnostics are kept in
[../archive/](../archive/README.md). Their reports remain under `results/`.
They use separate settings and are not part of this experiment.
