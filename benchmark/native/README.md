# Native benchmark

The retained benchmark covers the article's six GLM and four residual models,
with shared Julia-generated data, dense/CSC designs, Python/R native Spectral-PCG
and five Krylov solvers, applicable GLM references, and original Julia Spectral-MM.

- [Final report](results/family-performance/report.md)
- [Python LaTeX tables](results/family-performance/latex/python_tables.tex)
- [R LaTeX tables](results/family-performance/latex/R_tables.tex)
- [English LaTeX usage and reproduction guide](../../docs/usage_and_benchmark.tex)

## Reproduce without overwriting the archive

Run from the repository root after building the core and installing dependencies
as described in `cpp/README.md`. Julia, Python and Rscript must be on PATH.

```sh
BENCH_OUT=build/family-benchmark
julia --compiled-modules=existing --project=. benchmark/native/family_performance.jl 1000 "$BENCH_OUT"
python3 benchmark/native/family_performance.py --size 1000 --output "$BENCH_OUT" --language both
python3 benchmark/native/repeat_family_native.py --output "$BENCH_OUT" --language both
python3 benchmark/native/report_family_performance.py --output "$BENCH_OUT"
python3 benchmark/native/export_latex.py --results "$BENCH_OUT"
```

The argument 1000 is the number of features excluding the intercept; n=2000 and
p_total=1001. To run n=10000, p_total=5001, change both size arguments to 5000.
GLM references may take a long time when they hit the 1000-iteration limit.
Use `--language python` or `--language R` for a single interface. Both are needed
for the combined report. Input data were removed from the retained archive;
regenerate them with the Julia script before a new timing run.

## Files

- `common.py`: shared imports, binary input loader and result writer.
- `family_performance.jl`: data/initialization generation and Julia baseline.
- `family_performance.py` / `.R`: first Python/R runs, including GLM references.
- `repeat_family_native.py` / `.R`: five-round interleaved native timing.
- `report_family_performance.py`: report from recorded results.
- `export_latex.py`: two four-table documents and Julia supplement.

The archive retains original and repeat summaries, per-sample timings, settings,
environment metadata and LaTeX results. Keeping the original GLM/Julia timing
sessions explicit prevents mixing them silently with the later native repeat.

## Intercept compatibility

Shared benchmark matrices already contain an intercept column. The benchmark
scripts explicitly use `fit_intercept=False` / `intercept=FALSE` so the new
wrapper defaults do not add a duplicate column. The archived results are retained
measurements, not a timing run of the updated implementation. Re-run the commands
above into a fresh output directory to measure the current code.

Native benchmark fits explicitly set `inference=False` / `inference=FALSE`.
They measure optimization without covariance/statistical reporting. Inference
cost should be measured separately if required.

## Original Julia versus the native Julia wrapper

Run `julia --project=. benchmark/native/compare_julia_native.jl` from the repository
root for warmed, matched-configuration comparisons. See the retained
[wrapper timing investigation](results/julia-wrapper-timing.md) for measured
loading overhead, the caching fix, and limitations.

The subsequent [wrapper optimizations](results/wrapper-optimizations.md) record
Python/R/Julia validation, local before/after timings and Julia allocation changes.

The current local native library enables Apple Accelerate BLAS. See the
[BLAS comparison](results/blas-comparison.md) for numerical checks and timings
against the preceding Eigen-only library.
