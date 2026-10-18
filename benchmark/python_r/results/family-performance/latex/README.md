> **Historical record.** Commands and paths below describe the original run and may be obsolete. For current experiments, follow the [benchmark guide](../../../../README.md). Measurements and provenance files are retained unchanged.

# Retained LaTeX results

- `python_tables.tex`: four Python GLM/non-GLM dense/sparse tables.
- `R_tables.tex`: four R GLM/non-GLM dense/sparse tables.
- `julia_baseline_tables.tex`: four original Julia supplementary tables.
- `performance_tables.tex`: standalone wrapper for the eight Python/R tables.
- `performance_tables.pdf`: compiled preview of those eight tables.

The source fragments require booktabs. Compile the standalone wrapper from this
directory with `pdflatex performance_tables.tex`. Table data and footnotes retain
actual dimensions, convergence flags, gradient failures and timing-session differences.
Regenerate from the repository root with `python3 benchmark/native/export_latex.py`.
