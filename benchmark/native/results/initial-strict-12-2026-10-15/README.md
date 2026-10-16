# Latest initial-run results (2026-10-15)

These are the latest Python/R initial measurements copied from the user's completed temporary run: 152 rows per interface, three timing samples, n=2000, p=1000 predictors plus an intercept, and 12 requested threads. Native negligible/stalled-step acceptance was disabled. Official GLM stopping was unchanged.

Only initial-run CSVs are authoritative for this directory. The earlier five-round measurements are preserved separately in `../all-strict-12-2026-10-15/`; do not combine them with these GLM rows. For example, dense logistic MM changed from 31/1527 outer/inner iterations in the earlier repeat to 16/654 here.

The supplied `environment_python.json` was overwritten by the R-only launcher and records an older Python library hash. Therefore the exact binary provenance of the Python measurement cannot be certified retrospectively; the R metadata also lacks an exact library hash. Iteration counts alone are not proof of binary identity. These are measured results, not a certified reproduction of the current source. `provenance/` preserves the executed scripts as evidence, not as the recommended entry point.

Shared input binaries have been copied to the local ignored `benchmark/data/p1000/data/` directory. Future runs should use `benchmark/native/run_experiment.py`, which builds the current source and saves stage-specific provenance. The LaTeX files in `latex/` use only these initial-run measurements.
