# Archived diagnostic scripts

These scripts investigated Julia/C++ wrapper overhead and small-data performance.
They are retained for historical reference, are not part of either current
benchmark experiment, and have not been revalidated against the current API.

- `compare_julia_native.jl`: wrapper overhead with and without inference.
- `small_current_backends.jl`: small-data backend timings and numerical checks.

Their original reports remain in [python_r/results](../python_r/results/).
For current experiments, use the two entry points in the
[benchmark guide](../README.md). Do not run archived scripts against saved
result paths: some have historical default output destinations.
