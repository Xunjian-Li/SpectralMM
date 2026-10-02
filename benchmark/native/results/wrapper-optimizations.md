# Wrapper initialization and allocation optimizations

These changes reduce wrapper overhead without changing the native optimizer,
convergence tolerances, inference defaults or returned numerical results.

- Python caches library objects and C function signatures by absolute path,
  precomputes its default library path, and avoids scanning a complete dense
  column when the first two values already prove it is nonconstant.
- Julia extends its existing library-handle cache to function pointers and
  immutable default options. Compatible Float64 input matrices and vectors are
  borrowed during the native call, and logged trace rows are traversed through a
  view instead of a copied slice. CSC values are borrowed when compatible;
  zero-based index arrays are still required by the C ABI.
- R avoids assigning double storage mode to matrices already stored as doubles.
  Its existing initialization/load step remains outside the fitting loop. An
  experimental constant-column shortcut was withdrawn because it increased
  overhead on small R inputs.

The C++ core already maps supplied dense/CSC values and uses an implicit
intercept; no additional matrix-copy optimization was needed at that boundary.
Per-fit mutable buffers are not shared across fits. Arrays remain owned by the
caller and must not be modified concurrently with a native call.

## Validation

Temporary before/after scripts compared 224 Python fits and 224 R fits across
all 14 model families, eight solvers, and dense/CSC designs. Coefficients,
fit diagnostics, inference and captured traces matched, and inputs remained
unchanged. Julia passed 32 corresponding checks across eight solvers with
dense, CSC, Float32 and matrix-view inputs. Constant-column warning behavior,
integer-input conversion, Python cache reuse, the default library path and
recovery from an invalid library override were also checked. Development
scripts were kept outside the repository.

## Local timing and allocation results

Measurements compare each wrapper against its version immediately before this
change. In particular, the Julia baseline already includes the earlier library
handle caching fix. Timings include fitting and wrapper/result construction;
inference, printing and trace capture are disabled for timing. The workload is
Logistic with normal features and random binary responses, spectral PCG and
rank min(5, number of features), using the native default Lanczos dimension.
These configurations differ from the previous original-Julia comparison; the
numbers below must not be used to update its 1.43 ratio directly.

Each language was measured in a separate, serial run, with five warm-up calls
and nine randomly interleaved before/after rounds. Python and Julia used 20
fits per batch; R used 100. Thread-count environment variables were set to one,
and Julia also explicitly set its BLAS thread count to one. The same input and
options were used for each before/after pair within a language. Different
languages used their own RNGs, so these are not cross-language rankings.

| Wrapper | Input | Before (ms/fit) | After (ms/fit) |
|---|---|---:|---:|
| Python | 100 x 3 | 0.78179 | 0.23501 |
| Python | 2000 x 30 | 11.68699 | 10.48625 |
| R | 100 x 3 | 0.36000 | 0.28000 |
| R | 2000 x 30 | 11.67000 | 11.20000 |
| Julia | 100 x 3 | 0.11073 | 0.10576 |
| Julia | 2000 x 30 | 9.82223 | 10.31118 |

Python showed a clear benefit in this run. Julia did not show a consistent
elapsed-time improvement: the larger case was about 5% slower in the final
serial run, while a preliminary run had been slightly faster. The demonstrated
Julia benefit is reduced managed allocation, not a promised timing speedup:

| Julia input | Before allocated bytes/fit | After allocated bytes/fit |
|---|---:|---:|
| 100 x 3 | 6,288 | 4,672 |
| 2000 x 30 | 510,976 | 4,880 |

These are Julia heap allocations measured with `@allocated`, not native Eigen
allocations or peak process memory. C++ computation still dominates larger
fits, and avoiding one wrapper copy need not measurably improve wall time.
Restart Python/Julia after replacing a loaded native library at the same path.
