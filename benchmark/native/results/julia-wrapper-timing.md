# Julia and native-wrapper timing investigation

A local warmed Logistic comparison on macOS arm64 with Julia 1.12.6, one BLAS thread, 2,000 observations and 30 features (31 coefficients including the intercept) reproduced an approximately 2.6-fold native-wrapper slowdown. Rank was 5 and both starting Lanczos dimensions were 12. Both implementations took 10 outer iterations. Final relative gradients were 5.231e-9 (Julia) and 6.079e-9 (native).

The native wrapper previously opened and closed the shared library on every fit. It now retains a validated handle for each absolute library path. A before/after comparison in the same Julia process gave the following medians (milliseconds per fit):

| Inference | Original Julia before | Native before | Original Julia after | Native after | Native/Julia before | Native/Julia after |
|---|---:|---:|---:|---:|---:|---:|
| Disabled | 1.41924 | 3.73329 | 1.42936 | 2.04183 | 2.630 | 1.428 |
| Enabled | 1.62099 | 4.00971 | 1.59944 | 2.22442 | 2.474 | 1.391 |

Each phase used four warm-up calls per configuration and seven interleaved rounds, three fits per timed batch, with explicit garbage collection before each batch. The before phase ran first because the retained native handle also prevents the old wrapper from unloading the same library. Consequently phase order is not randomized. The original-Julia control stayed close between phases. These measurements are local diagnostics, not a general speed guarantee or a measurement of the user's unspecified workload.

The library cache reduces the native fit time by about 45% in this workload, while a remaining approximately 1.4-fold gap has not been isolated to a particular kernel. Native and original Julia have different numerical implementations and RNGs. Matrix copies, wrapper result construction and native linear algebra remain included in the end-to-end times. No claim is made that the remaining gap comes entirely from the C++ computation.

Reproduce the comparison with the current wrapper from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 julia --project=. benchmark/native/compare_julia_native.jl
```

The script covers 100 x 3, 2000 x 30 and 2000 x 100 inputs, with inference off/on. It explicitly aligns initial coefficients, rank, starting Lanczos dimension, solver and major tolerances. Defaults alone are not equivalent: original high-level Julia uses min(p, max(12, rank + 1)) for the starting Lanczos dimension, while the native core defaults to a larger dimension. The script disables printing, trace capture and stalled-step acceptance. Inference=true forces inference even above the automatic 50-parameter cutoff.

The library stays loaded until process exit. Restart Julia after replacing a library at the same path.
