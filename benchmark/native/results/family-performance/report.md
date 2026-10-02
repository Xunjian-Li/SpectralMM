# Performance comparison of ten models from the article (2026-09-29)

This run covers ten article benchmark models with n=2000, p=1001 (including the intercept), rather than the final article scale of n=10000, p=5001. Gaussian-log, Gamma-inverse, Tweedie and Binomial were not timed in this run.
Data generation and random seeds follow benchmark/scaling.jl and scaling_sparse.jl. Julia, Python and R share identical binary inputs and initial coefficients for each model.

## Timing and accuracy

- rho=1e-6, ridge=0, rank=5, krylovdim=12, at most 1000 outer and 200 inner iterations, eta_max=0.6.
- Gradient criterion: absolute gradient norm <=1e-6 or relative gradient <=1e-8. Default step-based stopping remains enabled; solver-declared convergence does not imply gradient convergence.
- Initial run: warm up each method and report the median of three timings. Native repeat: five rounds per scenario, shuffling the six methods each round; short calls are timed in batches.
- Main tables use native repeat results; GLM and Julia use initial-run results. Raw records from both runs are retained.
- Timing includes wrapper conversion and model construction, but excludes loading, compilation, data generation and scoring. Numerical libraries are configured for one thread.
- Python GLM uses statsmodels.GLM; R uses glm.fit. Negative-binomial theta is fixed at 4, with Python alpha=0.25.
- GLM tolerance is 1e-12, but all methods are assessed using the same final gradient criterion. The four non-GLM models have no GLM comparator.
- Sparse inputs are converted to dense for GLM, with conversion included in timing. This is therefore not a comparison of sparse GLM implementations.
- `*` denotes failure of an accuracy/validity check and excludes a result from equal-accuracy speed rankings. All times are in milliseconds.

## Spectral-MM versus other native solvers

Each ratio is the runtime of the fastest gradient-converged CG/CGLS/CRLS/LSQR/LSMR method divided by the Spectral-MM runtime. Values above 1 favor Spectral-MM. No ratio is reported when Spectral-MM fails the gradient criterion.

| Model | Python dense | Python sparse | R dense | R sparse |
|---|---:|---:|---:|---:|
| Gaussian | 0.65x (CRLS) | Gradient criterion not met | 0.66x (CRLS) | Gradient criterion not met |
| Logistic | 2.94x (CRLS) | 1.94x (CRLS) | 2.91x (CRLS) | 1.94x (CRLS) |
| Probit | 4.07x (CRLS) | 1.72x (CG) | 4.08x (LSMR) | 1.71x (CG) |
| Poisson | Gradient criterion not met | 0.78x (CRLS) | Gradient criterion not met | 0.77x (CRLS) |
| Gamma | Gradient criterion not met | Gradient criterion not met | Gradient criterion not met | Gradient criterion not met |
| Negative-binomial | Gradient criterion not met | Gradient criterion not met | Gradient criterion not met | Gradient criterion not met |
| SmoothQuantile | Gradient criterion not met | 3.84x (CRLS) | Gradient criterion not met | 3.97x (CRLS) |
| Expectile | Gradient criterion not met | 0.97x (CRLS) | Gradient criterion not met | 0.94x (CRLS) |
| PseudoHuber | 2.59x (CRLS) | 1.73x (CRLS) | 2.55x (CRLS) | 1.75x (CRLS) |
| Student-t | Gradient criterion not met | 2.58x (CRLS) | Gradient criterion not met | 2.61x (CRLS) |

## Python: complete native repeat timings

| Model | Matrix | Spectral-MM | CG | CGLS | CRLS | LSQR | LSMR |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian | dense | 47.57 | 31.85 | 32.24 | 30.93 | 32.63 | 31.38 |
| Gaussian | sparse | 4.74* | 2.16 | 2.17 | 2.21 | 2.23 | 2.25 |
| Logistic | dense | 208.17 | 623.28 | 624.75 | 612.36 | 627.61 | 612.76 |
| Logistic | sparse | 52.72 | 103.20 | 103.18 | 102.50 | 104.49 | 104.61 |
| Probit | dense | 151.83 | 630.71 | 632.14 | 617.69 | 631.80 | 621.12 |
| Probit | sparse | 51.91 | 89.27 | 89.63 | 95.30 | 91.55 | 98.18 |
| Poisson | dense | 107.96* | 146.56 | 149.51 | 135.45 | 149.28 | 137.83 |
| Poisson | sparse | 7.43 | 5.93 | 6.07 | 5.78 | 6.18 | 6.01 |
| Gamma | dense | 84.80* | 699.34 | 708.16 | 701.94 | 710.85 | 713.97 |
| Gamma | sparse | 16.55* | 39.02 | 41.06 | 36.27* | 41.45 | 38.54* |
| Negative-binomial | dense | 78.64* | 488.15 | 496.02 | 439.41 | 495.54 | 447.34 |
| Negative-binomial | sparse | 9.00* | 18.15 | 18.87 | 16.94* | 19.33 | 17.87* |
| SmoothQuantile | dense | 236.37* | 1713.08 | 1171.39 | 1107.20 | 1173.29 | 1242.58 |
| SmoothQuantile | sparse | 20.67 | 89.49 | 89.85 | 79.45 | 91.83 | 81.57 |
| Expectile | dense | 61.54* | 88.29 | 88.97 | 81.93 | 88.93 | 83.49 |
| Expectile | sparse | 6.20 | 5.99 | 6.05 | 5.99 | 6.21 | 6.08 |
| PseudoHuber | dense | 67.63 | 189.55 | 189.89 | 175.08 | 191.36 | 177.27 |
| PseudoHuber | sparse | 7.73 | 13.89 | 14.31 | 13.42 | 14.68 | 14.02 |
| Student-t | dense | 75.81* | 369.70 | 374.38 | 353.21 | 374.12 | 359.54 |
| Student-t | sparse | 9.80 | 26.07 | 26.87 | 25.30 | 27.49 | 26.65 |

## R: complete native repeat timings

| Model | Matrix | Spectral-MM | CG | CGLS | CRLS | LSQR | LSMR |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian | dense | 50.00 | 49.50 | 34.00 | 33.00 | 34.50 | 33.50 |
| Gaussian | sparse | 4.91* | 2.16 | 2.19 | 2.15 | 2.24 | 2.27 |
| Logistic | dense | 217.00 | 645.00 | 643.00 | 631.00 | 644.00 | 631.00 |
| Logistic | sparse | 53.00 | 104.00 | 104.00 | 103.00 | 106.00 | 106.00 |
| Probit | dense | 155.00 | 643.00 | 643.00 | 633.00 | 646.00 | 632.00 |
| Probit | sparse | 52.00 | 89.00 | 90.00 | 95.00 | 93.00 | 99.00 |
| Poisson | dense | 109.00* | 149.00 | 152.00 | 138.00 | 152.00 | 140.00 |
| Poisson | sparse | 7.38 | 5.89 | 6.00 | 5.67 | 6.11 | 5.89 |
| Gamma | dense | 86.00* | 707.00 | 719.00 | 709.00 | 722.00 | 723.00 |
| Gamma | sparse | 16.33* | 39.00 | 41.00 | 36.50* | 41.50 | 38.50* |
| Negative-binomial | dense | 80.00* | 493.00 | 498.00 | 446.00 | 502.00 | 450.00 |
| Negative-binomial | sparse | 8.83* | 18.00 | 18.67 | 16.50* | 19.00 | 17.67* |
| SmoothQuantile | dense | 238.00* | 1727.00 | 1179.00 | 1126.00 | 1183.00 | 1260.00 |
| SmoothQuantile | sparse | 19.67 | 88.00 | 88.00 | 78.00 | 90.00 | 81.00 |
| Expectile | dense | 62.00* | 89.00 | 91.00 | 84.00 | 90.00 | 84.00 |
| Expectile | sparse | 5.89 | 5.78 | 5.82 | 5.56 | 6.00 | 5.78 |
| PseudoHuber | dense | 67.00 | 186.00 | 188.00 | 171.00 | 187.00 | 175.00 |
| PseudoHuber | sparse | 7.43 | 13.50 | 14.00 | 13.00 | 14.25 | 13.50 |
| Student-t | dense | 77.00* | 371.00 | 375.00 | 356.00 | 377.00 | 362.00 |
| Student-t | sparse | 9.40 | 26.00 | 26.50 | 24.50 | 27.00 | 26.50 |

## GLM comparisons (initial run)

| Model | Matrix | Python GLM | R GLM |
|---|---|---:|---:|
| Gaussian | dense | 515.14 | 2061.00 |
| Gaussian | sparse | 510.12 | 2147.00 |
| Logistic | dense | 3557.27* | 54113.00* |
| Logistic | sparse | 20618.13* | 98913.00* |
| Probit | dense | 3332.18* | 38219.00* |
| Probit | sparse | 3575.12* | 40890.00* |
| Poisson | dense | 1135.25 | 11063.00 |
| Poisson | sparse | 888.51 | 6091.00 |
| Gamma | dense | 89725.63* | 730592.00* |
| Gamma | sparse | 306984.28* | 364979.00* |
| Negative-binomial | dense | 10307.62 | 17150.00* |
| Negative-binomial | sparse | 7909.84* | 13128.00* |

## Julia Spectral-MM baseline (initial run)

| Model | Dense | Sparse |
|---|---:|---:|
| Gaussian | 347.12* | 19.22* |
| Logistic | 1740.77 | 363.36 |
| Probit | 899.94 | 175.27 |
| Poisson | 855.62* | 28.94* |
| Gamma | 135.47* | 15.18* |
| Negative-binomial | 164.37* | 8.06* |
| SmoothQuantile | 367.19* | 21.95 |
| Expectile | 98.00 | 6.57 |
| PseudoHuber | 101.82 | 7.83 |
| Student-t | 110.34* | 9.70 |

## Interpretation and limitations

For Logistic/Probit, statsmodels detected complete separation or perfect prediction, while R reported fitted probabilities of 0/1. Even a small gradient does not establish a unique finite maximum-likelihood estimate. These timings measure attainment of numerical stopping criteria.

Both Gamma GLM implementations exhausted 1000 iterations without meeting the criterion. Their long runtimes cannot be used as successful-solve speedup denominators. Some negative-binomial GLM results also failed the common gradient threshold.

Several initial timings varied by multiples between minimum and maximum. Repeats alternate execution order to reduce order effects. Python and R share the same C++ library, so differences between measurement periods cannot be attributed directly to language performance. Julia was also timed in a different period.

Fixed parameters for the four non-GLM models: SmoothQuantile(tau=.25, smoothing=.1), Expectile(tau=.25), PseudoHuber(delta=1), Student-t(nu=4, sigma=1). Student-t uses MM surrogate weights.

## Spectral-MM results failing the gradient criterion

| Interface | Model | Matrix | Termination reason | Relative gradient |
|---|---|---|---|---:|
| Python | Gaussian | sparse | stalled_step | 1.19e-08 |
| Python | Poisson | dense | stalled_step | 1.04e-08 |
| Python | Gamma | dense | stalled_step | 2.68e-08 |
| Python | Gamma | sparse | stalled_step | 3.43e-08 |
| Python | Negative-binomial | dense | stalled_step | 2.39e-08 |
| Python | Negative-binomial | sparse | stalled_step | 1.72e-08 |
| Python | SmoothQuantile | dense | stalled_step | 2.74e-08 |
| Python | Expectile | dense | stalled_step | 1.15e-08 |
| Python | Student-t | dense | stalled_step | 5.28e-08 |
| R | Gaussian | sparse | stalled_step | 1.19e-08 |
| R | Poisson | dense | stalled_step | 1.04e-08 |
| R | Gamma | dense | stalled_step | 2.68e-08 |
| R | Gamma | sparse | stalled_step | 3.43e-08 |
| R | Negative-binomial | dense | stalled_step | 2.39e-08 |
| R | Negative-binomial | sparse | stalled_step | 1.72e-08 |
| R | SmoothQuantile | dense | stalled_step | 2.74e-08 |
| R | Expectile | dense | stalled_step | 1.15e-08 |
| R | Student-t | dense | stalled_step | 5.28e-08 |

## Raw records and reproduction

- `python.csv` / `R.csv` / `Julia.csv`: initial-run summaries.
- `python_repeat.csv` / `R_repeat.csv`: native repeat summaries.
- `*_samples.csv`: individual timings, including batch counts.
- `config.json` / `environment_*`: configuration and runtime environment; the Python record includes the shared-library SHA256.
- Binary inputs can be regenerated. The cleaned archive retains results and excludes `data/`.

```sh
BENCH_OUT=build/family-benchmark
julia --compiled-modules=existing --project=. benchmark/native/family_performance.jl 1000 "$BENCH_OUT"
python3 benchmark/native/family_performance.py --size 1000 --output "$BENCH_OUT" --language both
python3 benchmark/native/repeat_family_native.py --output "$BENCH_OUT" --language both
python3 benchmark/native/report_family_performance.py --output "$BENCH_OUT"
python3 benchmark/native/export_latex.py --results "$BENCH_OUT"
```
