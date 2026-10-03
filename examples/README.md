# Small-data model examples

These examples use the same saved data in all three languages: 100 observations and three feature columns. The default intercept gives four coefficients, with the intercept first. The data generator uses NumPy seed 20260930, independent normal features with standard deviation 0.5, and predictor 0.2 + 0.4 x1 - 0.3 x2 + 0.2 x3, except for the positive Gamma-inverse predictor described below.

All models use PCG, rank 3, at most 500 outer iterations, absolute gradient tolerance 1e-6 and relative tolerance 1e-8. Stalled-step acceptance is disabled. Initial coefficients are zero, except Gamma inverse, which starts at (2, 0, 0, 0). Rank 3 spans all but one of the four coefficient directions; these small examples illustrate usage and results, not large-scale performance.

Python and R call the shared C++ core. Julia uses the original Julia implementation. Negative binomial, Tweedie and grouped binomial use Julia's low-level API with an explicit intercept column; this API does not attach inference. Their inference is available in the Python/R results. All other examples use high-level APIs with automatic inference (the default cutoff is 50 parameters including the intercept).

On this saved data, Python and R meet the gradient criterion for all 14 models. Original Julia Probit reaches the 500-iteration limit (relative gradient approximately 1.51e-7), so its inference is unavailable at these tolerances. This exception is retained rather than classified as convergence.

The displayed coefficient tables are the Python results. The CSV files contain full-precision results for each language, including Wald chi-square statistics and their p-values. Standard errors, reference statistics and confidence intervals follow the model's inference convention. A t-reference coefficient p-value can differ from the asymptotic chi-square Wald p-value. Estimated coefficients need not equal the generating coefficients in a sample of 100 observations. Loss values should not be compared across different model families.

## Reproduce

```sh
# From the repository root; build the native library first (cpp/README.md).
python3 examples/run_python.py
Rscript examples/run_R.R
julia --project=. examples/run_julia.jl
python3 examples/render_results.py
```

Install the standard R package and the Python/Julia dependencies described in the repository guides first. The Julia runner additionally needs CSV and DataFrames in the active example environment; these are not SpectralMM runtime dependencies. The Python runner creates the common CSV and the Python/solver outputs; the other runners read that CSV. The renderer runs only after all three languages finish.

Complete iteration logs and fitted summaries: [Python](results/python.txt), [R](results/R.txt), [Julia](results/Julia.txt), [eight solvers](results/solvers.txt). Standalone Overleaf source: [Small_Model_Examples.tex](../docs/Small_Model_Examples.tex).

## Cross-language results

| Model | Language | Iterations | Relative gradient | Max coefficient difference from Python | Converged | Inference |
|---|---|---:|---:|---:|---|---|
| Gaussian identity | python | 1 | 1.94752e-08 | 0.00e+00 | true | ok |
| Gaussian identity | R | 1 | 1.94752e-08 | 4.44e-16 | true | ok |
| Gaussian identity | Julia | 1 | 1.94752e-08 | 3.89e-16 | true | ok |
| Bernoulli logit | python | 3 | 4.06673e-08 | 0.00e+00 | true | ok |
| Bernoulli logit | R | 3 | 4.06673e-08 | 2.22e-16 | true | ok |
| Bernoulli logit | Julia | 3 | 4.06673e-08 | 1.11e-16 | true | ok |
| Bernoulli probit | python | 8 | 4.57894e-08 | 0.00e+00 | true | ok |
| Bernoulli probit | R | 8 | 4.57894e-08 | 5.00e-16 | true | ok |
| Bernoulli probit | Julia | 500 | 1.50643e-07 | 9.66e-08 | false | unavailable |
| Poisson log | python | 4 | 3.58652e-09 | 0.00e+00 | true | ok |
| Poisson log | R | 4 | 3.58652e-09 | 4.44e-16 | true | ok |
| Poisson log | Julia | 4 | 3.58652e-09 | 5.55e-17 | true | ok |
| Gamma log | python | 8 | 9.40979e-09 | 0.00e+00 | true | ok |
| Gamma log | R | 8 | 9.40979e-09 | 3.89e-16 | true | ok |
| Gamma log | Julia | 8 | 9.40979e-09 | 5.55e-17 | true | ok |
| Negative binomial log | python | 7 | 3.43429e-08 | 0.00e+00 | true | ok |
| Negative binomial log | R | 7 | 3.43429e-08 | 4.44e-16 | true | ok |
| Negative binomial log | Julia | 7 | 3.43429e-08 | 5.55e-17 | true | low_level |
| Gaussian log | python | 6 | 1.71976e-08 | 0.00e+00 | true | ok |
| Gaussian log | R | 6 | 1.71976e-08 | 5.00e-16 | true | ok |
| Gaussian log | Julia | 6 | 1.71976e-08 | 1.11e-16 | true | ok |
| Gamma inverse | python | 4 | 1.60599e-08 | 0.00e+00 | true | ok |
| Gamma inverse | R | 4 | 1.60599e-08 | 8.88e-16 | true | ok |
| Gamma inverse | Julia | 4 | 1.60599e-08 | 4.44e-16 | true | ok |
| Tweedie log | python | 8 | 5.74469e-09 | 0.00e+00 | true | ok |
| Tweedie log | R | 8 | 5.74469e-09 | 4.72e-16 | true | ok |
| Tweedie log | Julia | 8 | 5.74469e-09 | 1.11e-16 | true | low_level |
| Binomial logit | python | 3 | 2.83872e-09 | 0.00e+00 | true | ok |
| Binomial logit | R | 3 | 2.83872e-09 | 4.44e-16 | true | ok |
| Binomial logit | Julia | 3 | 2.83872e-09 | 1.11e-16 | true | low_level |
| Smooth quantile | python | 6 | 1.74971e-08 | 0.00e+00 | true | ok |
| Smooth quantile | R | 6 | 1.74971e-08 | 1.67e-16 | true | ok |
| Smooth quantile | Julia | 6 | 1.74971e-08 | 5.55e-17 | true | ok |
| Expectile | python | 3 | 1.08293e-10 | 0.00e+00 | true | ok |
| Expectile | R | 3 | 1.08293e-10 | 3.33e-16 | true | ok |
| Expectile | Julia | 3 | 1.08293e-10 | 1.11e-16 | true | ok |
| Pseudo-Huber | python | 4 | 2.74657e-09 | 0.00e+00 | true | ok |
| Pseudo-Huber | R | 4 | 2.74657e-09 | 5.00e-16 | true | ok |
| Pseudo-Huber | Julia | 4 | 2.74657e-09 | 2.78e-17 | true | ok |
| Student-t | python | 13 | 5.58333e-09 | 0.00e+00 | true | ok |
| Student-t | R | 13 | 5.58333e-09 | 4.72e-16 | true | ok |
| Student-t | Julia | 13 | 5.58333e-09 | 5.55e-17 | true | ok |

## Gaussian identity

Normal errors with standard deviation 0.5. Family identifier: gaussian. Fixed options: defaults.

Converged in 1 iterations; loss 12.9148; relative gradient 1.94752e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.178739 | 0.0532126 | 3.35895 | 0.0011233 | [0.0731125, 0.284365] |
| x1 | 0.287177 | 0.114305 | 2.51237 | 0.0136599 | [0.0602829, 0.514072] |
| x2 | -0.167445 | 0.112773 | -1.48479 | 0.140875 | [-0.391298, 0.0564082] |
| x3 | 0.267741 | 0.0987282 | 2.7119 | 0.00792779 | [0.0717669, 0.463715] |

## Bernoulli logit

Binary responses with logistic probabilities. Family identifier: bernoulli. Fixed options: defaults.

Converged in 3 iterations; loss 67.1285; relative gradient 4.06673e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | -0.079785 | 0.209271 | -0.381251 | 0.703017 | [-0.48995, 0.33038] |
| x1 | 0.70347 | 0.46083 | 1.52653 | 0.126878 | [-0.19974, 1.60668] |
| x2 | 0.0122638 | 0.443908 | 0.0276269 | 0.97796 | [-0.85778, 0.882308] |
| x3 | 0.511956 | 0.397851 | 1.2868 | 0.198163 | [-0.267818, 1.29173] |

## Bernoulli probit

Binary responses with normal-CDF probabilities. Family identifier: probit. Fixed options: defaults.

Converged in 8 iterations; loss 63.7016; relative gradient 4.57894e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.345697 | 0.13406 | 2.57868 | 0.00991779 | [0.0829452, 0.60845] |
| x1 | 0.494629 | 0.292264 | 1.6924 | 0.0905689 | [-0.078198, 1.06746] |
| x2 | -0.20541 | 0.283849 | -0.72366 | 0.469274 | [-0.761745, 0.350924] |
| x3 | 0.392686 | 0.248272 | 1.58168 | 0.113724 | [-0.0939187, 0.879291] |

## Poisson log

Poisson counts with mean exp(eta). Family identifier: poisson. Fixed options: defaults.

Converged in 4 iterations; loss 94.1795; relative gradient 3.58652e-09; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.26438 | 0.08958 | 2.95133 | 0.0031641 | [0.0888065, 0.439954] |
| x1 | 0.235614 | 0.191292 | 1.2317 | 0.218063 | [-0.139312, 0.61054] |
| x2 | 0.00710371 | 0.191269 | 0.0371398 | 0.970374 | [-0.367777, 0.381984] |
| x3 | 0.267079 | 0.173804 | 1.53666 | 0.124376 | [-0.0735719, 0.607729] |

## Gamma log

Gamma responses with shape 5 and mean exp(eta). Family identifier: gamma. Fixed options: defaults.

Converged in 8 iterations; loss 121.225; relative gradient 9.40979e-09; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.208606 | 0.0461532 | 4.51986 | 1.76535e-05 | [0.116992, 0.300219] |
| x1 | 0.321096 | 0.0991411 | 3.23877 | 0.00164929 | [0.124302, 0.517889] |
| x2 | -0.282199 | 0.0978122 | -2.88511 | 0.00483161 | [-0.476355, -0.0880435] |
| x3 | 0.156231 | 0.0856304 | 1.82448 | 0.0711903 | [-0.0137442, 0.326206] |

## Negative binomial log

Negative-binomial counts with fixed theta=4. Family identifier: negative_binomial. Fixed options: {'theta': 4.0}.

Converged in 7 iterations; loss 830.861; relative gradient 3.43429e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.12349 | 0.111084 | 1.11168 | 0.266276 | [-0.0942308, 0.341211] |
| x1 | 0.28687 | 0.231154 | 1.24103 | 0.214594 | [-0.166184, 0.739925] |
| x2 | -0.501731 | 0.224874 | -2.23116 | 0.0256704 | [-0.942477, -0.0609855] |
| x3 | 0.140232 | 0.200141 | 0.700666 | 0.483512 | [-0.252038, 0.532502] |

## Gaussian log

Mean exp(eta), normal errors with standard deviation 0.2. Family identifier: gaussian_log. Fixed options: defaults.

Converged in 6 iterations; loss 3.0969; relative gradient 1.71976e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.194512 | 0.0222958 | 8.72415 | 8.15936e-14 | [0.150255, 0.238768] |
| x1 | 0.369816 | 0.043714 | 8.4599 | 2.99404e-13 | [0.283044, 0.456587] |
| x2 | -0.276474 | 0.0418762 | -6.60217 | 2.23539e-09 | [-0.359597, -0.19335] |
| x3 | 0.204954 | 0.0394911 | 5.18989 | 1.17318e-06 | [0.126565, 0.283344] |

## Gamma inverse

Gamma shape 5, mean 1/(2 + X beta); positive initial predictor. Family identifier: gamma_inverse. Fixed options: defaults.

Converged in 4 iterations; loss 35.2116; relative gradient 1.60599e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 1.90717 | 0.0954474 | 19.9813 | 5.65824e-36 | [1.71771, 2.09663] |
| x1 | 0.311016 | 0.201324 | 1.54485 | 0.125673 | [-0.0886102, 0.710642] |
| x2 | -0.606075 | 0.198053 | -3.06016 | 0.00286822 | [-0.999208, -0.212942] |
| x3 | 0.0849748 | 0.176731 | 0.480813 | 0.631743 | [-0.265834, 0.435784] |

## Tweedie log

Compound Poisson-Gamma responses, power=1.5 and dispersion=1. Family identifier: tweedie. Fixed options: {'power': 1.5}.

Converged in 8 iterations; loss 443.215; relative gradient 5.74469e-09; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.169866 | 0.0956993 | 1.775 | 0.079067 | [-0.0200951, 0.359828] |
| x1 | 0.56849 | 0.201697 | 2.81854 | 0.00585901 | [0.168125, 0.968855] |
| x2 | -0.413849 | 0.197253 | -2.09806 | 0.0385262 | [-0.805393, -0.0223042] |
| x3 | 0.206462 | 0.175411 | 1.17702 | 0.242099 | [-0.141727, 0.55465] |

## Binomial logit

Grouped counts with four trials per row. Family identifier: binomial. Fixed options: {'trials': 4.0}.

Converged in 3 iterations; loss 271.331; relative gradient 2.83872e-09; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.203769 | 0.104199 | 1.95557 | 0.0505156 | [-0.000457616, 0.407995] |
| x1 | 0.560934 | 0.226702 | 2.47433 | 0.0133487 | [0.116607, 1.00526] |
| x2 | -0.246536 | 0.221863 | -1.11121 | 0.266479 | [-0.68138, 0.188308] |
| x3 | 0.192511 | 0.193306 | 0.995889 | 0.319304 | [-0.186362, 0.571385] |

## Smooth quantile

Normal errors shifted to target the 0.25 quantile before smoothing. Family identifier: smooth_quantile. Fixed options: {'tau': 0.25, 'smoothing': 0.1}.

Converged in 6 iterations; loss 15.7187; relative gradient 1.74971e-08; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.172345 | 0.0481474 | 3.57954 | 0.000344202 | [0.0779783, 0.266713] |
| x1 | 0.478121 | 0.160722 | 2.97484 | 0.00293141 | [0.163113, 0.79313] |
| x2 | -0.284327 | 0.113768 | -2.49918 | 0.0124481 | [-0.507309, -0.0613456] |
| x3 | 0.381988 | 0.0788194 | 4.84638 | 1.25735e-06 | [0.227505, 0.536472] |

## Expectile

Normal errors shifted to target the 0.25 expectile. Family identifier: expectile. Fixed options: {'tau': 0.25}.

Converged in 3 iterations; loss 4.52245; relative gradient 1.08293e-10; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.214702 | 0.0531807 | 4.03722 | 5.40877e-05 | [0.11047, 0.318935] |
| x1 | 0.368848 | 0.111904 | 3.2961 | 0.000980357 | [0.14952, 0.588177] |
| x2 | -0.309377 | 0.109039 | -2.83729 | 0.00454978 | [-0.52309, -0.0956635] |
| x3 | 0.076899 | 0.0901609 | 0.852908 | 0.39371 | [-0.0998132, 0.253611] |

## Pseudo-Huber

Heavy-tailed errors: 0.5 times a Student-t(4) variate. Family identifier: pseudo_huber. Fixed options: {'delta': 1.0}.

Converged in 4 iterations; loss 18.2178; relative gradient 2.74657e-09; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.324575 | 0.0713007 | 4.5522 | 5.30873e-06 | [0.184828, 0.464322] |
| x1 | 0.409854 | 0.161259 | 2.54158 | 0.0110351 | [0.0937916, 0.725916] |
| x2 | -0.0768441 | 0.181104 | -0.42431 | 0.671339 | [-0.431801, 0.278112] |
| x3 | 0.108016 | 0.116685 | 0.925704 | 0.3546 | [-0.120682, 0.336714] |

## Student-t

Student-t(4) location errors with fixed scale 0.5. Family identifier: student_t. Fixed options: {'nu': 4.0, 'sigma': 0.5}.

Converged in 13 iterations; loss 66.0073; relative gradient 5.58333e-09; inference ok.

| Term | Estimate | SE | Statistic | p-value | 95% CI |
|---|---:|---:|---:|---:|---|
| (Intercept) | 0.217022 | 0.0596063 | 3.64092 | 0.000271665 | [0.100196, 0.333848] |
| x1 | 0.304619 | 0.114365 | 2.66357 | 0.00773156 | [0.0804682, 0.52877] |
| x2 | -0.355121 | 0.10166 | -3.49322 | 0.00047723 | [-0.554371, -0.155871] |
| x3 | 0.165891 | 0.135066 | 1.22822 | 0.219363 | [-0.0988329, 0.430615] |

## Eight solvers on the same Logistic data

All solvers start from zero on the same 100 x 3 input. Krylov relative/absolute tolerances are 1e-8/1e-12. These are numerical demonstrations, not timing comparisons.

| Solver | Iterations | Loss | Relative gradient |
|---|---:|---:|---:|
| pcg | 3 | 67.1285 | 4.06673e-08 |
| mm | 3 | 67.1285 | 4.06673e-08 |
| cg | 3 | 67.1285 | 4.66513e-09 |
| cho | 3 | 67.1285 | 4.66513e-09 |
| cgls | 3 | 67.1285 | 4.66513e-09 |
| crls | 3 | 67.1285 | 4.66513e-09 |
| lsqr | 3 | 67.1285 | 4.66513e-09 |
| lsmr | 3 | 67.1285 | 4.66513e-09 |
