#ifndef SPECTRALMM_C_API_H
#define SPECTRALMM_C_API_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental ABI v1. All arrays are borrowed for the duration of a call.
   Dense values are column-major. CSC uses sorted, unique, zero-based int64
   indices. No C++ exception crosses this boundary. Caller owns outputs.
   Empty CSC matrices may use NULL values/indices with a valid indptr. */
typedef struct {
    int64_t n, p, nnz;
    int32_t storage; /* 0: dense, 1: CSC */
    const double *values;
    const int64_t *indptr, *indices;
} smm_matrix;
typedef struct {
    int32_t family; /* family IDs documented below */
    int32_t nesterov;
    int64_t rank, maxiter, inner_maxiter, krylovdim;
    double ridge, floor, gtol, relgtol, eta_max, correction_tol, resid_tol;
} smm_options;
typedef struct {
    int64_t iterations, inner_iterations, restarts, corrections;
    int32_t converged;
    int32_t termination; /* 0: gradient tolerance, 1: iteration limit,
                           2: line search stalled, 3: inner breakdown,
                           4: negligible step, 5: stalled near tolerance */
    double loss, gradnorm, relgradnorm, eigresidual;
} smm_info;
/* Additive Krylov interface; ABI v1 of smm_fit remains unchanged. */
typedef struct {
    int32_t solver; /* 0: CG, 1: CGLS, 2: CRLS, 3: LSQR, 4: LSMR,
                       5: Cholesky, 6: spectral MM */
    int32_t jacobi;
    double rtol, atol;
} smm_krylov_options;
int32_t smm_fit_krylov(const smm_matrix *X, const double *y, const double *beta0,
                       const smm_options *options, const smm_krylov_options *krylov,
                       double *coef, smm_info *info, char *error, size_t error_capacity);
/* Additive extended interface. Legacy smm_fit/smm_fit_krylov remain strict.
   Codes 4/5 are accepted step-based termination, not proof of a small gradient. */
typedef struct {
    int32_t accept_negligible, accept_stalled;
    double negligible_step_tol, step_reltol, stalled_relgtol;
} smm_stop_options;
typedef struct {
    int64_t iteration, inner_iterations, restarts, corrections;
    double loss, gradnorm, relgradnorm;
} smm_trace_entry;
/* Additive detailed trace. Legacy smm_trace_entry layout is unchanged.
   spectrum: 0 none, 1 initial, 2 reuse, 3 correct, 4 restart,
             5 correct-fail, 6 restart+correct, 7 restart+fail.
   inner_iterations is cumulative; inner is per accepted update.
   inner_residual is ||g + H*delta|| / ||g|| for the unscaled WLS step.
   Row 0 and inapplicable fields use NaN (inner=-1). */
typedef struct {
    int64_t iteration, inner_iterations, restarts, corrections, inner;
    double loss, gradnorm, relgradnorm, inner_residual, eigresidual, step;
    int32_t spectrum;
} smm_trace_detail;
void smm_default_stop_options(smm_stop_options *options);
/* krylov=NULL selects spectral PCG; stop=NULL selects Julia-style defaults.
   trace=NULL disables tracing; otherwise capacity must be >= maxiter+1.
   trace_size may be NULL only when tracing is disabled. */
int32_t smm_fit_extended(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *options, const smm_krylov_options *krylov,
    const smm_stop_options *stop, double *coef, smm_info *info,
    smm_trace_entry *trace, int64_t trace_capacity, int64_t *trace_size,
    char *error, size_t error_capacity);
/* Additive family interface; existing ABI structs/functions remain unchanged.
   family IDs: 0 gaussian, 1 bernoulli, 2 probit, 3 poisson, 4 gamma,
   5 negative_binomial, 6 smooth_quantile, 7 expectile, 8 pseudo_huber,
   9 student_t, 10 gaussian_log, 11 gamma_inverse, 12 tweedie, 13 binomial.
   trials is a borrowed scalar (count=1) or length-n vector, only for binomial. */
typedef struct {
    double theta, tau, smoothing, delta, nu, sigma, power;
    const double *trials;
    int64_t trials_count;
} smm_family_options;
void smm_default_family_options(smm_family_options *options);
int32_t smm_fit_family(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *options, const smm_family_options *family_options,
    const smm_krylov_options *krylov, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_entry *trace, int64_t trace_capacity,
    int64_t *trace_size, char *error, size_t error_capacity);
/* Feature-matrix interface: intercept adds an implicit leading column of ones.
   beta0/coef have p+intercept entries, intercept first. With intercept=0 this
   is the legacy design-matrix interface; all columns are penalized.
   Flags must be 0 or 1. Existing entry points and structs retain ABI v1. */
int32_t smm_fit_intercept(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *options, const smm_family_options *family_options,
    const smm_krylov_options *krylov, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_entry *trace, int64_t trace_capacity,
    int64_t *trace_size, char *error, size_t error_capacity,
    int32_t intercept, int32_t penalize_intercept);
/* Same fit semantics as smm_fit_intercept, with an optional detailed trace. */
int32_t smm_fit_logged(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *options, const smm_family_options *family_options,
    const smm_krylov_options *krylov, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_detail *trace, int64_t trace_capacity,
    int64_t *trace_size, char *error, size_t error_capacity,
    int32_t intercept, int32_t penalize_intercept);
/* Optional normalized, unpenalized log likelihood per trace row.
   likelihood has trace_capacity elements; NaN means unavailable. */
int32_t smm_fit_logged_stats(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *options, const smm_family_options *family_options,
    const smm_krylov_options *krylov, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_detail *trace, int64_t trace_capacity,
    int64_t *trace_size, char *error, size_t error_capacity,
    int32_t intercept, int32_t penalize_intercept, double *likelihood, double dispersion);
/* Optional, dense p-by-p post-fit covariance (column major). Fit is unchanged.
   cov_type: 0 auto (GLM=model, residual=sandwich HC1), 1 model, 2 sandwich HC1.
   dispersion=NaN selects family defaults; caller checks fit convergence first.
   Requires ridge=0, n>p and nonsingular information; errors leave fit intact. */
typedef struct {
    int32_t cov_type, estimated_dispersion;
    double df_resid, dispersion, rcond;
} smm_inference_info;
int32_t smm_infer(const smm_matrix *X, const double *y, const double *coef,
    const smm_options *options, const smm_family_options *family_options,
    int32_t intercept, int32_t cov_type, double dispersion, double *covariance,
    smm_inference_info *info, char *error, size_t error_capacity);
int32_t smm_abi_version(void);
void smm_default_options(smm_options *options);
/* beta0 may be NULL: use mean-response initialization if a nonzero constant
   column exists, zero otherwise. ridge penalizes ALL coefficients. */
int32_t smm_fit(const smm_matrix *X, const double *y, const double *beta0,
                const smm_options *options, double *coef, smm_info *info,
                char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
