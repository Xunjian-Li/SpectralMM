# Predictors exclude the intercept; n = 2p. Edit this NamedTuple.
(
    dense_sizes = [5000], sparse_sizes = [5000],
    threads = 12,                 # BLAS threads; integer or "max".
    samples = 5,
    rank = 5, krylovdim = 12, floor = 1e-6,
    maxiter = 1000, inner_maxiter = 200,
    gtol = 1e-6, relgtol = 1e-8, eta_max = 0.6,
    correction_tol = 0.05, resid_tol = 0.5,
    inner_nesterov = true, outer_nesterov = true,
    accept_stalled = true,        # Preserve the original Julia experiment.
    seed = 3124, nnz_per_row = 20,
)
