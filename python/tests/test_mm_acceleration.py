"""Numerical checks for accelerated native MM with strict outer stopping."""
import unittest
import numpy as np
from scipy import sparse
from spectralmm._api import fit


class MMAccelerationTest(unittest.TestCase):
    def test_quadratic_solution_with_ridge_and_short_inner_solves(self):
        rng = np.random.default_rng(713)
        X = rng.normal(size=(120, 8)) * np.geomspace(0.3, 2., 8)
        design_matrix = np.column_stack([np.ones(120), X])
        y = np.einsum('ij,j->i', design_matrix, rng.normal(size=9)) + rng.normal(size=120) * 0.1
        for design in (X, sparse.csc_matrix(X)):
            for penalize_intercept in (False, True):
                for inner_maxiter in (2, 50):
                    with self.subTest(sparse=sparse.issparse(design),
                                      penalize_intercept=penalize_intercept,
                                      inner_maxiter=inner_maxiter):
                        ridge = .3
                        penalty = np.full(9, ridge)
                        if not penalize_intercept:
                            penalty[0] = 0.
                        expected = np.linalg.solve(np.einsum('ij,ik->jk', design_matrix, design_matrix) + np.diag(penalty), np.einsum('ij,i->j', design_matrix, y))
                        model = fit(design, y, family='gaussian', solver='mm',
                            fit_intercept=True, penalize_intercept=penalize_intercept,
                            inference=False, ridge=ridge, rank=2, krylovdim=9,
                            inner_maxiter=inner_maxiter, maxiter=1000,
                            nesterov=False, accept_negligible=False, accept_stalled=False,
                            gtol=1e-8, relgtol=1e-10)
                        self.assertTrue(model.info['gradient_converged'])
                        np.testing.assert_allclose(model.coef, expected, atol=2e-7, rtol=2e-7)

    def test_poisson_matches_cholesky(self):
        rng = np.random.default_rng(714)
        X = rng.normal(size=(150, 9))
        y = rng.poisson(np.exp(.1 + np.einsum('ij,j->i', X, rng.normal(scale=.15, size=9)))).astype(float)
        opts = dict(family='poisson', inference=False, ridge=.1, maxiter=1000,
                    rank=2, krylovdim=10, accept_negligible=False,
                    accept_stalled=False, gtol=1e-8, relgtol=1e-10)
        reference = fit(X, y, solver='cho', **opts)
        self.assertTrue(reference.info['gradient_converged'])
        for design in (X, sparse.csc_matrix(X)):
            for outer_momentum in (False, True):
                model = fit(design, y, solver='mm', nesterov=outer_momentum, **opts)
                self.assertTrue(model.info['gradient_converged'])
                np.testing.assert_allclose(model.coef, reference.coef, atol=2e-7, rtol=2e-7)


if __name__ == '__main__':
    unittest.main()
