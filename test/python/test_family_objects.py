import unittest
from dataclasses import FrozenInstanceError
import numpy as np
import spectralmm as sm


class FamilyObjectTest(unittest.TestCase):
    def test_old_new_matrix_formula(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(100, 6))
        y = .3 + X @ np.arange(1, 7)/10 + rng.normal(size=100)*.2
        data = {f'x{i+1}': X[:, i] for i in range(6)}
        data['y'] = y
        formula = 'y ~ ' + ' + '.join(f'x{i+1}' for i in range(6))
        cases = [(sm.PseudoHuber(), 'pseudo_huber', {'delta': 1.}),
                 (sm.Expectile(.25), 'expectile', {'tau': .25}),
                 (sm.SmoothQuantile(.25, .1), 'smooth_quantile', {'tau': .25, 'smoothing': .1}),
                 (sm.StudentT(4.), 'student_t', {'nu': 4.})]
        for spec, name, options in cases:
            with self.subTest(family=name):
                old = sm.fit(X, y, family=name, family_options=options, rank=5, solver='pcg')
                new = sm.fit(X, y, family=spec, rank=5, solver='pcg')
                form = sm.fit(formula, data=data, family=spec, rank=5, solver='pcg')
                if name == 'pseudo_huber':
                    bare = sm.fit(X, y, family=name, rank=5, solver='pcg')
                    np.testing.assert_array_equal(bare.params, new.params)
                for result in (new, form):
                    np.testing.assert_allclose(old.params, result.params, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(old.fittedvalues, result.fittedvalues, rtol=1e-12, atol=1e-12)
                    self.assertEqual(old.objective, result.objective)
                    self.assertEqual(result.diagnostics['rank'], 5)
                    self.assertEqual(result.diagnostics['solver'], 'pcg')
                    self.assertIsNone(result.link)
                np.testing.assert_allclose(form.predict(data), new.predict(X), rtol=1e-12, atol=1e-12)
                for options in ({}, None, {'delta': 2}):
                    with self.assertRaisesRegex(TypeError, 'family_options'):
                        sm.fit(X, y, family=spec, family_options=options)
                with self.assertRaises(ValueError):
                    sm.glm(X, y, family=spec)

    def test_validation_and_immutability(self):
        for constructor, field in ((sm.PseudoHuber, 'delta'), (sm.Expectile, 'q'),
                                    (sm.SmoothQuantile, 'q'), (sm.StudentT, 'nu')):
            for value in (0, -1, float('nan'), float('inf'), True, '1', [1]):
                with self.subTest(constructor=constructor, value=value):
                    with self.assertRaisesRegex(ValueError, field): constructor(value)
        for constructor in (sm.Expectile, sm.SmoothQuantile):
            with self.assertRaisesRegex(ValueError, 'q'): constructor(1)
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'epsilon'): sm.SmoothQuantile(epsilon=value)
        with self.assertRaises(FrozenInstanceError): sm.PseudoHuber().delta = 2
        self.assertEqual(sm.Expectile().q, .5)
        self.assertEqual(sm.SmoothQuantile().epsilon, .1)
        self.assertEqual(sm.StudentT().nu, 4.)
