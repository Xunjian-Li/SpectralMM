"""Generate reproducible 100 x 3 examples and run all native model families."""
from pathlib import Path
import contextlib
import csv
import sys
import numpy as np
from scipy import special, stats, optimize

ROOT = Path(__file__).resolve().parents[1]
from spectralmm import fit

HERE = ROOT / 'examples'
OUT = HERE / 'results'
CASES = [
    ('gaussian', {}, 'Gaussian identity', 'Normal errors with standard deviation 0.5.'),
    ('bernoulli', {}, 'Bernoulli logit', 'Binary responses with logistic probabilities.'),
    ('probit', {}, 'Bernoulli probit', 'Binary responses with normal-CDF probabilities.'),
    ('poisson', {}, 'Poisson log', 'Poisson counts with mean exp(eta).'),
    ('gamma', {}, 'Gamma log', 'Gamma responses with shape 5 and mean exp(eta).'),
    ('negative_binomial', {'theta': 4.}, 'Negative binomial log', 'Negative-binomial counts with fixed theta=4.'),
    ('gaussian_log', {}, 'Gaussian log', 'Mean exp(eta), normal errors with standard deviation 0.2.'),
    ('gamma_inverse', {}, 'Gamma inverse', 'Gamma shape 5, mean 1/(2 + X beta); positive initial predictor.'),
    ('tweedie', {'power': 1.5}, 'Tweedie log', 'Compound Poisson-Gamma responses, power=1.5 and dispersion=1.'),
    ('binomial', {'trials': 4.}, 'Binomial logit', 'Grouped counts with four trials per row.'),
    ('smooth_quantile', {'q': .25, 'smoothing': .1}, 'Smooth quantile', 'Normal errors shifted to target the 0.25 quantile before smoothing.'),
    ('expectile', {'q': .25}, 'Expectile', 'Normal errors shifted to target the 0.25 expectile.'),
    ('pseudo_huber', {'delta': 1.}, 'Pseudo-Huber', 'Heavy-tailed errors: 0.5 times a Student-t(4) variate.'),
    ('student_t', {'nu': 4., 'sigma': .5}, 'Student-t', 'Student-t(4) location errors with fixed scale 0.5.'),
]
FIELDS = ['family', 'term', 'estimate', 'std_error', 'statistic', 'p_value',
          'lower', 'upper', 'wald_chisq', 'wald_p_value', 'loss', 'relgradnorm',
          'iterations', 'converged', 'inference_status']


def generate_data():
    rng = np.random.default_rng(20260930)
    X = rng.normal(size=(100, 3)) * .5
    eta = .2 + X @ np.array([.4, -.3, .2])
    mu = np.exp(eta)
    y = {}
    y['gaussian'] = eta + .5 * rng.normal(size=100)
    y['bernoulli'] = rng.binomial(1, special.expit(eta)).astype(float)
    y['probit'] = rng.binomial(1, special.ndtr(eta)).astype(float)
    y['poisson'] = rng.poisson(mu).astype(float)
    y['gamma'] = rng.gamma(5, mu / 5)
    y['negative_binomial'] = rng.negative_binomial(4, 4 / (4 + mu)).astype(float)
    y['gaussian_log'] = mu + .2 * rng.normal(size=100)
    y['gamma_inverse'] = rng.gamma(5, 1 / (5 * (2 + eta - .2)))
    # Compound Poisson-Gamma construction matching src/simulation.jl, phi=1.
    power = 1.5
    counts = rng.poisson(mu ** (2-power) / (2-power))
    y['tweedie'] = np.zeros(100)
    positive = counts > 0
    y['tweedie'][positive] = rng.gamma(counts[positive]*(2-power)/(power-1),
                                      (power-1)*mu[positive]**(power-1))
    y['binomial'] = rng.binomial(4, special.expit(eta)).astype(float)
    y['smooth_quantile'] = eta + .5*(rng.normal(size=100)-stats.norm.ppf(.25))
    def expectile_equation(e):
        phi, Phi = stats.norm.pdf(e), stats.norm.cdf(e)
        return .25*(phi-e*(1-Phi))-.75*(phi+e*Phi)
    shift = optimize.brentq(expectile_equation, -4, 4)
    y['expectile'] = eta + .5*(rng.normal(size=100)-shift)
    y['pseudo_huber'] = eta + .5*rng.standard_t(4, size=100)
    y['student_t'] = eta + .5*rng.standard_t(4, size=100)
    names = ['x1', 'x2', 'x3'] + [case[0] for case in CASES]
    np.savetxt(HERE / 'data.csv', np.column_stack([X] + [y[c[0]] for c in CASES]),
               delimiter=',', header=','.join(names), comments='', fmt='%.17g')


def main():
    OUT.mkdir(exist_ok=True)
    generate_data()
    data = np.genfromtxt(HERE / 'data.csv', delimiter=',', names=True)
    X = np.column_stack([data[f'x{i}'] for i in (1, 2, 3)])
    with (OUT / 'python.csv').open('w', newline='') as result_file, (OUT / 'python.txt').open('w') as log:
        writer = csv.DictWriter(result_file, fieldnames=FIELDS)
        writer.writeheader()
        for family, parameters, title, description in CASES:
            beta0 = [2., 0., 0., 0.] if family == 'gamma_inverse' else [0., 0., 0., 0.]
            with contextlib.redirect_stdout(log):
                print(f'\n=== {title}: X.shape=(100, 3), four coefficients including intercept ===')
                print(description)
                model = fit(X, data[family], family, family_options=parameters, beta0=beta0,
                            solver='pcg', rank=3, maxiter=500, gtol=1e-6, relgtol=1e-8,
                            accept_stalled=False, verbose=True)
                print(model.summary())
                print('First five fitted values:', model.predict(X)[:5])
            inf = model.inference
            for j, term in enumerate(['(Intercept)', 'x1', 'x2', 'x3']):
                row = dict(family=family, term=term, estimate=model.coef[j], loss=model.info['loss'],
                           relgradnorm=model.info['relgradnorm'], iterations=model.info['iterations'],
                           converged=model.info['gradient_converged'], inference_status=inf['status'])
                if inf['status'] == 'ok':
                    for key in ('std_error', 'statistic', 'p_value', 'wald_chisq', 'wald_p_value'):
                        row[key] = inf[key][j]
                    row['lower'], row['upper'] = inf['conf_int'][j]
                writer.writerow(row)
            print(f'{family}: gradient_converged={model.info["gradient_converged"]}, inference={inf["status"]}')
    with (OUT / 'solvers.csv').open('w', newline='') as f, (OUT / 'solvers.txt').open('w') as log:
        writer = csv.writer(f)
        writer.writerow(['solver', 'iterations', 'loss', 'relgradnorm', 'converged', 'intercept', 'x1', 'x2', 'x3'])
        for solver in ('pcg', 'mm', 'cg', 'cho', 'cgls', 'crls', 'lsqr', 'lsmr'):
            with contextlib.redirect_stdout(log):
                model = fit(X, data['bernoulli'], 'bernoulli', beta0=np.zeros(4), solver=solver,
                            rank=3, maxiter=500, gtol=1e-6, relgtol=1e-8, krylov_rtol=1e-8,
                            krylov_atol=1e-12, accept_stalled=False, verbose=True)
                print(model.summary())
            writer.writerow([solver, model.info['iterations'], model.info['loss'], model.info['relgradnorm'],
                             model.info['gradient_converged'], *model.coef])

if __name__ == '__main__':
    main()
