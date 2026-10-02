import csv
import os
from pathlib import Path
import unittest
import numpy as np
import spectralmm as sm
from spectralmm import _api

CASES=[('gaussian','gaussian','identity',{}),('bernoulli','binomial','logit',{}),
       ('probit','binomial','probit',{}),('poisson','poisson','log',{}),
       ('gamma','gamma','log',{}),('negative_binomial','negative_binomial','log',{'theta':4.})]
HERE=Path(__file__).resolve().parent
FIXTURE=HERE/'data.csv'
if not FIXTURE.exists(): FIXTURE=HERE.parents[1]/'examples/data.csv'

class StatisticalAPITest(unittest.TestCase):
    def test_glm_formula_matrix_and_legacy(self):
        data=np.genfromtxt(FIXTURE,delimiter=',',names=True)
        X=np.column_stack([data[k] for k in ('x1','x2','x3')])
        exports=[]
        for col,family,link,options in CASES:
            with self.subTest(family=family,link=link):
                y=data[col]; dat={k:data[k] for k in data.dtype.names}; dat['y']=y
                kw=dict(family=family,link=link,family_options=options,start=np.zeros(4))
                m=sm.glm(X,y,**kw)
                f=sm.glm('y~x1+x2+x3',data=dat,**kw)
                explicit=sm.glm(np.column_stack([np.ones(len(y)),X]),y,fit_intercept=False,**kw)
                raw=_api.fit(X,y,col,family_options=options,beta0=np.zeros(4))
                np.testing.assert_array_equal(m.params,raw.coef)
                for other in (f,explicit):
                    np.testing.assert_allclose(m.params,other.params,atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(m.fittedvalues,other.fittedvalues,atol=1e-10)
                    np.testing.assert_allclose(m.linear_predictor,other.linear_predictor,atol=1e-10)
                    self.assertAlmostEqual(m.objective,other.objective,places=9)
                np.testing.assert_allclose(f.predict(dat),m.predict())
                np.testing.assert_allclose(f.predict(dat,which='linear'),m.linear_predictor)
                np.testing.assert_allclose(m.resid_response,y-m.fittedvalues)
                self.assertEqual(m.nobs,100); self.assertEqual(m.family,family); self.assertEqual(m.link,link)
                self.assertTrue(m.info['gradient_converged']); self.assertEqual(m.inference['status'],'ok')
                self.assertEqual(m.cov_params().shape,(4,4)); self.assertEqual(m.conf_int(alpha=.1).shape,(4,2))
                self.assertIn('SpectralMM optimization',m.summary())
                for quantity,values in [('coef',m.params),('eta',m.linear_predictor),('mu',m.fittedvalues),
                                        ('objective',[m.objective]),('deviance',[m.deviance]),('loglikelihood',[m.llf]),
                                        ('converged',[float(m.info['gradient_converged'])])]:
                    exports.extend((col,quantity,i,float(v)) for i,v in enumerate(values))
        if os.environ.get('SPECTRALMM_API_EXPORT'):
            with open(os.environ['SPECTRALMM_API_EXPORT'],'w',newline='') as out:
                w=csv.writer(out); w.writerow(['family','quantity','index','value']); w.writerows(exports)

    def test_categories_and_no_intercept(self):
        x=np.sin(np.arange(100)); g=np.array(['a','b','c','a']*25)
        y=.3+.4*x+.2*(g=='b')-.1*(g=='c')+.1*np.cos(np.arange(100)*.7)
        d=dict(y=y,x=x,g=g)
        f=sm.glm('y~x+C(g)',data=d)
        from patsy import dmatrices,build_design_matrices
        _,A=dmatrices('y~x+C(g)',d)
        m=sm.glm(np.asarray(A),y,fit_intercept=False)
        np.testing.assert_allclose(f.params,m.params,atol=1e-10)
        new=dict(x=np.array([.2,.5]),g=np.array(['c','b']))
        manual=np.asarray(build_design_matrices([f.design_info],new)[0])@f.params
        np.testing.assert_allclose(f.predict(new),manual)
        with self.assertRaises(Exception): f.predict(dict(x=[1.],g=['unseen']))
        f=sm.glm('y~0+x+C(g)',data=d)
        _,A=dmatrices('y~0+x+C(g)',d)
        m=sm.glm(np.asarray(A),y,fit_intercept=False)
        np.testing.assert_allclose(f.params,m.params,atol=1e-10)
        self.assertFalse(f.fit_intercept)
        self.assertIn('C(g)[a]',f.param_names)

    def test_control_limits_and_non_glm(self):
        X=np.random.default_rng(9).normal(size=(80,51)); y=np.sin(np.arange(80))
        m=sm.glm(X,y,control=sm.Control(maxiter=1))
        self.assertEqual(m.inference['status'],'skipped')
        with self.assertRaises(ValueError): m.cov_params()
        self.assertIn('Estimate',m.summary())
        for key in ('weights','freq_weights','var_weights','offset'):
            with self.assertRaises(NotImplementedError): sm.glm(X,y,**{key:np.ones(80)})
        with self.assertRaises(TypeError): sm.glm(X,y,start=np.zeros(52),beta0=np.zeros(52))
        with self.assertRaises(TypeError): sm.glm(X,y,control=sm.Control(maxiter=1),maxiter=2)
        with self.assertRaises(ValueError): sm.glm(X,y,family='pseudo_huber')
        m=sm.fit(X[:,:2],y,family='pseudo_huber')
        self.assertIsNone(m.link)
        with self.assertRaises(NotImplementedError): _=m.deviance
        f=sm.fit('y~x',data=dict(y=y,x=X[:,0]),family='pseudo_huber')
        self.assertEqual(f.nobs,80)
