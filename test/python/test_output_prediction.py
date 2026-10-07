import contextlib
import io
import unittest
from pathlib import Path
import numpy as np
import spectralmm as sm
from spectralmm.model import _statistics

class OutputPredictionTest(unittest.TestCase):
    def test_likelihood_and_unchanged_fit(self):
        fixture=Path(__file__).with_name('data.csv')
        if not fixture.exists(): fixture=Path(__file__).resolve().parents[2]/'examples/data.csv'
        d=np.genfromtxt(fixture,delimiter=',',names=True)
        X=np.column_stack([d[k] for k in ('x1','x2','x3')])
        for family in ('gaussian','gaussian_log','bernoulli','probit','poisson','gamma','gamma_inverse',
                       'negative_binomial','binomial','student_t','pseudo_huber','expectile','smooth_quantile','tweedie'):
            options={'trials':5} if family=='binomial' else {}
            for dispersion in (None,.7) if family in ('gaussian','gamma') else (None,):
                kw=dict(family=family,family_options=options,inference=False,ridge=.1,dispersion=dispersion,maxiter=4)
                quiet=sm.fit(X,d[family],**kw)
                output=io.StringIO()
                with contextlib.redirect_stdout(output):
                    logged=sm.fit(X,d[family],verbose=True,trace=True,**kw)
                np.testing.assert_array_equal(quiet.params,logged.params)
                text=output.getvalue()
                for label in ('GradNorm','Stepsize'): self.assertIn(label,text)
                for label in ('InnerRes','EigRes'): self.assertNotIn(label,text)
                row=logged.trace[-1]
                self.assertEqual(row['loss'],quiet.objective)
                if family in ('pseudo_huber','expectile','smooth_quantile','tweedie'):
                    self.assertIn('Objective',text); self.assertTrue(np.isnan(row['loglikelihood']))
                else:
                    self.assertIn('LogLik',text)
                    self.assertAlmostEqual(row['loglikelihood'],logged.llf,places=8)
        # All eight solvers share the same display semantics.
        for solver in ('pcg','mm','cg','cho','cgls','crls','lsqr','lsmr'):
            with contextlib.redirect_stdout(io.StringIO()):
                m=sm.fit(X,d['bernoulli'],family='bernoulli',solver=solver,trace=True,verbose=True,inference=False)
            self.assertAlmostEqual(m.trace[-1]['loglikelihood'],m.llf,places=9)

    def test_nonglm_newdata(self):
        rng=np.random.default_rng(52); X=rng.normal(size=(100,3)); y=.2+X@np.array([.1,.3,-.2])+rng.normal(size=100)*.2
        new=rng.normal(size=(7,3))
        for family in (sm.PseudoHuber(),sm.Expectile(.3),sm.SmoothQuantile(.3,.2),sm.StudentT(4)):
            for intercept in (True,False):
                m=sm.fit(X,y,family=family,fit_intercept=intercept,inference=False)
                expected=new@m.coef_+m.intercept_
                np.testing.assert_allclose(m.predict(new),expected)
                data=dict(y=y,x1=X[:,0],x2=X[:,1],x3=X[:,2])
                f=sm.fit('y~x1+x2+x3' if intercept else 'y~0+x1+x2+x3',data=data,family=family,inference=False)
                np.testing.assert_allclose(f.predict(dict(x1=new[:,0],x2=new[:,1],x3=new[:,2])),expected,atol=1e-9)

    def test_original_c_abi(self):
        import ctypes as C
        from spectralmm import _api as a
        lib=a._library(); old=lib.smm_fit_logged
        old.argtypes=lib.smm_fit_logged_stats.argtypes[:-2]; old.restype=C.c_int32
        X=np.asfortranarray(np.arange(12,dtype=float).reshape(6,2)/10); y=np.arange(6,dtype=float)/6
        desc=a._Matrix(6,2,0,0,X.ctypes.data_as(a._D),None,None)
        options=a._Options(); lib.smm_default_options(C.byref(options)); options.family=0
        family=a._Family(); lib.smm_default_family_options(C.byref(family))
        results=[]
        for fn in (old,lib.smm_fit_logged_stats):
            coef=np.zeros(3); info=a._Info(); rows=(a._Trace*(options.maxiter+1))(); n=C.c_int64(); error=C.create_string_buffer(1024)
            args=[C.byref(desc),y.ctypes.data_as(a._D),None,C.byref(options),C.byref(family),None,None,coef.ctypes.data_as(a._D),C.byref(info),rows,len(rows),C.byref(n),error,len(error),1,0]
            ll=(C.c_double*len(rows))()
            if fn is not old: args.extend([ll,float('nan')])
            self.assertEqual(fn(*args),0,error.value)
            results.append((coef.copy(),info.loss,n.value,[rows[i].loss for i in range(n.value)]))
        np.testing.assert_array_equal(results[0][0],results[1][0]); self.assertEqual(results[0][1:],results[1][1:])

    def test_unified_summary_and_q(self):
        rng=np.random.default_rng(4); X=rng.normal(size=(100,3)); y=.3+X[:,0]*.2+rng.normal(size=100)*.3
        for family in ('expectile','smooth_quantile'):
            a=sm.fit(X,y,family=family,family_options={'q':.3})
            b=sm.fit(X,y,family=family,family_options={'tau':.3})
            np.testing.assert_array_equal(a.params,b.params)
            self.assertIn('q',a.family_options); self.assertNotIn('tau',a.family_options)
            with self.assertRaisesRegex(TypeError,'both'): sm.fit(X,y,family=family,family_options={'q':.3,'tau':.3})
            text=a.summary(); self.assertEqual(text.count('Family:'),2)
            self.assertIn('q=0.3',text); self.assertNotIn('tau',text)
        for family,response,stat in [('gaussian',y,'t'),('bernoulli',(y>0).astype(float),'z')]:
            m=sm.fit(X,response,family=family)
            text=m.summary(); self.assertTrue(text.startswith('SpectralMM Regression Model'))
            self.assertIn(stat+' statistic',text); self.assertIn('p-value',text)
            self.assertLess(text.index('Stopping criterion'),text.index('Inference:'))
            with contextlib.redirect_stdout(io.StringIO()) as out: sm.fit(X,response,family=family,verbose=True)
            self.assertEqual(out.getvalue().count('Family:'),2)
        m=sm.fit(X,y,solver='cho',inference=False)
        self.assertNotIn('Rank:',m.summary()); self.assertIn('Reason:',m.summary())
