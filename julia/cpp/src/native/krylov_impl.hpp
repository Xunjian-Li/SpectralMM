// Internal implementation, included in core.cpp's anonymous namespace.
// The LSQR/LSMR bidiagonal recurrences are adapted from SciPy's BSD-licensed
// implementations; see ../THIRD_PARTY_NOTICES.md. Stopping rules are unified
// here on the scaled normal residual, rather than SciPy's backward-error tests.
struct LeastSquaresOp {
    const Design& X;
    Vec sqrtw, invscale, ptmp, ntmp, normalbuf;
    double damping;
    LeastSquaresOp(const Design& x, double ridge)
        : X(x), sqrtw(x.x.n), invscale(x.x.p), ptmp(x.x.p), ntmp(x.x.n),
          normalbuf(x.x.n+(ridge>0 ? x.x.p : 0)), damping(std::sqrt(ridge)) {}
    void refresh(const Vec& w, double ridge, bool jacobi) {
        sqrtw=w.array().sqrt(); invscale.setOnes();
        if (jacobi) for (int64_t j=0;j<X.x.p;++j) {
            double d=X.diagonal(w,j)+((X.unpenalized && j==0) ? 0. : ridge);
            invscale[j]=1./std::sqrt(std::max(d,1e-12));
        }
    }
    void av_into(const Vec& u, Vec& result) {
        ptmp.array()=invscale.array()*u.array();
        X.mv_into(ptmp,ntmp);
        result.head(X.x.n).array()=sqrtw.array()*ntmp.array();
        if (damping>0) {
            result.tail(X.x.p)=damping*ptmp;
            if (X.unpenalized) result[X.x.n]=0.;
        }
    }
    void at_into(const Vec& u, Vec& result) {
        ntmp.array()=sqrtw.array()*u.head(X.x.n).array();
        X.tmv_into(ntmp,ptmp);
        if (damping>0) {
            ptmp.tail(X.x.p-X.unpenalized)+=damping*u.tail(X.x.p-X.unpenalized);
        }
        result.array()=invscale.array()*ptmp.array();
    }
    void normal_into(const Vec& u, Vec& result) { av_into(u,normalbuf); at_into(normalbuf,result); }
};
struct KrylovWorkspace {
    LeastSquaresOp A;
    Vec rhs,x,r,p,Hp,Hr,b,q,v,h,hbar,tmpP;
    KrylovWorkspace(const Design& X, double ridge, int solver)
        : A(X,ridge),rhs(X.x.p),x(X.x.p),r(X.x.p),p(X.x.p),Hp(X.x.p),Hr(X.x.p),
          b(solver==0 || solver==2 ? 0 : A.normalbuf.size()),q(b.size()),
          v(solver>=3 ? X.x.p : 0),h(v.size()),hbar(v.size()),tmpP(v.size()) {}
};
void krylov_step(const Design& X, const Vec& w, const Vec& eta, const Vec& y,
                const Vec& beta_eval, const Vec& g, const ModelOptions& o,
                const smm_krylov_options& k, smm_info& info, bool& breakdown,
                KrylovWorkspace& ws, Vec& result) {
    SMM_SCOPE(3);
#ifdef EIGEN_RUNTIME_NO_MALLOC
    NoEigenAllocation allocation_check;
#endif
    auto& A=ws.A; A.refresh(w,o.ridge,k.jacobi);
    auto& rhs=ws.rhs; auto& x=ws.x;
    rhs=-(A.invscale.array()*g.array()).matrix();
    double initial=rhs.norm(), target=k.atol+k.rtol*initial;
    x.setZero();
    breakdown=false;
    auto finish=[&]() { result.array()=A.invscale.array()*x.array(); };
    if (initial<=target) return finish();
    // CG and CR solve the symmetrically scaled normal equations directly.
    if (k.solver==0 || k.solver==2) {
        auto& r=ws.r; auto& p=ws.p; auto& Hp=ws.Hp;
        r=rhs; p=r; A.normal_into(p,Hp);
        double gamma=k.solver==0 ? r.squaredNorm() : r.dot(Hp);
        for (int64_t it=0;it<o.inner_maxiter;++it) {
            double denom=k.solver==0 ? p.dot(Hp) : Hp.squaredNorm();
            if (!(denom>0) || !(gamma>0) || !std::isfinite(denom) || !std::isfinite(gamma)) {
                breakdown=true; break;
            }
            double a=gamma/denom;
            x+=a*p; r-=a*Hp; ++info.inner_iterations;
            if (r.norm()<=target) break;
            auto& Hr=ws.Hr;
            if (k.solver==2) A.normal_into(r,Hr);
            double next=k.solver==0 ? r.squaredNorm() : r.dot(Hr);
            p=r+(next/gamma)*p;
            if (k.solver==0) A.normal_into(p,Hp);
            else Hp=Hr+(next/gamma)*Hp;
            gamma=next;
        }
        return finish();
    }
    // A = [sqrt(W) X; sqrt(ridge) P] D^{-1/2}, with intercept penalty mask P.
    // The augmented RHS includes -sqrt(ridge)*P*beta_eval so that every method
    // solves the same penalized Newton correction, not an unpenalized RHS.
    auto& b=ws.b;
    for (int64_t i=0;i<X.x.n;++i) {
        b[i]=-family_point(eta[i],y[i],o,i).score/A.sqrtw[i];
    }
    if (o.ridge>0) {
        b.tail(X.x.p)=-A.damping*beta_eval;
        if (X.unpenalized) b[X.x.n]=0.;
    }
    if (k.solver==1) { // CGLS: maintain the data-space residual.
        auto& residual=b; auto& r=ws.r; auto& p=ws.p; auto& q=ws.q;
        A.at_into(residual,r); p=r;
        double gamma=r.squaredNorm();
        for (int64_t it=0;it<o.inner_maxiter;++it) {
            A.av_into(p,q); double denom=q.squaredNorm();
            if (!(denom>0) || !std::isfinite(denom)) { breakdown=true; break; }
            double a=gamma/denom;
            x+=a*p; residual-=a*q;
            A.at_into(residual,r); ++info.inner_iterations;
            double next=r.squaredNorm();
            if (std::sqrt(next)<=target) break;
            p=r+(next/gamma)*p; gamma=next;
        }
        return finish();
    }
    auto& u=b; double beta=u.norm();
    if (beta==0) return finish();
    u/=beta;
    auto& v=ws.v; A.at_into(u,v); double alpha=v.norm();
    if (alpha==0) return finish();
    v/=alpha;
    auto& h=ws.h; auto& hbar=ws.hbar; h=v; hbar.setZero();
    double rhobar=alpha, phibar=beta;
    double zetabar=alpha*beta, alphabar=alpha, rho=1., rbar=1., cbar=1., sbar=0.;
    for (int64_t it=0;it<o.inner_maxiter;++it) {
        A.av_into(v,ws.q); u=ws.q-alpha*u; beta=u.norm();
        if (beta>0) u/=beta;
        A.at_into(u,ws.tmpP); v=ws.tmpP-beta*v; alpha=v.norm();
        if (alpha>0) v/=alpha;
        double normal_norm;
        if (k.solver==3) { // LSQR
            rho=std::hypot(rhobar,beta);
            if (!(rho>0) || !std::isfinite(rho)) { breakdown=true; break; }
            double c=rhobar/rho, s=beta/rho, theta=s*alpha;
            rhobar=-c*alpha;
            double phi=c*phibar; phibar=s*phibar;
            x+=(phi/rho)*h; h=v-(theta/rho)*h;
            normal_norm=alpha*std::abs(s*phi);
        } else { // LSMR
            double oldrho=rho, oldrbar=rbar;
            rho=std::hypot(alphabar,beta);
            if (!(rho>0) || !std::isfinite(rho)) { breakdown=true; break; }
            double c=alphabar/rho, s=beta/rho, theta=s*alpha;
            alphabar=c*alpha;
            double thetabar=sbar*rho, temp=cbar*rho;
            rbar=std::hypot(temp,theta);
            if (!(rbar>0) || !std::isfinite(rbar)) { breakdown=true; break; }
            cbar=temp/rbar; sbar=theta/rbar;
            double zeta=cbar*zetabar; zetabar=-sbar*zetabar;
            hbar=h-(thetabar*rho/(oldrho*oldrbar))*hbar;
            x+=(zeta/(rho*rbar))*hbar;
            h=v-(theta/rho)*h;
            normal_norm=std::abs(zetabar);
        }
        ++info.inner_iterations;
        if (!x.allFinite() || !std::isfinite(normal_norm)) { breakdown=true; break; }
        if (normal_norm<=target) break;
    }
    return finish();
}
