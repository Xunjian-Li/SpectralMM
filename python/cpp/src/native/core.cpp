#include "spectralmm/core.hpp"
#include "profile.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <memory>

namespace spectralmm {
namespace {
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int64_t>;
constexpr double eps = std::numeric_limits<double>::epsilon();
void require(bool ok, const char *message) {
    if (!ok) throw std::invalid_argument(message);
}
struct Design {
    smm_matrix x; // Logical dimensions, including an implicit intercept.
    const smm_matrix& raw;
    bool intercept, unpenalized;
    explicit Design(const smm_matrix& a, bool add=false, bool penalize=true)
        : x(a), raw(a), intercept(add), unpenalized(add && !penalize) {
        require(x.n > 0 && x.p >= (intercept ? 0 : 1), "require n > 0 and at least one parameter");
        require(x.storage == 0 || x.storage == 1, "unknown matrix storage");
        require(x.n <= std::numeric_limits<Eigen::Index>::max() / std::max<int64_t>(x.p,1) / 8,
                "matrix dimensions overflow address space");
        if (!x.storage) {
            require(x.p==0 || x.values, "missing dense values");
            for (int64_t i = 0; i < x.n*x.p; ++i)
                require(std::isfinite(x.values[i]), "X must be finite");
        } else {
            require(x.nnz >= 0 && x.indptr, "invalid CSC metadata");
            require(x.nnz == 0 || (x.values && x.indices), "missing CSC arrays");
            require(x.indptr[0] == 0 && x.indptr[x.p] == x.nnz,
                    "CSC indptr must span [0, nnz]");
            for (int64_t j = 0; j < x.p; ++j) {
                require(x.indptr[j] >= 0 && x.indptr[j] <= x.indptr[j+1] &&
                        x.indptr[j+1] <= x.nnz, "invalid CSC indptr");
                int64_t prev = -1;
                for (int64_t k = x.indptr[j]; k < x.indptr[j+1]; ++k) {
                    require(x.indices[k] > prev && x.indices[k] < x.n,
                            "CSC rows must be sorted, unique, zero-based and in range");
                    require(std::isfinite(x.values[k]), "X must be finite");
                    prev = x.indices[k];
                }
            }
        }
        x.p += intercept;
    }
    Eigen::Map<const Mat> dense() const { return {raw.values, raw.n, raw.p}; }
    Eigen::Map<const Sparse> sparse() const {
        return {raw.n, raw.p, raw.nnz, raw.indptr, raw.indices, raw.values};
    }
    void mv_into(const Vec& v, Vec& out) const {
        SMM_PRODUCT(0); SMM_SCOPE(4);
        if (raw.storage) out.noalias()=sparse()*v.tail(raw.p);
        else out.noalias()=dense()*v.tail(raw.p);
        if (intercept) out.array()+=v[0];
    }
    void tmv_into(const Vec& v, Vec& out) const {
        SMM_PRODUCT(1); SMM_SCOPE(4);
        if (raw.storage) out.tail(raw.p).noalias()=sparse().transpose()*v;
        else out.tail(raw.p).noalias()=dense().transpose()*v;
        if (intercept) out[0]=v.sum();
    }
    Vec mv(const Vec& v) const { Vec out(x.n); mv_into(v,out); return out; }
    void mm_into(const Mat& v, Mat& out) const {
        SMM_PRODUCT(2); SMM_SCOPE(4);
        if (raw.storage) out.noalias()=sparse()*v.bottomRows(raw.p);
        else out.noalias()=dense()*v.bottomRows(raw.p);
        if (intercept) out.rowwise()+=v.row(0);
    }
    double diagonal(const Vec& w, int64_t j) const {
        if (intercept && j==0) return w.sum();
        j-=intercept;
        double d=0.;
        if (raw.storage) {
            for (int64_t q=raw.indptr[j];q<raw.indptr[j+1];++q)
                d+=w[raw.indices[q]]*raw.values[q]*raw.values[q];
        } else d=(w.array()*dense().col(j).array().square()).sum();
        return d;
    }

    Vec initial(double eta) const {
        Vec b = Vec::Zero(x.p);
        if (intercept) { b[0]=eta; return b; }
        for (int64_t j = 0; j < x.p; ++j) {
            const double *v;
            if (x.storage) {
                if (x.indptr[j+1]-x.indptr[j] != x.n) continue;
                v = x.values + x.indptr[j];
            } else v = x.values + j*x.n;
            double c = v[0];
            if (std::abs(c) <= std::sqrt(eps)) continue;
            bool same = true;
            for (int64_t i = 1; i < x.n; ++i)
                if (std::abs(v[i]-c) > std::sqrt(eps)*std::max({1., std::abs(c), std::abs(v[i])})) {
                    same = false; break;
                }
            if (same) { b[j] = eta/c; break; }
        }
        return b;
    }
};
double sigmoid(double e) {
    if (e >= 0) return 1./(1.+std::exp(-e));
    double z = std::exp(e); return z/(1.+z);
}
#include "families.hpp"

double loss(const Vec& eta, const Vec& y, const Vec& b, const ModelOptions& o) {
    SMM_SCOPE(6);
    if (!eta.allFinite() || !b.allFinite()) return INFINITY;
    double value = .5*o.ridge*b.tail(b.size()-o.unpenalized).squaredNorm();
    if (!o.family) return value + .5*(eta-y).squaredNorm();
    if(o.family==11 && (eta.array()<=0).any()) return INFINITY;
    for (Eigen::Index i=0; i<y.size(); ++i)
        value += family_point(eta[i],y[i],o,i).objective;
    return value;
}
// Output-only normalized likelihood, excluding ridge. Scale defaults match wrappers.
double display_loglik(const Vec& eta,const Vec& y,const ModelOptions& o,int64_t p,double dispersion) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    if(o.family==6 || o.family==7 || o.family==8 || o.family==12) return nan;
    double kernel=0., pearson=0., constant=0.;
    for(int64_t i=0;i<y.size();++i) {
        const auto pt=family_point(eta[i],y[i],o,i);
        kernel+=pt.objective;
        if(o.family==4 || o.family==11) pearson+=std::pow((y[i]-pt.mu)/pt.mu,2);
        if(o.family==1 || o.family==2 || o.family==3 || o.family==5 || o.family==13) {
            if(y[i]!=std::floor(y[i])) return nan;
        }
        if(o.family==3) constant-=std::lgamma(y[i]+1.);
        if(o.family==5) constant+=std::lgamma(y[i]+o.model.theta)-std::lgamma(o.model.theta)-std::lgamma(y[i]+1.)+o.model.theta*std::log(o.model.theta);
        if(o.family==13) {
            double n=trials_at(o,i);
            if(n!=std::floor(n)) return nan;
            constant+=std::lgamma(n+1.)-std::lgamma(y[i]+1.)-std::lgamma(n-y[i]+1.);
        }
    }
    const double pi=std::acos(-1.);
    if(o.family==0 || o.family==10) {
        double scale=std::isnan(dispersion) ? 2.*kernel/y.size() : dispersion;
        return scale==0. ? INFINITY : -.5*y.size()*std::log(2.*pi*scale)-kernel/scale;
    }
    if(o.family==4 || o.family==11) {
        double scale=std::isnan(dispersion) ? (y.size()>p ? pearson/(y.size()-p) : nan) : dispersion;
        if(!(scale>0.) || !std::isfinite(scale)) return nan;
        double shape=1./scale;
        return -kernel/scale+(shape-1.)*y.array().log().sum()-y.size()*(std::lgamma(shape)+shape*std::log(scale));
    }
    if(o.family==9) constant=y.size()*(std::lgamma((o.model.nu+1.)/2.)-std::lgamma(o.model.nu/2.)-.5*std::log(o.model.nu*pi)-std::log(o.model.sigma));
    return constant-kernel;
}
void weights_into(const Vec& eta,const Vec& y,const ModelOptions& o,Vec& w) {
    SMM_SCOPE(7);
    for(Eigen::Index i=0;i<eta.size();++i) w[i]=family_point(eta[i],y[i],o,i).weight;
}
void gradient_into(const Design& X,const Vec& eta,const Vec& y,const Vec& b,
             const ModelOptions& o,Vec& out,Vec& residual) {
    SMM_SCOPE(5);
    for(Eigen::Index i=0;i<eta.size();++i) residual[i]=family_point(eta[i],y[i],o,i).score;
    X.tmv_into(residual,out); out.tail(b.size()-X.unpenalized)+=o.ridge*b.tail(b.size()-X.unpenalized);
}
struct Spectrum {
    Mat V, XV;
    Vec lambda, w;
    double residual = INFINITY;
    void resize(int64_t n,int64_t p,int64_t r) { V.resize(p,r); XV.resize(n,r); lambda.resize(r); w.resize(n); }
    void swap(Spectrum& other) {
        V.swap(other.V); XV.swap(other.XV); lambda.swap(other.lambda); w.swap(other.w);
        std::swap(residual,other.residual);
    }
};
struct SpectralWorkspace {
    Mat Q,HQ,T,HV,rotatedHV,small,rotation,weightedXV,change;
    Vec alpha,off,q,z,Xq,weighted;
    Spectrum candidate;
    Eigen::SelfAdjointEigenSolver<Mat> eig,smallEig;
    SpectralWorkspace(int64_t n,int64_t p,int64_t rank,const ModelOptions& o)
        : HV(p,rank+1),rotatedHV(p,rank+1),small(rank+1,rank+1),rotation(rank+1,rank+1),
          weightedXV(n,rank+1),change(rank+1,rank+1),q(p),z(p),Xq(n),weighted(n),smallEig(rank+1) {
        int64_t r=rank+1,kd=std::min(p,o.krylovdim ? o.krylovdim : std::max(3*r+20,r+10));
        for(int attempt=1;attempt<3;++attempt) kd=std::min(p,std::max(kd+20,(3*kd+1)/2));
        Q.resize(p,kd); HQ.resize(p,kd); alpha.resize(kd); off.resize(kd);
        candidate.resize(n,p,r);
    }
};
void hess_block_into(const Design& X,const Vec& w,const Mat& V,const Mat& XV,
                     double ridge,SpectralWorkspace& ws) {
    for(Eigen::Index j=0;j<V.cols();++j) {
        ws.weighted.array()=w.array()*XV.col(j).array();
        X.tmv_into(ws.weighted,ws.z);
        ws.HV.col(j)=ws.z;
        ws.HV.col(j).tail(V.rows()-X.unpenalized)+=ridge*V.col(j).tail(V.rows()-X.unpenalized);
    }
}
double spectral_residual(const Mat& HV,const Spectrum& s) {
    // Explicit scalar reduction avoids allocating the p by rank residual.
    double norm2=0.;
    for(Eigen::Index j=0;j<HV.cols();++j)
        norm2+=(HV.col(j)-s.lambda[j]*s.V.col(j)).squaredNorm();
    return std::sqrt(norm2)/std::max(HV.norm(),eps);
}
void lanczos(const Design& X,const Vec& w,int64_t r,int64_t kd,
             double ridge,uint64_t seed,SpectralWorkspace& ws) {
    auto& Q=ws.Q; auto& alpha=ws.alpha; auto& off=ws.off; auto& q=ws.q; auto& z=ws.z;
    off.head(kd).setZero();
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal;
    for(Eigen::Index i=0;i<q.size();++i) q[i]=normal(rng);
    q.normalize();
    for(int64_t j=0;j<kd;++j) {
        Q.col(j)=q;
        X.mv_into(q,ws.Xq); ws.weighted.array()=w.array()*ws.Xq.array();
        X.tmv_into(ws.weighted,z); z.tail(q.size()-X.unpenalized)+=ridge*q.tail(q.size()-X.unpenalized);
        ws.HQ.col(j)=z; // Save H*q before recurrence and reorthogonalization.
        if(j) z-=off[j-1]*Q.col(j-1);
        alpha[j]=q.dot(z); z-=alpha[j]*q;
        for(int pass=0;pass<2;++pass)
            for(int64_t k=0;k<=j;++k) z-=Q.col(k).dot(z)*Q.col(k);
        if(j+1==kd) break;
        off[j]=z.norm();
        if(off[j]>1e-12*std::max(1.,std::abs(alpha[j]))) q=z/off[j];
        else {
            off[j]=0.; bool found=false;
            for(int64_t axis=0;axis<X.x.p && !found;++axis) {
                q.setZero(); q[axis]=1.;
                for(int pass=0;pass<2;++pass)
                    for(int64_t k=0;k<=j;++k) q-=Q.col(k).dot(q)*Q.col(k);
                double norm=q.norm();
                if(norm>1e-8) { q/=norm; found=true; }
            }
            if(!found) throw std::runtime_error("Lanczos could not extend basis");
        }
    }
    ws.T=alpha.head(kd).asDiagonal();
    for(int64_t j=0;j+1<kd;++j) ws.T(j,j+1)=ws.T(j+1,j)=off[j];
    ws.eig.compute(ws.T);
    if(ws.eig.info()!=Eigen::Success) throw std::runtime_error("Lanczos eigensolve failed");
    auto& s=ws.candidate;
    s.V.noalias()=Q.leftCols(kd)*ws.eig.eigenvectors().rightCols(r).rowwise().reverse();
    s.lambda=ws.eig.eigenvalues().tail(r).reverse();
    // Preserve matrix-matrix product arithmetic; do not recompute columnwise.
    X.mm_into(s.V,s.XV); s.w=w;
    // V=Q*U, so H*V=(H*Q)*U. Reuse the products already evaluated
    // during Lanczos rather than applying X' W to each Ritz vector again.
    ws.HV.noalias()=ws.HQ.leftCols(kd)*ws.eig.eigenvectors().rightCols(r).rowwise().reverse();
    s.residual=spectral_residual(ws.HV,s);
}
void restart(Spectrum& best,const Design& X,const Vec& w,const ModelOptions& o,
             int64_t rank,uint64_t seed,SpectralWorkspace& ws) {
    SMM_SCOPE(1);
    int64_t r=rank+1,kd=std::min(X.x.p,o.krylovdim ? o.krylovdim : std::max(3*r+20,r+10));
    best.residual=INFINITY;
    for(int attempt=0;attempt<3;++attempt) {
        lanczos(X,w,r,kd,o.ridge,seed+7919*(attempt+1),ws);
        if(ws.candidate.residual<best.residual) best.swap(ws.candidate);
        if(best.residual<=o.resid_tol || kd==X.x.p) break;
        kd=std::min(X.x.p,std::max(kd+20,(3*kd+1)/2));
    }
    if(!std::isfinite(best.residual)) throw std::runtime_error("non-finite Lanczos residual");
}
bool correct(Spectrum& s,const Design& X,const Vec& w,const ModelOptions& o,SpectralWorkspace& ws) {
    SMM_SCOPE(2);
    ws.candidate.residual=std::numeric_limits<double>::quiet_NaN();
    hess_block_into(X,w,s.V,s.XV,o.ridge,ws);
    ws.small.noalias()=s.V.transpose()*ws.HV;
    for(Eigen::Index j=0;j<ws.small.cols();++j)
        for(Eigen::Index i=0;i<j;++i) {
            double v=.5*(ws.small(i,j)+ws.small(j,i)); ws.small(i,j)=ws.small(j,i)=v;
        }
    ws.smallEig.compute(ws.small);
    if(ws.smallEig.info()!=Eigen::Success) return false;
    ws.rotation=ws.smallEig.eigenvectors().rowwise().reverse();
    auto& next=ws.candidate;
    next.V.noalias()=s.V*ws.rotation; next.XV.noalias()=s.XV*ws.rotation;
    next.lambda=ws.smallEig.eigenvalues().reverse(); next.w=w;
    // H(VR)=(HV)R: reuse HV instead of applying X' W X a second time.
    ws.rotatedHV.noalias()=ws.HV*ws.rotation;
    next.residual=spectral_residual(ws.rotatedHV,next);
    if(!std::isfinite(next.residual) || next.residual>o.resid_tol) return false;
    s.swap(next); return true;
}
// One workspace per fit, reused across outer and PCG iterations. No shared
// mutable state, so separate fits remain independent and thread-safe.
#ifdef EIGEN_RUNTIME_NO_MALLOC
struct NoEigenAllocation {
    bool previous=Eigen::internal::is_malloc_allowed();
    NoEigenAllocation() { Eigen::internal::set_is_malloc_allowed(false); }
    ~NoEigenAllocation() { Eigen::internal::set_is_malloc_allowed(previous); }
};
#endif
struct PCGWorkspace {
    Vec r,z,p,Hp,Xp,weighted,coeff,projected;
    PCGWorkspace(int64_t n, int64_t psize, int64_t rank)
        : r(psize),z(psize),p(psize),Hp(psize),Xp(n),weighted(n),coeff(rank),projected(rank) {}
};
void pcg(const Design& X, const Vec& w, const Vec& g, const Spectrum& s,
        const ModelOptions& o, int64_t rank, smm_info& info, bool& breakdown,
        PCGWorkspace& ws, Vec& delta, Vec& Xdelta, bool mm=false) {
#ifdef EIGEN_RUNTIME_NO_MALLOC
    NoEigenAllocation allocation_check;
#endif
    SMM_SCOPE(3);
    auto& r=ws.r; auto& z=ws.z; auto& p=ws.p; auto& Hp=ws.Hp;
    double base=1./(std::max(0.,s.lambda[rank])+o.floor);
    for (int64_t j=0;j<rank;++j)
        ws.coeff[j]=1./(std::max(0.,s.lambda[j])+o.floor)-base;
    auto precondition_into=[&]() {
        ws.projected.noalias()=s.V.leftCols(rank).transpose()*r;
        ws.projected.array()*=ws.coeff.array();
        z.noalias()=s.V.leftCols(rank)*ws.projected;
        z+=base*r;
    };
    delta.setZero(); Xdelta.setZero(); r=-g; precondition_into(); p=z;
    double rz=r.dot(z), initial=g.norm();
    breakdown=false;
    if (initial<=1e-10) return;
    for (int64_t i=0;i<o.inner_maxiter;++i) {
        X.mv_into(p,ws.Xp);
        ws.weighted.array()=w.array()*ws.Xp.array();
        X.tmv_into(ws.weighted,Hp); Hp.tail(p.size()-X.unpenalized)+=o.ridge*p.tail(p.size()-X.unpenalized);
        double pHp=p.dot(Hp);
        if (!(rz>0.) || !(pHp>0.) || !std::isfinite(rz) || !std::isfinite(pHp)) {
            breakdown=true; break;
        }
        double a=rz/pHp;
        if (mm) a=std::clamp(a,1e-10,10.);
        if (!std::isfinite(a)) { breakdown=true; break; }
        delta+=a*p; Xdelta+=a*ws.Xp; r-=a*Hp; ++info.inner_iterations;
        if (!delta.allFinite() || !r.allFinite()) { breakdown=true; break; }
        if (r.norm()<=std::max(1e-10,o.eta_max*initial)) break;
        precondition_into();
        double next=r.dot(z);
        if (mm) p=z;
        else p=z+(next/rz)*p;
        rz=next;
    }
}

#include "krylov_impl.hpp"
#include "inference.hpp"
} // namespace

void fit(const smm_matrix& input, const double *yp, const double *beta0,
         const smm_options& options, double *out, smm_info& info, const smm_krylov_options *krylov, const smm_stop_options *stop,
         smm_trace_entry *trace, int64_t trace_capacity, int64_t *trace_size, const smm_family_options *family, bool intercept, bool penalize_intercept, smm_trace_detail *detail, double *likelihood, double dispersion) {
    ModelOptions o(options,family);
    SMM_RESET();
    SMM_SCOPE(0);
    Design X(input,intercept,penalize_intercept);
    o.unpenalized=X.unpenalized;
    const int64_t p=X.x.p;
    if (trace || detail) require(trace_size && trace_capacity>o.maxiter, "trace capacity must exceed maxiter");
    if (trace_size) *trace_size=0;
    if (stop) {
        require((stop->accept_negligible==0 || stop->accept_negligible==1) &&
                (stop->accept_stalled==0 || stop->accept_stalled==1), "invalid stop flag");
        require(std::isfinite(stop->negligible_step_tol) && stop->negligible_step_tol>=0 &&
                std::isfinite(stop->step_reltol) && stop->step_reltol>=0 &&
                std::isfinite(stop->stalled_relgtol) && stop->stalled_relgtol>=0, "invalid step tolerance");
    }
    if (krylov) {
        require(krylov->solver>=0 && krylov->solver<=6, "invalid solver");
        require(krylov->jacobi==0 || krylov->jacobi==1, "jacobi must be 0 or 1");
        require(std::isfinite(krylov->rtol) && krylov->rtol>0 && krylov->rtol<1 &&
                std::isfinite(krylov->atol) && krylov->atol>=0, "invalid Krylov tolerances");
    }
    require(o.family>=0 && o.family<=13, "unsupported family");
    require(o.nesterov == 0 || o.nesterov == 1, "nesterov must be 0 or 1");
    int64_t rank = o.rank ? o.rank : (p<20 ? p-1 : 10);
    require((p==1 && rank==0) || (rank>=1 && rank<p), "rank must satisfy 1 <= rank < p");
    require(o.maxiter>0 && o.inner_maxiter>0, "iteration limits must be positive");
    require(o.krylovdim == 0 || (o.krylovdim>=rank+1 && o.krylovdim<=p),
            "krylovdim must be zero (automatic) or in [rank+1,p]");
    require(std::isfinite(o.ridge) && o.ridge>=0., "ridge must be finite and nonnegative");
    require(std::isfinite(o.floor) && o.floor>0., "floor must be finite and positive");
    require(std::isfinite(o.gtol) && o.gtol>0. && std::isfinite(o.relgtol) && o.relgtol>0.,
            "gradient tolerances must be finite and positive");
    require(std::isfinite(o.eta_max) && o.eta_max>0. && o.eta_max<1., "eta_max must be in (0,1)");
    require(std::isfinite(o.correction_tol) && o.correction_tol>=0. &&
            std::isfinite(o.resid_tol) && o.resid_tol>0., "invalid spectral tolerances");
    Vec y = Eigen::Map<const Vec>(yp,input.n);
    require(y.allFinite(), "y must be finite");
    validate_family(o,y);
    double eta0=initial_eta(y,o);
    Vec b=beta0 ? Vec(Eigen::Map<const Vec>(beta0,p)) : X.initial(eta0);
    require(b.allFinite(), "beta0 must be finite");
    Vec prev=b, eta=X.mv(b), g(p),gradres(input.n),w(input.n),nw(input.n);
    gradient_into(X,eta,y,b,o,g,gradres);
    bool base_gradient_valid=true;
    double scale=1.+g.norm(), f=loss(eta,y,b,o), t=1.;
    if (!std::isfinite(f) || !g.allFinite()) throw std::runtime_error("non-finite initial objective or gradient");
    const bool spectral_mode=!krylov || krylov->solver==6;
    const bool cholesky_mode=krylov && krylov->solver==5;
    Spectrum spec; bool need_restart=true;
    std::unique_ptr<SpectralWorkspace> spectralws;
    if (spectral_mode) {
        spec.resize(input.n,p,rank+1);
        spectralws=std::make_unique<SpectralWorkspace>(input.n,p,rank,o);
    }
    PCGWorkspace pcgws(spectral_mode ? input.n : 0,spectral_mode ? p : 0,spectral_mode ? rank : 0);
    std::unique_ptr<KrylovWorkspace> krylovws;
    if (!spectral_mode && !cholesky_mode) krylovws=std::make_unique<KrylovWorkspace>(X,o.ridge,krylov->solver);
    Vec eval(p),ee(input.n),step(p),dEta(input.n),trial(p),trialEta(input.n);
    info={}; info.termination=1; info.eigresidual=std::numeric_limits<double>::quiet_NaN();
    const double missing=std::numeric_limits<double>::quiet_NaN();
    int64_t row_inner=-1;
    double row_residual=missing, row_eig=missing, row_step=missing;
    int32_t row_spectrum=spectral_mode ? 1 : 0;
    Vec trace_product(detail ? input.n : 0), trace_residual(detail ? p : 0);
    auto record = [&]() {
        if (!trace && !detail) return;
        int64_t slot=*trace_size;
        if (slot && (trace ? trace[slot-1].iteration : detail[slot-1].iteration)==info.iterations) --slot;
        else ++*trace_size;
        if (likelihood) likelihood[slot]=display_loglik(eta,y,o,p,dispersion);
        if (trace) trace[slot]={info.iterations,info.inner_iterations,info.restarts,info.corrections,
                               f,g.norm(),g.norm()/scale};
        if (detail) detail[slot]={info.iterations,info.inner_iterations,info.restarts,info.corrections,
            row_inner,f,g.norm(),g.norm()/scale,row_residual,row_eig,row_step,row_spectrum};
    };
    for (int64_t iter=0; iter<o.maxiter; ++iter) {
        if (!base_gradient_valid) gradient_into(X,eta,y,b,o,g,gradres);
        base_gradient_valid=true;
        if (!g.allFinite()) throw std::runtime_error("non-finite gradient");
        record();
        double base_relgrad=g.norm()/scale;
        if (g.norm()<=o.gtol || g.norm()/scale<=o.relgtol) {
            info.converged=1; info.termination=0; break;
        }
        eval=b; ee=eta;
        if (o.nesterov && iter>0) {
            eval=b+((t-1.)/(t+1.))*(b-prev); X.mv_into(eval,ee);
            if (loss(ee,y,eval,o)>f) { eval=b; ee=eta; t=1.; }
            else { gradient_into(X,ee,y,eval,o,g,gradres); base_gradient_valid=false; }
        }
        weights_into(ee,y,o,w);
        const bool restarted=spectral_mode && need_restart;
        if (restarted) {
            restart(spec,X,w,o,rank,static_cast<uint64_t>(iter+1),*spectralws);
            ++info.restarts; need_restart=false;
            if (detail && iter==0) { detail[0].eigresidual=spec.residual; detail[0].restarts=info.restarts; }
        }
        const double restart_residual=restarted ? spec.residual : missing;
        bool breakdown=false;
        const int64_t inner_before=info.inner_iterations;
        if (cholesky_mode) {
            Mat H=weighted_information(X,w);
            H.diagonal().tail(p-X.unpenalized).array()+=o.ridge;
            Eigen::LLT<Mat> factor;
            for (int attempt=0;attempt<8;++attempt) {
                Mat candidate=H;
                if (attempt) candidate.diagonal().array()+=1e-10*std::pow(10.,attempt-1);
                factor.compute(candidate);
                if (factor.info()==Eigen::Success) break;
            }
            if (factor.info()!=Eigen::Success) { info.termination=3; break; }
            step=factor.solve(-g);
            if (!step.allFinite()) { info.termination=3; break; }
            ++info.inner_iterations;
            X.mv_into(step,dEta);
        } else if (!spectral_mode) {
            krylov_step(X,w,ee,y,eval,g,o,*krylov,info,breakdown,*krylovws,step);
            X.mv_into(step,dEta);
        } else pcg(X,w,g,spec,o,rank,info,breakdown,pcgws,step,dEta,krylov && krylov->solver==6);
        double inner_residual=missing;
        if (detail && step.allFinite()) {
            trace_product.array()=w.array()*dEta.array();
            X.tmv_into(trace_product,trace_residual);
            trace_residual.tail(p-X.unpenalized)+=o.ridge*step.tail(p-X.unpenalized);
            trace_residual+=g;
            inner_residual=trace_residual.norm()/std::max(g.norm(),std::numeric_limits<double>::min());
        }
        double dnorm=step.norm();
        // Match Julia spectral_mm: a failed inner solve must be retried before
        // checking negligible steps, otherwise a zero failed step looks converged.
        if (spectral_mode && stop && breakdown &&
            (info.inner_iterations==inner_before || !std::isfinite(dnorm))) {
            need_restart=true; t=1.; continue;
        }
        if (step.allFinite() && stop && stop->accept_negligible &&
            dnorm<=stop->negligible_step_tol*(1.+b.norm())) {
            info.converged=1; info.termination=4; break;
        }
        if (!step.allFinite() || !(g.dot(step)<0.)) {
            info.termination=3; break;
        }
        double a=1., nf=INFINITY;
        for (; a>=1e-10; a*=.5) {
            trial=eval+a*step; trialEta=ee+a*dEta;
            nf=loss(trialEta,y,trial,o);
            // Account for roundoff in summed objectives near a stationary point.
            if (std::isfinite(nf) && nf<=f+8.*eps*std::max(1.,std::abs(f))) break;
        }
        if (a<1e-10) {
            if (spectral_mode && stop) { need_restart=true; t=1.; continue; }
            info.termination=2; break;
        }
        row_inner=info.inner_iterations-inner_before;
        row_residual=inner_residual; row_step=a;
        row_spectrum=spectral_mode ? (restarted && iter>0 ? 4 : 2) : 0;
        row_eig=restarted && iter>0 ? restart_residual : missing;
        if (stop && stop->accept_stalled && base_relgrad<=stop->stalled_relgtol &&
            a*dnorm<=stop->step_reltol*(1.+b.norm())) {
            b.swap(trial); eta.swap(trialEta); f=nf; ++info.iterations;
            base_gradient_valid=false;
            info.converged=1; info.termination=5; break;
        }
        bool correction_failed=false;
        if (spectral_mode) {
            weights_into(trialEta,y,o,nw);
            auto& sw=*spectralws;
            sw.weightedXV=(nw-spec.w).asDiagonal()*spec.XV;
            sw.change.noalias()=spec.XV.transpose()*sw.weightedXV;
            double relative=sw.change.norm()/std::max(spec.lambda.norm(),std::sqrt(eps));
            if (!std::isfinite(relative) || relative>o.correction_tol) {
                if (correct(spec,X,nw,o,*spectralws)) {
                    ++info.corrections;
                    row_spectrum=restarted && iter>0 ? 6 : 3;
                    row_eig=spec.residual;
                } else {
                    correction_failed=true; need_restart=true;
                    row_spectrum=restarted && iter>0 ? 7 : 5;
                    row_eig=spectralws->candidate.residual;
                }
            }
        }
        if (breakdown) need_restart=true;
        prev=b; b.swap(trial); eta.swap(trialEta); f=nf;
        base_gradient_valid=false;
        ++info.iterations;
        if (o.nesterov) t=correction_failed ? 1. : .5*(1.+std::sqrt(1.+4.*t*t));
    }
    if (!base_gradient_valid) gradient_into(X,eta,y,b,o,g,gradres);
    record();
    info.loss=loss(eta,y,b,o); info.gradnorm=g.norm(); info.relgradnorm=g.norm()/scale;
    info.eigresidual=info.restarts ? spec.residual : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(info.loss) || !std::isfinite(info.gradnorm))
        throw std::runtime_error("non-finite final objective or gradient");
    if (info.gradnorm<=o.gtol || info.relgradnorm<=o.relgtol) {
        info.converged=1; info.termination=0;
    }
    Eigen::Map<Vec>(out,p)=b;
}

void infer(const smm_matrix& input, const double* yp, const double* bp,
           const smm_options& options, const smm_family_options* family,
           bool intercept, int cov_type, double dispersion, double* covariance,
           smm_inference_info& info) {
    require(options.ridge==0., "ordinary Wald inference is unavailable for ridge-penalized fits");
    require(cov_type>=0 && cov_type<=2, "invalid covariance type");
    require(std::isnan(dispersion) || (std::isfinite(dispersion) && dispersion>0.), "dispersion must be positive");
    Design X(input,intercept,false);
    const int64_t n=X.x.n, p=X.x.p;
    require(n>p, "inference requires n > number of parameters");
    Vec y=Eigen::Map<const Vec>(yp,n), beta=Eigen::Map<const Vec>(bp,p);
    require(y.allFinite() && beta.allFinite(), "inference inputs must be finite");
    ModelOptions o(options,family); validate_family(o,y);
    const bool residual=o.family>=6 && o.family<=9;
    if (!cov_type) cov_type=residual ? 2 : 1;
    require(!residual || cov_type==2, "residual models require sandwich covariance");
    require(cov_type!=2 || std::isnan(dispersion), "dispersion is not used by sandwich covariance");
    Vec eta=X.mv(beta), w(n), score2(n);
    require(eta.allFinite() && (o.family!=11 || (eta.array()>0).all()), "invalid fitted predictor for inference");
    bool separated=true, positive_margin=false;
    double pearson=0.;
    const bool estimate_scale=o.family==0 || o.family==4 || o.family==10 || o.family==11 || o.family==12;
    for (int64_t i=0;i<n;++i) {
        auto point=family_point(eta[i],y[i],o,i);
        double mu=point.mu, r=y[i]-eta[i], v=1., derivative=1.;
        score2[i]=point.score*point.score;
        if (!residual) {
            switch (o.family) {
            case 1: case 2: v=mu*(1.-mu); derivative=o.family==1 ? v : std::exp(-.5*eta[i]*eta[i])/std::sqrt(2.*std::acos(-1.)); break;
            case 3: v=mu; derivative=mu; break;
            case 4: v=mu*mu; derivative=mu; break;
            case 5: v=mu+mu*mu/o.model.theta; derivative=mu; break;
            case 10: derivative=mu; break;
            case 11: v=mu*mu; derivative=-v; break;
            case 12: v=std::pow(mu,o.model.power); derivative=mu; break;
            case 13: v=mu*(1.-mu/trials_at(o,i)); derivative=v; break;
            }
            require(v>0. && std::isfinite(v), "degenerate fitted variance");
            w[i]=o.family==2 ? probit_information(eta[i]) : derivative*derivative/v; // No optimizer weight floor in inference.
            if (cov_type==2) {
                // Sandwich bread uses the derivative of the actual score.
                if (o.family==2) { double h=std::abs(point.score), z=y[i]==1. ? eta[i] : -eta[i]; w[i]=h*(h+z); }
                else if (o.family==4) w[i]=y[i]/mu;
                else if (o.family==5) {
                    double t=o.model.theta; w[i]=t*mu*(t+y[i])/((t+mu)*(t+mu));
                } else if (o.family==10) w[i]=mu*(2.*mu-y[i]);
                else if (o.family==12) {
                    double a=o.model.power;
                    w[i]=(2.-a)*std::pow(mu,2.-a)+(a-1.)*y[i]*std::pow(mu,1.-a);
                }
            }
            if (estimate_scale) pearson+=(y[i]-mu)*(y[i]-mu)/v;
        } else if (o.family==6) {
            const double a=o.model.smoothing, t=r*r+a*a;
            w[i]=a*a/(2.*std::pow(t,1.5));
        } else if (o.family==7) w[i]=r>=0 ? o.model.tau : 1.-o.model.tau;
        else if (o.family==8) w[i]=std::pow(1.+r*r/(o.model.delta*o.model.delta),-1.5);
        else {
            const double c=o.model.nu*o.model.sigma*o.model.sigma;
            // Derivative of the Student-t score, NOT its positive MM weights.
            w[i]=(o.model.nu+1.)*(c-r*r)/((c+r*r)*(c+r*r));
        }
        if (o.family==1 || o.family==2 || o.family==13) {
            const double trials=o.family==13 ? trials_at(o,i) : 1.;
            if (y[i]!=0. && y[i]!=trials) separated=false;
            const double margin=(y[i]==0. ? -eta[i] : eta[i]);
            separated=separated && margin>=0.; positive_margin=positive_margin || margin>0.;
        } else separated=false;
    }
    require(!(separated && positive_margin), "separation detected; ordinary Wald inference is not reliable");
    require(w.allFinite() && score2.allFinite(), "non-finite inference weights");
    Mat H=weighted_information(X,w);
    require((H.diagonal().array()>0.).all(), "information matrix is not positive definite");
    Vec scale=H.diagonal().array().sqrt().inverse();
    Mat scaled=scale.asDiagonal()*H*scale.asDiagonal();
    Eigen::LLT<Mat> chol(scaled);
    require(chol.info()==Eigen::Success, "information matrix is singular or not positive definite");
    const double rcond=chol.rcond();
    require(std::isfinite(rcond) && rcond>1e-12, "information matrix is singular or ill-conditioned");
    Mat inverse=scale.asDiagonal()*chol.solve(Mat::Identity(p,p))*scale.asDiagonal();
    const double df=double(n-p);
    const bool estimated=cov_type==1 && std::isnan(dispersion) && estimate_scale;
    const double phi=cov_type==2 ? 1. : (std::isnan(dispersion) ? (estimated ? pearson/df : 1.) : dispersion);
    require(std::isfinite(phi) && phi>0., "estimated dispersion is zero or non-finite");
    Mat cov;
    if (cov_type==1) cov=phi*inverse;
    else cov=(double(n)/df)*inverse*weighted_information(X,score2)*inverse;
    cov=.5*(cov+cov.transpose()).eval();
    require(cov.allFinite() && (cov.diagonal().array()>0.).all(), "degenerate covariance matrix");
    Eigen::Map<Mat>(covariance,p,p)=cov;
    info={cov_type,int32_t(estimated),df,phi,rcond};
}
} // namespace spectralmm
