#include <Rcpp.h>
#include "spectralmm/c_api.h"
#include <cmath>
#include <limits>
#include <vector>
using namespace Rcpp;
// [[Rcpp::export]]
List smm_fit_bridge(SEXP input, NumericVector y, int family, List options, List family_options,
                    Nullable<NumericVector> beta0 = R_NilValue, int solver = -1,
                    bool jacobi = true, double krylov_rtol = 1e-4, double krylov_atol = 1e-8,
                    bool intercept = true, bool penalize_intercept = false,
                    int inference = 1, int inference_max_p = 50, int cov_type = 0,
                    double dispersion = NA_REAL, bool capture_trace = false) {
    if (smm_abi_version() != 1) stop("native ABI mismatch");
    smm_matrix x{};
    NumericVector values;
    std::vector<int64_t> indptr, indices;
    if (Rf_isS4(input)) {
        S4 a(input);
        if (!a.is("dgCMatrix")) stop("sparse X must be a dgCMatrix");
        IntegerVector dims=a.slot("Dim"), p=a.slot("p"), i=a.slot("i");
        values=a.slot("x");
        x.n=dims[0]; x.p=dims[1]; x.nnz=values.size(); x.storage=1;
        if (p.size()!=x.p+1 || i.size()!=x.nnz) stop("invalid CSC array lengths");
        indptr.assign(p.begin(),p.end()); indices.assign(i.begin(),i.end());
        x.indptr=indptr.data(); x.indices=indices.data();
    } else {
        if (!Rf_isMatrix(input) || TYPEOF(input)!=REALSXP) stop("X must be a numeric matrix");
        NumericMatrix a(input);
        values=NumericVector(input); x.n=a.nrow(); x.p=a.ncol();
    }
    x.values=values.begin();
    if (y.size()!=x.n) stop("y must have length n");
    NumericVector initial;
    const double *b0=nullptr;
    if (beta0.isNotNull()) {
        initial=NumericVector(beta0);
        if (initial.size()!=x.p+intercept) stop("beta0 must have length p");
        b0=initial.begin();
    }
    smm_options o; smm_default_options(&o); o.family=family;
    smm_stop_options stops; smm_default_stop_options(&stops);
    smm_family_options fp; smm_default_family_options(&fp);
    NumericVector trials;
    CharacterVector fn = family_options.size() ? CharacterVector(family_options.names()) : CharacterVector(0);
    if(fn.size()!=family_options.size()) stop("family_options must be named");
    for(R_xlen_t j=0;j<family_options.size();++j) {
        std::string key=as<std::string>(fn[j]);
        if(key=="trials" && family==13) {
            trials=as<NumericVector>(family_options[j]); fp.trials=trials.begin(); fp.trials_count=trials.size();
        } else {
            double v=as<double>(family_options[j]);
            if(key=="theta" && family==5) fp.theta=v;
            else if(key=="tau" && (family==6 || family==7)) fp.tau=v;
            else if(key=="smoothing" && family==6) fp.smoothing=v;
            else if(key=="delta" && family==8) fp.delta=v;
            else if(key=="nu" && family==9) fp.nu=v;
            else if(key=="sigma" && family==9) fp.sigma=v;
            else if(key=="power" && family==12) fp.power=v;
            else stop("unknown parameter for family: %s",key.c_str());
        }
    }
    CharacterVector names = options.size() ? CharacterVector(options.names()) : CharacterVector(0);
    if (options.size() && names.size()!=options.size()) stop("options must be named");
    for (R_xlen_t j=0;j<options.size();++j) {
        std::string key=as<std::string>(names[j]);
        double v=as<double>(options[j]);
        if (!std::isfinite(v)) stop("options must be finite");
        if (key=="negligible_step_tol") stops.negligible_step_tol=v;
        else if (key=="step_reltol") stops.step_reltol=v;
        else if (key=="stalled_relgtol") stops.stalled_relgtol=v;
        else if (key=="ridge") o.ridge=v;
        else if (key=="floor") o.floor=v;
        else if (key=="gtol") o.gtol=v;
        else if (key=="relgtol") o.relgtol=v;
        else if (key=="eta_max") o.eta_max=v;
        else if (key=="correction_tol") o.correction_tol=v;
        else if (key=="resid_tol") o.resid_tol=v;
        else {
            if (v!=std::floor(v) || std::abs(v)>2147483647.) stop("integer option out of range");
            if (key=="accept_negligible" || key=="accept_stalled") {
                if (v!=0 && v!=1) stop("stop flags must be boolean");
                if (key=="accept_negligible") stops.accept_negligible=int32_t(v);
                else stops.accept_stalled=int32_t(v);
            }
            else if (key=="rank") o.rank=static_cast<int64_t>(v);
            else if (key=="maxiter") o.maxiter=static_cast<int64_t>(v);
            else if (key=="inner_maxiter") o.inner_maxiter=static_cast<int64_t>(v);
            else if (key=="krylovdim") o.krylovdim=static_cast<int64_t>(v);
            else if (key=="nesterov") o.nesterov=static_cast<int32_t>(v);
            else stop("unknown option: %s",key.c_str());
        }
    }
    NumericVector coef(x.p+intercept);
    smm_info info{}; char error[1024];
    smm_krylov_options k{solver, int32_t(jacobi), krylov_rtol, krylov_atol};
    if (o.maxiter<=0) stop("maxiter must be positive");
    std::vector<smm_trace_detail> trace(capture_trace ? o.maxiter+1 : 0);
    int64_t trace_size=0;
    std::vector<double> likelihood(trace.size());
    int code=smm_fit_logged_stats(&x,y.begin(),b0,&o,&fp,solver==-1 ? nullptr : &k,&stops,
        coef.begin(),&info,capture_trace ? trace.data() : nullptr,trace.size(),&trace_size,error,sizeof(error),intercept,penalize_intercept,capture_trace ? likelihood.data() : nullptr,dispersion);
    const char *reasons[]={"gradient","maxiter","line_search_failed","inner_breakdown","negligible_step","stalled_step"};
    if (code) stop("%s",error);
    List result=List::create(_["coef"]=coef, _["family_id"]=family,
        _["info"]=List::create(_["gradient_converged"]=(info.gradnorm<=o.gtol || info.relgradnorm<=o.relgtol),
        _["termination_reason"]=reasons[info.termination], _["converged"]=bool(info.converged), _["termination"]=info.termination,
        _["iterations"]=double(info.iterations), _["inner_iterations"]=double(info.inner_iterations),
        _["restarts"]=double(info.restarts), _["corrections"]=double(info.corrections),
        _["loss"]=info.loss, _["gradnorm"]=info.gradnorm, _["relgradnorm"]=info.relgradnorm,
        _["eigresidual"]=info.eigresidual));
    if (capture_trace) {
        NumericVector iteration(trace_size), total_inner(trace_size), restarts(trace_size), corrections(trace_size), inner(trace_size),
            ll(trace_size), loss(trace_size), grad(trace_size), relgrad(trace_size), inner_res(trace_size), eigres(trace_size), step(trace_size);
        CharacterVector spectrum(trace_size);
        const char *states[]={"-","initial","reuse","correct","restart","correct-fail","restart+correct","restart+fail"};
        for(int64_t i=0;i<trace_size;++i) {
            const auto &r=trace[i];
            iteration[i]=r.iteration; total_inner[i]=r.inner_iterations; restarts[i]=r.restarts; corrections[i]=r.corrections;
            ll[i]=likelihood[i]; inner[i]=r.inner; loss[i]=r.loss; grad[i]=r.gradnorm; relgrad[i]=r.relgradnorm;
            inner_res[i]=r.inner_residual; eigres[i]=r.eigresidual; step[i]=r.step; spectrum[i]=states[r.spectrum];
        }
        result["trace"]=DataFrame::create(_["iteration"]=iteration,_["inner_iterations"]=total_inner,
            _["restarts"]=restarts,_["corrections"]=corrections,_["inner"]=inner,
            _["loglikelihood"]=ll,_["loss"]=loss,_["gradnorm"]=grad,_["relgradnorm"]=relgrad,
            _["inner_residual"]=inner_res,_["eigresidual"]=eigres,_["step"]=step,_["spectrum"]=spectrum);
    }
    std::string status="unavailable", reason;
    if (!inference) { status="disabled"; reason="inference=FALSE"; }
    else if (inference==1 && coef.size()>inference_max_p) { status="skipped"; reason="parameter count exceeds inference_max_p"; }
    else if (o.ridge!=0.) reason="ordinary Wald inference is unavailable for ridge-penalized fits";
    else if (!(info.gradnorm<=o.gtol || info.relgradnorm<=o.relgtol)) reason="fit did not satisfy the gradient convergence criterion";
    else if (x.n<=coef.size()) reason="inference requires n > number of parameters";
    else {
        NumericMatrix covariance(coef.size(),coef.size()); smm_inference_info meta{};
        int infer_code=smm_infer(&x,y.begin(),coef.begin(),&o,&fp,intercept,cov_type,
            dispersion,covariance.begin(),&meta,error,sizeof(error));
        if (!infer_code) {
            result["inference"]=List::create(_["status"]="ok",_["reason"]="",_["covariance"]=covariance,
                _["cov_type"]=meta.cov_type==1 ? "model" : "sandwich",_["estimated_dispersion"]=bool(meta.estimated_dispersion),
                _["df_resid"]=meta.df_resid,_["dispersion"]=meta.dispersion,_["rcond"]=meta.rcond);
            return result;
        }
        reason=error;
    }
    result["inference"]=List::create(_["status"]=status,_["reason"]=reason);
    return result;

}
