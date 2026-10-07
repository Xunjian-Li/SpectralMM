#include "spectralmm/core.hpp"
#include <cstdio>
#include <cmath>
#include <limits>
#include <exception>
#include <stdexcept>
extern "C" {
int32_t smm_abi_version(void) { return 1; }
void smm_default_options(smm_options *o) {
    if (o) *o = {0, 1, 0, 200, 100, 0, 0., 1e-6, 1e-7, 1e-8, .5, .01, .5};
}
int32_t smm_fit(const smm_matrix *X, const double *y, const double *beta0,
                const smm_options *o, double *coef, smm_info *info,
                char *error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    if (info) *info = {};
    try {
        if (!X || !y || !o || !coef || !info)
            throw std::invalid_argument("null required argument");
        spectralmm::fit(*X, y, beta0, *o, coef, *info);
        return 0;
    } catch (const std::exception& e) {
        if (error && capacity) std::snprintf(error, capacity, "%s", e.what());
        return 1;
    } catch (...) {
        if (error && capacity) std::snprintf(error, capacity, "unknown native error");
        return 2;
    }
}
int32_t smm_fit_krylov(const smm_matrix *X, const double *y, const double *beta0,
                       const smm_options *o, const smm_krylov_options *k,
                       double *coef, smm_info *info, char *error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    if (info) *info = {};
    try {
        if (!X || !y || !o || !k || !coef || !info)
            throw std::invalid_argument("null required argument");
        spectralmm::fit(*X,y,beta0,*o,coef,*info,k);
        return 0;
    } catch (const std::exception& e) {
        if (error && capacity) std::snprintf(error,capacity,"%s",e.what());
        return 1;
    } catch (...) {
        if (error && capacity) std::snprintf(error,capacity,"unknown native error");
        return 2;
    }
}

void smm_default_stop_options(smm_stop_options *s) {
    if (s) *s = {1,1,1e-14,std::sqrt(std::numeric_limits<double>::epsilon()),1e-4};
}
int32_t smm_fit_extended(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *o, const smm_krylov_options *k, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_entry *trace, int64_t capacity,
    int64_t *trace_size, char *error, size_t error_capacity) {
    if (error && error_capacity) error[0]='\0';
    if (info) *info={};
    if (trace_size) *trace_size=0;
    try {
        if (!X || !y || !o || !coef || !info) throw std::invalid_argument("null required argument");
        smm_stop_options defaults; smm_default_stop_options(&defaults);
        spectralmm::fit(*X,y,beta0,*o,coef,*info,k,stop ? stop : &defaults,trace,capacity,trace_size);
        return 0;
    } catch (const std::exception& e) {
        if (error && error_capacity) std::snprintf(error,error_capacity,"%s",e.what());
        return 1;
    } catch (...) {
        if (error && error_capacity) std::snprintf(error,error_capacity,"unknown native error");
        return 2;
    }
}

void smm_default_family_options(smm_family_options *f) {
    if(f) *f={1.,.5,.1,1.0,4.,1.,1.5,nullptr,0};
}
int32_t smm_fit_family(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *o, const smm_family_options *family,
    const smm_krylov_options *k, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_entry *trace, int64_t capacity,
    int64_t *trace_size, char *error, size_t error_capacity) {
    if(error && error_capacity) error[0]='\0';
    if(info) *info={};
    if(trace_size) *trace_size=0;
    try {
        if(!X || !y || !o || !coef || !info) throw std::invalid_argument("null required argument");
        smm_stop_options defaults; smm_default_stop_options(&defaults);
        spectralmm::fit(*X,y,beta0,*o,coef,*info,k,stop ? stop : &defaults,trace,capacity,trace_size,family);
        return 0;
    } catch(const std::exception& e) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"%s",e.what());
        return 1;
    } catch(...) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"unknown native error");
        return 2;
    }
}

int32_t smm_fit_intercept(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *o, const smm_family_options *family,
    const smm_krylov_options *k, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_entry *trace, int64_t capacity,
    int64_t *trace_size, char *error, size_t error_capacity, int32_t intercept, int32_t penalize_intercept) {
    if(error && error_capacity) error[0]='\0';
    if(info) *info={};
    if(trace_size) *trace_size=0;
    try {
        if(!X || !y || !o || !coef || !info) throw std::invalid_argument("null required argument");
        if ((intercept!=0 && intercept!=1) || (penalize_intercept!=0 && penalize_intercept!=1))
            throw std::invalid_argument("intercept flags must be boolean");
        smm_stop_options defaults; smm_default_stop_options(&defaults);
        spectralmm::fit(*X,y,beta0,*o,coef,*info,k,stop ? stop : &defaults,trace,capacity,trace_size,family,intercept,penalize_intercept);
        return 0;
    } catch(const std::exception& e) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"%s",e.what());
        return 1;
    } catch(...) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"unknown native error");
        return 2;
    }
}
int32_t smm_fit_logged(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *o, const smm_family_options *family,
    const smm_krylov_options *k, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_detail *trace, int64_t capacity,
    int64_t *trace_size, char *error, size_t error_capacity, int32_t intercept, int32_t penalize_intercept) {
    if(error && error_capacity) error[0]='\0';
    if(info) *info={};
    if(trace_size) *trace_size=0;
    try {
        if(!X || !y || !o || !coef || !info) throw std::invalid_argument("null required argument");
        if ((intercept!=0 && intercept!=1) || (penalize_intercept!=0 && penalize_intercept!=1))
            throw std::invalid_argument("intercept flags must be boolean");
        smm_stop_options defaults; smm_default_stop_options(&defaults);
        spectralmm::fit(*X,y,beta0,*o,coef,*info,k,stop ? stop : &defaults,nullptr,capacity,trace_size,family,intercept,penalize_intercept,trace);
        return 0;
    } catch(const std::exception& e) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"%s",e.what());
        return 1;
    } catch(...) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"unknown native error");
        return 2;
    }
}

int32_t smm_infer(const smm_matrix *X, const double *y, const double *coef,
    const smm_options *options, const smm_family_options *family,
    int32_t intercept, int32_t cov_type, double dispersion, double *covariance,
    smm_inference_info *info, char *error, size_t capacity) {
    if (error && capacity) error[0]='\0';
    if (info) *info={};
    try {
        if (!X || !y || !coef || !options || !covariance || !info)
            throw std::invalid_argument("null required inference argument");
        if (intercept!=0 && intercept!=1) throw std::invalid_argument("invalid intercept flag");
        spectralmm::infer(*X,y,coef,*options,family,intercept,cov_type,dispersion,covariance,*info);
        return 0;
    } catch (const std::exception& e) {
        if (error && capacity) std::snprintf(error,capacity,"%s",e.what());
        return 1;
    } catch (...) {
        if (error && capacity) std::snprintf(error,capacity,"inference failed");
        return 2;
    }
}

int32_t smm_fit_logged_stats(const smm_matrix *X, const double *y, const double *beta0,
    const smm_options *o, const smm_family_options *family,
    const smm_krylov_options *k, const smm_stop_options *stop,
    double *coef, smm_info *info, smm_trace_detail *trace, int64_t capacity,
    int64_t *trace_size, char *error, size_t error_capacity, int32_t intercept, int32_t penalize_intercept, double *likelihood, double dispersion) {
    if(error && error_capacity) error[0]='\0';
    if(info) *info={};
    if(trace_size) *trace_size=0;
    try {
        if(!X || !y || !o || !coef || !info) throw std::invalid_argument("null required argument");
        if ((intercept!=0 && intercept!=1) || (penalize_intercept!=0 && penalize_intercept!=1))
            throw std::invalid_argument("intercept flags must be boolean");
        if (!std::isnan(dispersion) && (!(dispersion>0.) || !std::isfinite(dispersion)))
            throw std::invalid_argument("dispersion must be positive and finite or NaN");
        smm_stop_options defaults; smm_default_stop_options(&defaults);
        spectralmm::fit(*X,y,beta0,*o,coef,*info,k,stop ? stop : &defaults,nullptr,capacity,trace_size,family,intercept,penalize_intercept,trace,likelihood,dispersion);
        return 0;
    } catch(const std::exception& e) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"%s",e.what());
        return 1;
    } catch(...) {
        if(error && error_capacity) std::snprintf(error,error_capacity,"unknown native error");
        return 2;
    }
}

} // extern "C"
