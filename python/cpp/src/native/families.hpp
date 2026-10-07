// Included inside the core's anonymous namespace. Mirrors src/families.jl.
struct ModelOptions : smm_options {
    bool unpenalized = false;
    smm_family_options model;
    ModelOptions(const smm_options& o,const smm_family_options* f) : smm_options(o) {
        smm_default_family_options(&model); if(f) model=*f;
    }
};
double trials_at(const ModelOptions& o,int64_t i) {
    return o.model.trials ? o.model.trials[o.model.trials_count==1 ? 0 : i] : 1.;
}
double normal_logpdf(double x) { return -.5*x*x-.5*std::log(2.*std::acos(-1.)); }
double normal_logcdf(double x) {
    if(x>=-10.) return std::log(.5*std::erfc(-x/std::sqrt(2.)));
    // Mills-series tail: avoids erfc underflow and subtracting from one.
    double sum=1.,term=1.,previous=1.,xx=x*x;
    for(int k=1;k<=100;++k) {
        term*=-(2.*k-1.)/xx;
        if(std::abs(term)>=previous) break;
        sum+=term;
        if(std::abs(term)<=eps*std::abs(sum)) break;
        previous=std::abs(term);
    }
    return normal_logpdf(x)-std::log(-x)+std::log(sum);
}
double probit(double e) {
    return std::clamp(.5*std::erfc(-e/std::sqrt(2.)),eps,1.-eps);
}
double probit_information(double e) {
    return std::exp(2.*normal_logpdf(e)-normal_logcdf(e)-normal_logcdf(-e));
}
struct FamilyPoint { double mu, score, weight, objective; };
FamilyPoint family_point(double e,double y,const ModelOptions& o,int64_t i) {
    const auto& f=o.model;
    double mu=e,d=1.,v=1.,q=0.,w=1.,l=0.,r=y-e;
    switch(o.family) {
    case 0: return {e,-r,1.,.5*r*r};
    case 1:
        mu=sigmoid(e); return {mu,mu-y,std::max(1e-12,mu*(1.-mu)),
            std::max(e,0.)-y*e+std::log1p(std::exp(-std::abs(e)))};
    case 2:
        l=-normal_logcdf(y==1. ? e : -e);
        q=std::exp(normal_logpdf(e)+l)*(y==1. ? -1. : 1.);
        return {probit(e),q,std::max(1e-12,probit_information(e)),l};
    case 3: case 4: case 5: case 10: case 12:
        mu=std::exp(std::min(e,700.)); d=mu;
        if(o.family==3) { v=mu; l=mu-y*e; }
        else if(o.family==4) { v=mu*mu; l=y/mu+std::log(mu); }
        else if(o.family==5) { v=mu+mu*mu/f.theta; l=(y+f.theta)*std::log(f.theta+mu)-y*e; }
        else if(o.family==10) { v=1.; l=.5*(mu-y)*(mu-y); }
        else {
            v=std::pow(mu,f.power);
            if(std::abs(f.power-1.)<=std::sqrt(eps)) l=mu-y*e;
            else if(std::abs(f.power-2.)<=std::sqrt(eps)) l=y/mu+std::log(mu);
            else l=std::pow(mu,2.-f.power)/(2.-f.power)-y*std::pow(mu,1.-f.power)/(1.-f.power);
        }
        break;
    case 6: {
        double s=std::sqrt(r*r+f.smoothing*f.smoothing);
        q=-(f.tau-.5+r/(2.*s)); w=f.smoothing*f.smoothing/(2.*s*s*s);
        l=(f.tau-.5)*r+.5*s; return {e,q,std::max(1e-12,w),l};
    }
    case 7:
        w=r>=0 ? f.tau : 1.-f.tau;
        return {e,-w*r,std::max(1e-12,w),.5*w*r*r};
    case 8: {
        double u=r/f.delta, s=std::sqrt(1.+u*u);
        return {e,-r/s,std::max(1e-12,std::pow(1.+u*u,-1.5)),f.delta*f.delta*(s-1.)};
    }
    case 9: {
        double c=f.nu*f.sigma*f.sigma;
        w=std::max(1e-12,(f.nu+1.)/(c+r*r));
        return {e,-w*r,w,.5*(f.nu+1.)*std::log1p(r*r/c)};
    }
    case 11: {
        double pos=std::max(e,std::sqrt(eps)); mu=1./pos; d=-mu*mu; v=mu*mu;
        l=y*pos-std::log(pos); break;
    }
    case 13: {
        double n=trials_at(o,i); mu=n*sigmoid(e);
        double p=std::clamp(mu/n,eps,1.-eps); d=v=n*p*(1.-p);
        l=n*(std::max(e,0.)+std::log1p(std::exp(-std::abs(e))))-y*e; break;
    }
    default: throw std::invalid_argument("unsupported family");
    }
    v=std::max(v,eps);
    return {mu,(mu-y)*d/v,std::max(1e-12,d*d/v),l};
}
void validate_family(const ModelOptions& o,const Vec& y) {
    const auto& f=o.model;
    require(o.family>=0 && o.family<=13,"unsupported family");
    auto positive=[](double x) {return std::isfinite(x) && x>0.;};
    require(positive(f.theta) && positive(f.smoothing) && positive(f.delta) && positive(f.nu) && positive(f.sigma),"model scales must be finite and positive");
    require(std::isfinite(f.tau) && f.tau>0 && f.tau<1,"tau must be in (0,1)");
    require(std::isfinite(f.power) && f.power>=1 && f.power<=2,"power must be in [1,2]");
    require((!f.trials && f.trials_count==0) || (f.trials && (f.trials_count==1 || f.trials_count==y.size())),"trials must be scalar or length n");
    for(int64_t i=0;i<y.size();++i) {
        if(o.family==1 || o.family==2) require(y[i]==0 || y[i]==1,"Bernoulli response must be 0 or 1");
        if(o.family==3 || o.family==5 || o.family==12) require(y[i]>=0,"response must be nonnegative");
        if(o.family==4 || o.family==10 || o.family==11) require(y[i]>0,"response must be positive");
        if(o.family==13) require(positive(trials_at(o,i)) && y[i]>=0 && y[i]<=trials_at(o,i),"binomial requires positive trials and 0 <= y <= trials");
    }
}
double initial_eta(const Vec& y,const ModelOptions& o) {
    double m=y.mean();
    if(o.family==1 || o.family==2 || o.family==13) {
        if(o.family==13) { m=0.; for(int64_t i=0;i<y.size();++i) m+=y[i]/trials_at(o,i); m/=y.size(); }
        m=std::clamp(m,std::sqrt(eps),1.-std::sqrt(eps));
        if(o.family==2) {
            double lo=-10.,hi=10.;
            for(int j=0;j<60;++j) { double mid=(lo+hi)/2.; if(.5*std::erfc(-mid/std::sqrt(2.))<m) lo=mid; else hi=mid; }
            return (lo+hi)/2.;
        }
        return std::log(m/(1.-m));
    }
    if(o.family==3 || o.family==4 || o.family==5 || o.family==10 || o.family==12) return std::log(std::max(m,std::sqrt(eps)));
    if(o.family==11) return 1./std::max(m,std::sqrt(eps));
    return m;
}
