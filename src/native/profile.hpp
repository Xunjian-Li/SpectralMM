#pragma once
// Diagnostic-only instrumentation. Timings are inclusive; nested matrix time
// is already included in its caller. Allocation counters cover Eigen buffers.
#ifdef SPECTRALMM_PROFILE
#include <chrono>
#include <cstdint>
#include <cstddef>
namespace smm_profile {
inline constexpr int count=8;
inline thread_local double seconds[count]{};
inline thread_local uint64_t allocations[count]{}, bytes[count]{};
inline thread_local uint64_t products[3]{}; // Xv, transpose(X)v, matrix block
inline thread_local int phase=0;
inline void reset() {
    for(int i=0;i<count;++i) { seconds[i]=0; allocations[i]=bytes[i]=0; }
    for(int i=0;i<3;++i) products[i]=0;
    phase=0;
}
struct Scope {
    int id,previous;
    std::chrono::steady_clock::time_point start;
    explicit Scope(int p):id(p),previous(phase),start(std::chrono::steady_clock::now()) { phase=p; }
    ~Scope() { seconds[id]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); phase=previous; }
};
}
extern "C" void smm_profile_allocation(std::size_t n) {
    ++smm_profile::allocations[smm_profile::phase]; smm_profile::bytes[smm_profile::phase]+=n;
}
extern "C" int smm_get_profile(double *t,uint64_t *a,uint64_t *b,int n) {
    if(n!=smm_profile::count) return -1;
    for(int i=0;i<n;++i) { t[i]=smm_profile::seconds[i]; a[i]=smm_profile::allocations[i]; b[i]=smm_profile::bytes[i]; }
    return n;
}
extern "C" int smm_get_products(uint64_t *out,int n) {
    if(!out || n!=3) return -1;
    for(int i=0;i<3;++i) out[i]=smm_profile::products[i];
    return 3;
}
#define SMM_PRODUCT(id) (++smm_profile::products[id])
#define SMM_SCOPE(id) smm_profile::Scope smm_scope_##id(id)
#define SMM_RESET() smm_profile::reset()
#else
#define SMM_PRODUCT(id)
#define SMM_SCOPE(id)
#define SMM_RESET()
#endif
