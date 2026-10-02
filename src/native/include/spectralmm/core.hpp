#pragma once
#include "c_api.h"
namespace spectralmm {
// Throws on invalid input or numerical failure; use smm_fit across an FFI.
void fit(const smm_matrix&, const double *y, const double *beta0,
         const smm_options&, double *coef, smm_info&,
         const smm_krylov_options *krylov = nullptr,
         const smm_stop_options *stop = nullptr, smm_trace_entry *trace = nullptr,
         int64_t trace_capacity = 0, int64_t *trace_size = nullptr, const smm_family_options *family = nullptr,
         bool intercept = false, bool penalize_intercept = true, smm_trace_detail *detail = nullptr);
void infer(const smm_matrix&, const double*, const double*, const smm_options&,
           const smm_family_options*, bool, int, double, double*, smm_inference_info&);
}
