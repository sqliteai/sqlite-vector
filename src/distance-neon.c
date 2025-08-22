//
//  distance-neon.c
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#include "distance-neon.h"
#include "distance-cpu.h"
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>


#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern char *distance_backend_name;

// MARK: FLOAT32 -

float float32_distance_l2_impl_neon (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d  = vsubq_f32(va, vb);
        acc = vmlaq_f32(acc, d, d);  // acc += d * d
    }

    float sum;
    #if defined(__aarch64__)
    sum = vaddvq_f32(acc);              // fast horizontal add on arm64
    #else
    float tmp[4]; vst1q_f32(tmp, acc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    #endif

    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }

    return use_sqrt ? sqrtf(sum) : sum;
}

float float32_distance_l2_neon (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_neon(v1, v2, n, true);
}

float float32_distance_l2_squared_neon (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_neon(v1, v2, n, false);
}

float float32_distance_cosine_neon (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float32x4_t acc_dot  = vdupq_n_f32(0.0f);
    float32x4_t acc_a2   = vdupq_n_f32(0.0f);
    float32x4_t acc_b2   = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        acc_dot = vmlaq_f32(acc_dot, va, vb);      // dot += a * b
        acc_a2  = vmlaq_f32(acc_a2, va, va);       // norm_a += a * a
        acc_b2  = vmlaq_f32(acc_b2, vb, vb);       // norm_b += b * b
    }

    float d[4], a2[4], b2[4];
    vst1q_f32(d, acc_dot);
    vst1q_f32(a2, acc_a2);
    vst1q_f32(b2, acc_b2);

    float dot = d[0] + d[1] + d[2] + d[3];
    float norm_a = a2[0] + a2[1] + a2[2] + a2[3];
    float norm_b = b2[0] + b2[1] + b2[2] + b2[3];

    for (; i < n; ++i) {
        float ai = a[i];
        float bi = b[i];
        dot     += ai * bi;
        norm_a  += ai * ai;
        norm_b  += bi * bi;
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
    return 1.0f - (dot / (sqrtf(norm_a) * sqrtf(norm_b)));
}

float float32_distance_dot_neon (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);  // acc += a * b
    }

    float tmp[4];
    vst1q_f32(tmp, acc);
    float dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }

    return -dot;
}

float float32_distance_l1_neon (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d  = vabdq_f32(va, vb);  // |a - b|
        acc = vaddq_f32(acc, d);
    }

    float tmp[4];
    vst1q_f32(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < n; ++i) {
        sum += fabsf(a[i] - b[i]);
    }

    return sum;
}

// MARK: - BFLOAT16 -

static inline float32x4_t bf16x4_to_f32x4_u16 (uint16x4_t h) {
    // widen u16 -> u32 and shift left 16: exact bf16->f32 bit pattern
    uint32x4_t u32 = vshll_n_u16(h, 16);
    return vreinterpretq_f32_u32(u32);
}

float bfloat16_distance_l2_impl_neon (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    // Accumulate in f64 to avoid overflow from huge bf16 values.
    float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = vdupq_n_f64(0.0);
    int i = 0;
    
    for (; i <= n - 8; i += 8) {
        uint16x8_t av16 = vld1q_u16(a + i);
        uint16x8_t bv16 = vld1q_u16(b + i);
        
        // low 4
        float32x4_t va0 = bf16x4_to_f32x4_u16(vget_low_u16(av16));
        float32x4_t vb0 = bf16x4_to_f32x4_u16(vget_low_u16(bv16));
        float32x4_t d0  = vsubq_f32(va0, vb0);
        // mask-out NaNs: m = (d==d)
        uint32x4_t m0   = vceqq_f32(d0, d0);
        d0 = vbslq_f32(m0, d0, vdupq_n_f32(0.0f));
        float64x2_t d0lo = vcvt_f64_f32(vget_low_f32(d0));
        float64x2_t d0hi = vcvt_f64_f32(vget_high_f32(d0));
        acc0 = vfmaq_f64(acc0, d0lo, d0lo);
        acc1 = vfmaq_f64(acc1, d0hi, d0hi);
        
        // high 4
        float32x4_t va1 = bf16x4_to_f32x4_u16(vget_high_u16(av16));
        float32x4_t vb1 = bf16x4_to_f32x4_u16(vget_high_u16(bv16));
        float32x4_t d1  = vsubq_f32(va1, vb1);
        uint32x4_t m1   = vceqq_f32(d1, d1);
        d1 = vbslq_f32(m1, d1, vdupq_n_f32(0.0f));
        float64x2_t d1lo = vcvt_f64_f32(vget_low_f32(d1));
        float64x2_t d1hi = vcvt_f64_f32(vget_high_f32(d1));
        acc0 = vfmaq_f64(acc0, d1lo, d1lo);
        acc1 = vfmaq_f64(acc1, d1hi, d1hi);
    }
    
    if (i <= n - 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);
        float32x4_t d   = vsubq_f32(bf16x4_to_f32x4_u16(av16),
                                    bf16x4_to_f32x4_u16(bv16));
        uint32x4_t m    = vceqq_f32(d, d);
        d = vbslq_f32(m, d, vdupq_n_f32(0.0f));
        float64x2_t dlo = vcvt_f64_f32(vget_low_f32(d));
        float64x2_t dhi = vcvt_f64_f32(vget_high_f32(d));
        acc0 = vfmaq_f64(acc0, dlo, dlo);
        acc1 = vfmaq_f64(acc1, dhi, dhi);
        i += 4;
    }
    
    double sum = vaddvq_f64(vaddq_f64(acc0, acc1));
    
    // scalar tail; treat NaN as 0, Inf as +Inf result
    for (; i < n; ++i) {
        float d = bfloat16_to_float32(a[i]) - bfloat16_to_float32(b[i]);
        if (isinf(d)) return INFINITY;
        if (!isnan(d)) sum = fma((double)d, (double)d, sum);
    }
    
    return use_sqrt ? (float)sqrt(sum) : (float)sum;
}

float bfloat16_distance_l2_neon (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_neon(v1, v2, n, true);
}

float bfloat16_distance_l2_squared_neon (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_neon(v1, v2, n, false);
}

float bfloat16_distance_cosine_neon (const void *v1, const void *v2, int n) {
    const uint16_t *restrict a = (const uint16_t *restrict)v1;
    const uint16_t *restrict b = (const uint16_t *restrict)v2;

    float32x4_t acc_dot = vdupq_n_f32(0.0f);
    float32x4_t acc_a2  = vdupq_n_f32(0.0f);
    float32x4_t acc_b2  = vdupq_n_f32(0.0f);
    int i = 0;

    // process 8 elements per iteration
    for (; i <= n - 8; i += 8) {
        uint16x8_t av16 = vld1q_u16(a + i);
        uint16x8_t bv16 = vld1q_u16(b + i);

        // low 4
        float32x4_t va0 = bf16x4_to_f32x4_u16(vget_low_u16(av16));
        float32x4_t vb0 = bf16x4_to_f32x4_u16(vget_low_u16(bv16));
        acc_dot = vmlaq_f32(acc_dot, va0, vb0);
        acc_a2  = vmlaq_f32(acc_a2,  va0, va0);
        acc_b2  = vmlaq_f32(acc_b2,  vb0, vb0);

        // high 4
        float32x4_t va1 = bf16x4_to_f32x4_u16(vget_high_u16(av16));
        float32x4_t vb1 = bf16x4_to_f32x4_u16(vget_high_u16(bv16));
        acc_dot = vmlaq_f32(acc_dot, va1, vb1);
        acc_a2  = vmlaq_f32(acc_a2,  va1, va1);
        acc_b2  = vmlaq_f32(acc_b2,  vb1, vb1);
    }

    // optional mid-tail of 4
    if (i <= n - 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);
        float32x4_t va = bf16x4_to_f32x4_u16(av16);
        float32x4_t vb = bf16x4_to_f32x4_u16(bv16);
        acc_dot = vmlaq_f32(acc_dot, va, vb);
        acc_a2  = vmlaq_f32(acc_a2,  va, va);
        acc_b2  = vmlaq_f32(acc_b2,  vb, vb);
        i += 4;
    }

    // horizontal reduction
    float dot, norm_a, norm_b;
#if defined(__aarch64__)
    dot    = vaddvq_f32(acc_dot);
    norm_a = vaddvq_f32(acc_a2);
    norm_b = vaddvq_f32(acc_b2);
#else
    float d[4], a2[4], b2[4];
    vst1q_f32(d,  acc_dot);
    vst1q_f32(a2, acc_a2);
    vst1q_f32(b2, acc_b2);
    dot    = d[0]  + d[1]  + d[2]  + d[3];
    norm_a = a2[0] + a2[1] + a2[2] + a2[3];
    norm_b = b2[0] + b2[1] + b2[2] + b2[3];
#endif

    // scalar tail
    for (; i < n; ++i) {
        float fa = bfloat16_to_float32(a[i]);
        float fb = bfloat16_to_float32(b[i]);
        dot    += fa * fb;
        norm_a += fa * fa;
        norm_b += fb * fb;
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
    return 1.0f - (dot / (sqrtf(norm_a) * sqrtf(norm_b)));
}

float bfloat16_distance_dot_neon (const void *v1, const void *v2, int n) {
    const uint16_t *restrict a = (const uint16_t *restrict)v1;
    const uint16_t *restrict b = (const uint16_t *restrict)v2;

    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    // process 8 elements per iteration
    for (; i <= n - 8; i += 8) {
        uint16x8_t av16 = vld1q_u16(a + i);
        uint16x8_t bv16 = vld1q_u16(b + i);

        // low 4
        float32x4_t va0 = bf16x4_to_f32x4_u16(vget_low_u16(av16));
        float32x4_t vb0 = bf16x4_to_f32x4_u16(vget_low_u16(bv16));
        acc = vmlaq_f32(acc, va0, vb0);

        // high 4
        float32x4_t va1 = bf16x4_to_f32x4_u16(vget_high_u16(av16));
        float32x4_t vb1 = bf16x4_to_f32x4_u16(vget_high_u16(bv16));
        acc = vmlaq_f32(acc, va1, vb1);
    }

    // optional mid-tail of 4
    if (i <= n - 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);
        float32x4_t va = bf16x4_to_f32x4_u16(av16);
        float32x4_t vb = bf16x4_to_f32x4_u16(bv16);
        acc = vmlaq_f32(acc, va, vb);
        i += 4;
    }

    // horizontal sum
    float dot;
#if defined(__aarch64__)
    dot = vaddvq_f32(acc);
#else
    float tmp[4]; vst1q_f32(tmp, acc);
    dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif

    // scalar tail
    for (; i < n; ++i) {
        dot += bfloat16_to_float32(a[i]) * bfloat16_to_float32(b[i]);
    }

    return -dot;
}

float bfloat16_distance_l1_neon (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);

        float32x4_t va = bf16x4_to_f32x4_u16(av16);
        float32x4_t vb = bf16x4_to_f32x4_u16(bv16);

        float32x4_t d  = vabdq_f32(va, vb);   // |a - b|
        acc = vaddq_f32(acc, d);
    }

    // horizontal reduction
    float sum;
#if defined(__aarch64__)
    sum = vaddvq_f32(acc);
#else
    float tmp[4]; vst1q_f32(tmp, acc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif

    // scalar tail
    for (; i < n; ++i) {
        float fa = bfloat16_to_float32(a[i]);
        float fb = bfloat16_to_float32(b[i]);
        sum += fabsf(fa - fb);
    }

    return sum;
}

// MARK: - FLOAT16 -

// vector converter: 4×f16 bits (u16) -> f32x4
static inline float32x4_t f16x4_to_f32x4_u16(uint16x4_t h) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    /* Fast path: NEON FP16 -> FP32 */
    float16x4_t h16 = vreinterpret_f16_u16(h);
    return vcvt_f32_f16(h16);
#else
    /* Portable per-lane conversion via your helper */
    float tmp[4];
    tmp[0] = float16_to_float32(vget_lane_u16(h, 0));
    tmp[1] = float16_to_float32(vget_lane_u16(h, 1));
    tmp[2] = float16_to_float32(vget_lane_u16(h, 2));
    tmp[3] = float16_to_float32(vget_lane_u16(h, 3));
    return vld1q_f32(tmp);
#endif
}

float float16_distance_l2_impl_neon (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    const uint16x4_t EXP_MASK  = vdup_n_u16(0x7C00u);
    const uint16x4_t FRAC_MASK = vdup_n_u16(0x03FFu);
    const uint16x4_t SIGN_MASK = vdup_n_u16(0x8000u);
    const uint16x4_t ZERO16    = vdup_n_u16(0);

    float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = vdupq_n_f64(0.0);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);

        /* detect Inf mismatches: (a Inf XOR b Inf) OR (both Inf and sign differs) */
        uint16x4_t a_exp_all1 = vceq_u16(vand_u16(av16, EXP_MASK), EXP_MASK);
        uint16x4_t b_exp_all1 = vceq_u16(vand_u16(bv16, EXP_MASK), EXP_MASK);
        uint16x4_t a_frac_zero= vceq_u16(vand_u16(av16, FRAC_MASK), ZERO16);
        uint16x4_t b_frac_zero= vceq_u16(vand_u16(bv16, FRAC_MASK), ZERO16);

        uint16x4_t a_inf = vand_u16(a_exp_all1, a_frac_zero);
        uint16x4_t b_inf = vand_u16(b_exp_all1, b_frac_zero);

        uint16x4_t a_sign = vand_u16(av16, SIGN_MASK);
        uint16x4_t b_sign = vand_u16(bv16, SIGN_MASK);
        uint16x4_t same_sign = vceq_u16(veor_u16(a_sign, b_sign), ZERO16);
        uint16x4_t sign_diff = vmvn_u16(same_sign);

        uint16x4_t mismatch = vorr_u16(
                                vorr_u16(vand_u16(a_inf, vmvn_u16(b_inf)),
                                         vand_u16(b_inf, vmvn_u16(a_inf))),
                                vand_u16(vand_u16(a_inf, b_inf), sign_diff));
        if (vmaxv_u16(mismatch)) return INFINITY;

        /* convert to f32 then to f64, subtract in f64, mask NaNs to zero */
        float32x4_t af = f16x4_to_f32x4_u16(av16);
        float32x4_t bf = f16x4_to_f32x4_u16(bv16);
        float32x4_t d32 = vsubq_f32(af, bf);
        uint32x4_t m = vceqq_f32(d32, d32);                    /* true where not-NaN */
        d32 = vbslq_f32(m, d32, vdupq_n_f32(0.0f));

        float64x2_t dlo = vcvt_f64_f32(vget_low_f32(d32));
        float64x2_t dhi = vcvt_f64_f32(vget_high_f32(d32));
#if defined(__ARM_FEATURE_FMA)
        acc0 = vfmaq_f64(acc0, dlo, dlo);
        acc1 = vfmaq_f64(acc1, dhi, dhi);
#else
        acc0 = vaddq_f64(acc0, vmulq_f64(dlo, dlo));
        acc1 = vaddq_f64(acc1, vmulq_f64(dhi, dhi));
#endif
    }

    double sum = vaddvq_f64(vaddq_f64(acc0, acc1));

    /* tail (scalar; same Inf/NaN policy) */
    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((f16_is_inf(ai) || f16_is_inf(bi)) && !(f16_is_inf(ai) && f16_is_inf(bi) && f16_sign(ai)==f16_sign(bi))) return INFINITY;
        float xa = float16_to_float32(ai);
        float xb = float16_to_float32(bi);
        float d  = xa - xb;
        if (!isnan(d)) sum = fma((double)d, (double)d, sum);
    }

    return use_sqrt ? (float)sqrt(sum) : (float)sum;
}

float float16_distance_l2_neon (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_neon(v1, v2, n, true);
}
float float16_distance_l2_squared_neon (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_neon(v1, v2, n, false);
}

/* =========================================================================
   Cosine distance (1 - dot/(||a||*||b||)) -- float16 (uint16_t)
   ========================================================================= */
float float16_distance_cosine_neon (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    const uint16x4_t EXP_MASK  = vdup_n_u16(0x7C00u);
    const uint16x4_t FRAC_MASK = vdup_n_u16(0x03FFu);
    const uint16x4_t ZERO16    = vdup_n_u16(0);

    float64x2_t acc_dot_lo = vdupq_n_f64(0.0), acc_dot_hi = vdupq_n_f64(0.0);
    float64x2_t acc_a2_lo  = vdupq_n_f64(0.0), acc_a2_hi  = vdupq_n_f64(0.0);
    float64x2_t acc_b2_lo  = vdupq_n_f64(0.0), acc_b2_hi  = vdupq_n_f64(0.0);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);

        /* if any lane has ±Inf, return 1.0 (max distance) */
        uint16x4_t a_inf = vand_u16(vceq_u16(vand_u16(av16, EXP_MASK), EXP_MASK),
                                    vceq_u16(vand_u16(av16, FRAC_MASK), ZERO16));
        uint16x4_t b_inf = vand_u16(vceq_u16(vand_u16(bv16, EXP_MASK), EXP_MASK),
                                    vceq_u16(vand_u16(bv16, FRAC_MASK), ZERO16));
        if (vmaxv_u16(vorr_u16(a_inf, b_inf))) return 1.0f;

        float32x4_t ax = f16x4_to_f32x4_u16(av16);
        float32x4_t by = f16x4_to_f32x4_u16(bv16);

        /* zero out NaNs */
        uint32x4_t mx = vceqq_f32(ax, ax);
        uint32x4_t my = vceqq_f32(by, by);
        ax = vbslq_f32(mx, ax, vdupq_n_f32(0.0f));
        by = vbslq_f32(my, by, vdupq_n_f32(0.0f));

        /* widen to f64 and accumulate */
        float64x2_t ax_lo = vcvt_f64_f32(vget_low_f32(ax)), ax_hi = vcvt_f64_f32(vget_high_f32(ax));
        float64x2_t by_lo = vcvt_f64_f32(vget_low_f32(by)), by_hi = vcvt_f64_f32(vget_high_f32(by));

#if defined(__ARM_FEATURE_FMA)
        acc_dot_lo = vfmaq_f64(acc_dot_lo, ax_lo, by_lo);
        acc_dot_hi = vfmaq_f64(acc_dot_hi, ax_hi, by_hi);
        acc_a2_lo  = vfmaq_f64(acc_a2_lo,  ax_lo, ax_lo);
        acc_a2_hi  = vfmaq_f64(acc_a2_hi,  ax_hi, ax_hi);
        acc_b2_lo  = vfmaq_f64(acc_b2_lo,  by_lo, by_lo);
        acc_b2_hi  = vfmaq_f64(acc_b2_hi,  by_hi, by_hi);
#else
        acc_dot_lo = vaddq_f64(acc_dot_lo, vmulq_f64(ax_lo, by_lo));
        acc_dot_hi = vaddq_f64(acc_dot_hi, vmulq_f64(ax_hi, by_hi));
        acc_a2_lo  = vaddq_f64(acc_a2_lo,  vmulq_f64(ax_lo, ax_lo));
        acc_a2_hi  = vaddq_f64(acc_a2_hi,  vmulq_f64(ax_hi, ax_hi));
        acc_b2_lo  = vaddq_f64(acc_b2_lo,  vmulq_f64(by_lo, by_lo));
        acc_b2_hi  = vaddq_f64(acc_b2_hi,  vmulq_f64(by_hi, by_hi));
#endif
    }

    double dot  = vaddvq_f64(vaddq_f64(acc_dot_lo, acc_dot_hi));
    double normx= vaddvq_f64(vaddq_f64(acc_a2_lo,  acc_a2_hi));
    double normy= vaddvq_f64(vaddq_f64(acc_b2_lo,  acc_b2_hi));

    /* tail (scalar) */
    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        if (f16_is_inf(ai) || f16_is_inf(bi)) return 1.0f;
        double x = (double)float16_to_float32(ai);
        double y = (double)float16_to_float32(bi);
        dot  += x * y;
        normx+= x * x;
        normy+= y * y;
    }

    double denom = sqrt(normx) * sqrt(normy);
    if (!(denom > 0.0) || !isfinite(denom) || !isfinite(dot)) return 1.0f;

    double c = dot / denom;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    return (float)(1.0 - c);
}

/* =========================================================================
   Dot (returns -dot) -- float16 (uint16_t)
   ========================================================================= */
float float16_distance_dot_neon (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    const uint16x4_t EXP_MASK  = vdup_n_u16(0x7C00u);
    const uint16x4_t FRAC_MASK = vdup_n_u16(0x03FFu);
    const uint16x4_t ZERO16    = vdup_n_u16(0);

    float64x2_t acc_lo = vdupq_n_f64(0.0), acc_hi = vdupq_n_f64(0.0);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);

        /* if any lane is ±Inf, do scalar fallback for this block to get sign-correct ±Inf */
        uint16x4_t a_inf = vand_u16(vceq_u16(vand_u16(av16, EXP_MASK), EXP_MASK),
                                    vceq_u16(vand_u16(av16, FRAC_MASK), ZERO16));
        uint16x4_t b_inf = vand_u16(vceq_u16(vand_u16(bv16, EXP_MASK), EXP_MASK),
                                    vceq_u16(vand_u16(bv16, FRAC_MASK), ZERO16));
        if (vmaxv_u16(vorr_u16(a_inf, b_inf))) {
            for (int k=0;k<4;++k){
                float x = float16_to_float32(a[i+k]);
                float y = float16_to_float32(b[i+k]);
                if (isnan(x) || isnan(y)) continue;
                double p = (double)x * (double)y;
                if (isinf(p)) return (p>0)? -INFINITY : INFINITY;
                acc_lo = vsetq_lane_f64(vgetq_lane_f64(acc_lo,0)+p, acc_lo, 0); /* cheap add */
            }
            continue;
        }

        float32x4_t ax = f16x4_to_f32x4_u16(av16);
        float32x4_t by = f16x4_to_f32x4_u16(bv16);

        /* zero out NaNs */
        uint32x4_t mx = vceqq_f32(ax, ax);
        uint32x4_t my = vceqq_f32(by, by);
        ax = vbslq_f32(mx, ax, vdupq_n_f32(0.0f));
        by = vbslq_f32(my, by, vdupq_n_f32(0.0f));

        float32x4_t prod = vmulq_f32(ax, by);
        float64x2_t lo = vcvt_f64_f32(vget_low_f32(prod));
        float64x2_t hi = vcvt_f64_f32(vget_high_f32(prod));
        acc_lo = vaddq_f64(acc_lo, lo);
        acc_hi = vaddq_f64(acc_hi, hi);
    }

    double dot = vaddvq_f64(vaddq_f64(acc_lo, acc_hi));

    for (; i < n; ++i) {
        float x = float16_to_float32(a[i]);
        float y = float16_to_float32(b[i]);
        if (isnan(x) || isnan(y)) continue;
        double p = (double)x * (double)y;
        if (isinf(p)) return (p>0)? -INFINITY : INFINITY;
        dot += p;
    }

    return (float)(-dot);
}

/* =========================================================================
   L1 (sum |a-b|) -- float16 (uint16_t)
   ========================================================================= */
float float16_distance_l1_neon (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    const uint16x4_t EXP_MASK  = vdup_n_u16(0x7C00u);
    const uint16x4_t FRAC_MASK = vdup_n_u16(0x03FFu);
    const uint16x4_t SIGN_MASK = vdup_n_u16(0x8000u);
    const uint16x4_t ZERO16    = vdup_n_u16(0);

    float64x2_t acc = vdupq_n_f64(0.0);
    int i = 0;

    for (; i <= n - 4; i += 4) {
        uint16x4_t av16 = vld1_u16(a + i);
        uint16x4_t bv16 = vld1_u16(b + i);

        /* Inf mismatch => +Inf */
        uint16x4_t a_exp_all1 = vceq_u16(vand_u16(av16, EXP_MASK), EXP_MASK);
        uint16x4_t b_exp_all1 = vceq_u16(vand_u16(bv16, EXP_MASK), EXP_MASK);
        uint16x4_t a_frac_zero= vceq_u16(vand_u16(av16, FRAC_MASK), ZERO16);
        uint16x4_t b_frac_zero= vceq_u16(vand_u16(bv16, FRAC_MASK), ZERO16);
        uint16x4_t a_inf = vand_u16(a_exp_all1, a_frac_zero);
        uint16x4_t b_inf = vand_u16(b_exp_all1, b_frac_zero);
        uint16x4_t a_sign = vand_u16(av16, SIGN_MASK);
        uint16x4_t b_sign = vand_u16(bv16, SIGN_MASK);
        uint16x4_t same_sign = vceq_u16(veor_u16(a_sign, b_sign), ZERO16);
        uint16x4_t sign_diff = vmvn_u16(same_sign);
        uint16x4_t mismatch = vorr_u16(
                                vorr_u16(vand_u16(a_inf, vmvn_u16(b_inf)),
                                         vand_u16(b_inf, vmvn_u16(a_inf))),
                                vand_u16(vand_u16(a_inf, b_inf), sign_diff));
        if (vmaxv_u16(mismatch)) return INFINITY;

        float32x4_t af = f16x4_to_f32x4_u16(av16);
        float32x4_t bf = f16x4_to_f32x4_u16(bv16);
        float32x4_t d  = vabdq_f32(af, bf);                 /* |a-b| */
        uint32x4_t m   = vceqq_f32(d, d);                   /* mask NaNs -> 0 */
        d = vbslq_f32(m, d, vdupq_n_f32(0.0f));

        float64x2_t lo = vcvt_f64_f32(vget_low_f32(d));
        float64x2_t hi = vcvt_f64_f32(vget_high_f32(d));
        acc = vaddq_f64(acc, lo);
        acc = vaddq_f64(acc, hi);
    }

    double sum = vaddvq_f64(acc);

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((f16_is_inf(ai) || f16_is_inf(bi)) && !(f16_is_inf(ai) && f16_is_inf(bi) && f16_sign(ai)==f16_sign(bi))) return INFINITY;
        float da = float16_to_float32(ai);
        float db = float16_to_float32(bi);
        float d  = fabsf(da - db);
        if (!isnan(d)) sum += d;
    }
    return (float)sum;
}


// MARK: - UINT8 -

static inline float uint8_distance_l2_impl_neon(const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    uint32x4_t acc = vmovq_n_u32(0);
    int i = 0;

    for (; i <= n - 16; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);

        // compute 8-bit differences widened to signed 16-bit
        int16x8_t diff_lo = (int16x8_t)vsubl_u8(vget_low_u8(va), vget_low_u8(vb));
        int16x8_t diff_hi = (int16x8_t)vsubl_u8(vget_high_u8(va), vget_high_u8(vb));

        // widen to signed 32-bit and square
        int32x4_t diff_lo_0 = vmovl_s16(vget_low_s16(diff_lo));
        int32x4_t diff_lo_1 = vmovl_s16(vget_high_s16(diff_lo));
        int32x4_t diff_hi_0 = vmovl_s16(vget_low_s16(diff_hi));
        int32x4_t diff_hi_1 = vmovl_s16(vget_high_s16(diff_hi));

        diff_lo_0 = vmulq_s32(diff_lo_0, diff_lo_0);
        diff_lo_1 = vmulq_s32(diff_lo_1, diff_lo_1);
        diff_hi_0 = vmulq_s32(diff_hi_0, diff_hi_0);
        diff_hi_1 = vmulq_s32(diff_hi_1, diff_hi_1);

        // accumulate into uint32_t accumulator
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_lo_0));
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_lo_1));
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_hi_0));
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_hi_1));
    }

    // horizontal sum
    uint64x2_t sum64 = vpaddlq_u32(acc);
    uint64_t final_sum = vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);

    // tail
    for (; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        final_sum += (uint64_t)(diff * diff);
    }

    return use_sqrt ? sqrtf((float)final_sum) : (float)final_sum;
}

float uint8_distance_l2_neon (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_neon(v1, v2, n, true);
}

float uint8_distance_l2_squared_neon (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_neon(v1, v2, n, false);
}

float uint8_distance_cosine_neon (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    uint32x4_t dot_acc = vmovq_n_u32(0);
    uint32x4_t norm_a_acc = vmovq_n_u32(0);
    uint32x4_t norm_b_acc = vmovq_n_u32(0);
    
    int i = 0;
    for (; i <= n - 16; i += 16) {
        // Load 16 bytes from each vector
        uint8x16_t va_u8 = vld1q_u8(a + i);
        uint8x16_t vb_u8 = vld1q_u8(b + i);
        
        // Convert to uint16x8_t
        uint16x8_t va_lo_u16 = vmovl_u8(vget_low_u8(va_u8));
        uint16x8_t va_hi_u16 = vmovl_u8(vget_high_u8(va_u8));
        uint16x8_t vb_lo_u16 = vmovl_u8(vget_low_u8(vb_u8));
        uint16x8_t vb_hi_u16 = vmovl_u8(vget_high_u8(vb_u8));
        
        // Multiply for dot product
        uint32x4_t dot_lo = vmull_u16(vget_low_u16(va_lo_u16), vget_low_u16(vb_lo_u16));
        uint32x4_t dot_hi = vmull_u16(vget_high_u16(va_lo_u16), vget_high_u16(vb_lo_u16));
        uint32x4_t dot_lo2 = vmull_u16(vget_low_u16(va_hi_u16), vget_low_u16(vb_hi_u16));
        uint32x4_t dot_hi2 = vmull_u16(vget_high_u16(va_hi_u16), vget_high_u16(vb_hi_u16));
        
        // Multiply for norms
        uint32x4_t a2_lo = vmull_u16(vget_low_u16(va_lo_u16), vget_low_u16(va_lo_u16));
        uint32x4_t a2_hi = vmull_u16(vget_high_u16(va_lo_u16), vget_high_u16(va_lo_u16));
        uint32x4_t a2_lo2 = vmull_u16(vget_low_u16(va_hi_u16), vget_low_u16(va_hi_u16));
        uint32x4_t a2_hi2 = vmull_u16(vget_high_u16(va_hi_u16), vget_high_u16(va_hi_u16));
        
        uint32x4_t b2_lo = vmull_u16(vget_low_u16(vb_lo_u16), vget_low_u16(vb_lo_u16));
        uint32x4_t b2_hi = vmull_u16(vget_high_u16(vb_lo_u16), vget_high_u16(vb_lo_u16));
        uint32x4_t b2_lo2 = vmull_u16(vget_low_u16(vb_hi_u16), vget_low_u16(vb_hi_u16));
        uint32x4_t b2_hi2 = vmull_u16(vget_high_u16(vb_hi_u16), vget_high_u16(vb_hi_u16));
        
        // Accumulate
        dot_acc     = vaddq_u32(dot_acc, dot_lo);
        dot_acc     = vaddq_u32(dot_acc, dot_hi);
        dot_acc     = vaddq_u32(dot_acc, dot_lo2);
        dot_acc     = vaddq_u32(dot_acc, dot_hi2);
        
        norm_a_acc  = vaddq_u32(norm_a_acc, a2_lo);
        norm_a_acc  = vaddq_u32(norm_a_acc, a2_hi);
        norm_a_acc  = vaddq_u32(norm_a_acc, a2_lo2);
        norm_a_acc  = vaddq_u32(norm_a_acc, a2_hi2);
        
        norm_b_acc  = vaddq_u32(norm_b_acc, b2_lo);
        norm_b_acc  = vaddq_u32(norm_b_acc, b2_hi);
        norm_b_acc  = vaddq_u32(norm_b_acc, b2_lo2);
        norm_b_acc  = vaddq_u32(norm_b_acc, b2_hi2);
    }
    
    // Horizontal sum
    uint32_t dot = vgetq_lane_u32(dot_acc, 0) + vgetq_lane_u32(dot_acc, 1) +
    vgetq_lane_u32(dot_acc, 2) + vgetq_lane_u32(dot_acc, 3);
    
    uint32_t norm_a = vgetq_lane_u32(norm_a_acc, 0) + vgetq_lane_u32(norm_a_acc, 1) +
    vgetq_lane_u32(norm_a_acc, 2) + vgetq_lane_u32(norm_a_acc, 3);
    
    uint32_t norm_b = vgetq_lane_u32(norm_b_acc, 0) + vgetq_lane_u32(norm_b_acc, 1) +
    vgetq_lane_u32(norm_b_acc, 2) + vgetq_lane_u32(norm_b_acc, 3);
    
    // Tail loop
    for (; i < n; ++i) {
        int ai = a[i];
        int bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    
    if (norm_a == 0 || norm_b == 0) return 1.0f;
    return 1.0f - (dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b)));
}

float uint8_distance_dot_neon (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    uint32x4_t dot_acc = vmovq_n_u32(0);  // 4-lane accumulator
    int i = 0;
    
    for (; i <= n - 16; i += 16) {
        uint8x16_t va_u8 = vld1q_u8(a + i);
        uint8x16_t vb_u8 = vld1q_u8(b + i);
        
        // Widen to 16-bit
        uint16x8_t va_lo = vmovl_u8(vget_low_u8(va_u8));
        uint16x8_t vb_lo = vmovl_u8(vget_low_u8(vb_u8));
        uint16x8_t va_hi = vmovl_u8(vget_high_u8(va_u8));
        uint16x8_t vb_hi = vmovl_u8(vget_high_u8(vb_u8));
        
        // Multiply low and high halves
        uint32x4_t dot_lo = vmull_u16(vget_low_u16(va_lo), vget_low_u16(vb_lo));
        uint32x4_t dot_hi = vmull_u16(vget_high_u16(va_lo), vget_high_u16(vb_lo));
        uint32x4_t dot_lo2 = vmull_u16(vget_low_u16(va_hi), vget_low_u16(vb_hi));
        uint32x4_t dot_hi2 = vmull_u16(vget_high_u16(va_hi), vget_high_u16(vb_hi));
        
        // Accumulate
        dot_acc = vaddq_u32(dot_acc, dot_lo);
        dot_acc = vaddq_u32(dot_acc, dot_hi);
        dot_acc = vaddq_u32(dot_acc, dot_lo2);
        dot_acc = vaddq_u32(dot_acc, dot_hi2);
    }
    
    // Horizontal add of 4 lanes
    uint32_t dot = vgetq_lane_u32(dot_acc, 0) +
    vgetq_lane_u32(dot_acc, 1) +
    vgetq_lane_u32(dot_acc, 2) +
    vgetq_lane_u32(dot_acc, 3);
    
    // Tail loop
    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }
    
    return -(float)dot;  // negative dot product = dot distance
}

float uint8_distance_l1_neon (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    uint16x8_t sum_acc_lo = vdupq_n_u16(0);
    uint16x8_t sum_acc_hi = vdupq_n_u16(0);
    int i = 0;
    
    for (; i <= n - 16; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        
        // Compute absolute difference
        uint8x16_t abs_diff = vabdq_u8(va, vb);
        
        // Split and widen to 16-bit
        uint16x8_t abs_lo = vmovl_u8(vget_low_u8(abs_diff));
        uint16x8_t abs_hi = vmovl_u8(vget_high_u8(abs_diff));
        
        sum_acc_lo = vaddq_u16(sum_acc_lo, abs_lo);
        sum_acc_hi = vaddq_u16(sum_acc_hi, abs_hi);
    }
    
    // Combine 16-bit accumulators
    uint16x8_t total_acc = vaddq_u16(sum_acc_lo, sum_acc_hi);
    
    // Horizontal sum
    uint16_t tmp[8];
    vst1q_u16(tmp, total_acc);
    uint32_t total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    
    // Tail loop
    for (; i < n; ++i) {
        total += (uint32_t)abs((int)a[i] - (int)b[i]);
    }
    
    return (float)total;
}

// MARK: - INT8 -

static inline float int8_distance_l2_neon_imp (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    uint32x4_t acc = vmovq_n_u32(0);
    int i = 0;

    for (; i <= n - 16; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        // signed widening subtraction: int8 → int16
        int16x8_t diff_lo = vsubl_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t diff_hi = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));

        // widen to int32 and square
        int32x4_t diff_lo_0 = vmovl_s16(vget_low_s16(diff_lo));
        int32x4_t diff_lo_1 = vmovl_s16(vget_high_s16(diff_lo));
        int32x4_t diff_hi_0 = vmovl_s16(vget_low_s16(diff_hi));
        int32x4_t diff_hi_1 = vmovl_s16(vget_high_s16(diff_hi));

        diff_lo_0 = vmulq_s32(diff_lo_0, diff_lo_0);
        diff_lo_1 = vmulq_s32(diff_lo_1, diff_lo_1);
        diff_hi_0 = vmulq_s32(diff_hi_0, diff_hi_0);
        diff_hi_1 = vmulq_s32(diff_hi_1, diff_hi_1);

        // accumulate, cast to uint32 to match accumulator type
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_lo_0));
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_lo_1));
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_hi_0));
        acc = vaddq_u32(acc, vreinterpretq_u32_s32(diff_hi_1));
    }

    // horizontal sum
    uint64x2_t sum64 = vpaddlq_u32(acc);
    uint64_t final_sum = vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);

    // tail
    for (; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        final_sum += (uint64_t)(diff * diff);
    }

    return use_sqrt ? sqrtf((float)final_sum) : (float)final_sum;
}

float int8_distance_l2_neon (const void *v1, const void *v2, int n) {
    return int8_distance_l2_neon_imp(v1, v2, n, true);
}

float int8_distance_l2_squared_neon (const void *v1, const void *v2, int n) {
    return int8_distance_l2_neon_imp(v1, v2, n, false);
}

float int8_distance_cosine_neon (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    int32x4_t acc_dot  = vdupq_n_s32(0);
    int32x4_t acc_a2   = vdupq_n_s32(0);
    int32x4_t acc_b2   = vdupq_n_s32(0);
    int i = 0;

    for (; i <= n - 16; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t lo_a = vmovl_s8(vget_low_s8(va));
        int16x8_t hi_a = vmovl_s8(vget_high_s8(va));
        int16x8_t lo_b = vmovl_s8(vget_low_s8(vb));
        int16x8_t hi_b = vmovl_s8(vget_high_s8(vb));

        // Dot product
        int32x4_t dot_lo = vmull_s16(vget_low_s16(lo_a), vget_low_s16(lo_b));
        int32x4_t dot_hi = vmull_s16(vget_high_s16(lo_a), vget_high_s16(lo_b));
        int32x4_t dot_lo2 = vmull_s16(vget_low_s16(hi_a), vget_low_s16(hi_b));
        int32x4_t dot_hi2 = vmull_s16(vget_high_s16(hi_a), vget_high_s16(hi_b));

        acc_dot = vaddq_s32(acc_dot, dot_lo);
        acc_dot = vaddq_s32(acc_dot, dot_hi);
        acc_dot = vaddq_s32(acc_dot, dot_lo2);
        acc_dot = vaddq_s32(acc_dot, dot_hi2);

        // Norm a²
        int32x4_t a2_lo = vmull_s16(vget_low_s16(lo_a), vget_low_s16(lo_a));
        int32x4_t a2_hi = vmull_s16(vget_high_s16(lo_a), vget_high_s16(lo_a));
        int32x4_t a2_lo2 = vmull_s16(vget_low_s16(hi_a), vget_low_s16(hi_a));
        int32x4_t a2_hi2 = vmull_s16(vget_high_s16(hi_a), vget_high_s16(hi_a));

        acc_a2 = vaddq_s32(acc_a2, a2_lo);
        acc_a2 = vaddq_s32(acc_a2, a2_hi);
        acc_a2 = vaddq_s32(acc_a2, a2_lo2);
        acc_a2 = vaddq_s32(acc_a2, a2_hi2);

        // Norm b²
        int32x4_t b2_lo = vmull_s16(vget_low_s16(lo_b), vget_low_s16(lo_b));
        int32x4_t b2_hi = vmull_s16(vget_high_s16(lo_b), vget_high_s16(lo_b));
        int32x4_t b2_lo2 = vmull_s16(vget_low_s16(hi_b), vget_low_s16(hi_b));
        int32x4_t b2_hi2 = vmull_s16(vget_high_s16(hi_b), vget_high_s16(hi_b));

        acc_b2 = vaddq_s32(acc_b2, b2_lo);
        acc_b2 = vaddq_s32(acc_b2, b2_hi);
        acc_b2 = vaddq_s32(acc_b2, b2_lo2);
        acc_b2 = vaddq_s32(acc_b2, b2_hi2);
    }

    int32_t dot = vgetq_lane_s32(acc_dot, 0) + vgetq_lane_s32(acc_dot, 1)
                + vgetq_lane_s32(acc_dot, 2) + vgetq_lane_s32(acc_dot, 3);
    int32_t norm_a = vgetq_lane_s32(acc_a2, 0) + vgetq_lane_s32(acc_a2, 1)
                   + vgetq_lane_s32(acc_a2, 2) + vgetq_lane_s32(acc_a2, 3);
    int32_t norm_b = vgetq_lane_s32(acc_b2, 0) + vgetq_lane_s32(acc_b2, 1)
                   + vgetq_lane_s32(acc_b2, 2) + vgetq_lane_s32(acc_b2, 3);

    for (; i < n; ++i) {
        int ai = a[i];
        int bi = b[i];
        dot    += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if (norm_a == 0 || norm_b == 0) return 1.0f;
    return 1.0f - (dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b)));
}

float int8_distance_dot_neon (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;

    for (; i <= n - 16; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t lo_a = vmovl_s8(vget_low_s8(va));
        int16x8_t hi_a = vmovl_s8(vget_high_s8(va));
        int16x8_t lo_b = vmovl_s8(vget_low_s8(vb));
        int16x8_t hi_b = vmovl_s8(vget_high_s8(vb));

        int32x4_t prod_lo = vmull_s16(vget_low_s16(lo_a), vget_low_s16(lo_b));
        int32x4_t prod_hi = vmull_s16(vget_high_s16(lo_a), vget_high_s16(lo_b));
        int32x4_t prod_lo2 = vmull_s16(vget_low_s16(hi_a), vget_low_s16(hi_b));
        int32x4_t prod_hi2 = vmull_s16(vget_high_s16(hi_a), vget_high_s16(hi_b));

        acc = vaddq_s32(acc, prod_lo);
        acc = vaddq_s32(acc, prod_hi);
        acc = vaddq_s32(acc, prod_lo2);
        acc = vaddq_s32(acc, prod_hi2);
    }

    int32_t dot = vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1)
                + vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);

    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }

    return -(float)dot;  // negative dot product
}

float int8_distance_l1_neon(const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    uint32x4_t acc = vdupq_n_u32(0);
    int i = 0;

    for (; i <= n - 16; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        // Widen to 16-bit signed
        int16x8_t diff_lo = vsubl_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t diff_hi = vsubl_s8(vget_high_s8(va), vget_high_s8(vb));

        // Absolute values (safe for -128)
        int16x8_t abs_lo = vabsq_s16(diff_lo);
        int16x8_t abs_hi = vabsq_s16(diff_hi);

        // Widen to 32-bit and accumulate
        acc = vaddq_u32(acc, vmovl_u16(vget_low_u16(vreinterpretq_u16_s16(abs_lo))));
        acc = vaddq_u32(acc, vmovl_u16(vget_high_u16(vreinterpretq_u16_s16(abs_lo))));
        acc = vaddq_u32(acc, vmovl_u16(vget_low_u16(vreinterpretq_u16_s16(abs_hi))));
        acc = vaddq_u32(acc, vmovl_u16(vget_high_u16(vreinterpretq_u16_s16(abs_hi))));
    }

    // Horizontal sum
    uint64x2_t sum64 = vpaddlq_u32(acc);
    uint64_t final = vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);

    // Tail loop
    for (; i < n; ++i) {
        final += (uint32_t)abs((int)a[i] - (int)b[i]);
    }

    return (float)final;
}
#endif

// MARK: -

void init_distance_functions_neon (void) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_neon;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_neon;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_neon;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_neon;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_neon;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_neon;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_neon;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_neon;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_neon;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_neon;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_neon;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_neon;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_neon;
    
    distance_backend_name = "NEON";
#endif
}
