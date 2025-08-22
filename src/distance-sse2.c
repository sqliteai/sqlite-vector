//
//  distance-sse2.c
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#include "distance-sse2.h"
#include "distance-cpu.h"

#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || (_M_IX86_FP >= 2)))
#include <emmintrin.h>
#include <stdint.h>
#include <math.h>

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern char *distance_backend_name;

// accumulate 32-bit
#define ACCUMULATE(MUL, ACC)                    \
    acc_tmp = _mm_unpacklo_epi16(MUL, _mm_setzero_si128()); \
    ACC = _mm_add_epi32(ACC, acc_tmp);          \
    acc_tmp = _mm_unpackhi_epi16(MUL, _mm_setzero_si128()); \
    ACC = _mm_add_epi32(ACC, acc_tmp);

// proper sign-extension from int16_t to int32_t
#define SIGN_EXTEND_EPI16_TO_EPI32_LO(v) \
    _mm_srai_epi32(_mm_unpacklo_epi16(_mm_slli_epi32((v), 16), (v)), 16)

#define SIGN_EXTEND_EPI16_TO_EPI32_HI(v) \
    _mm_srai_epi32(_mm_unpackhi_epi16(_mm_slli_epi32((v), 16), (v)), 16)

static inline double hsum128d(__m128d v) {
    __m128d sh = _mm_unpackhi_pd(v, v);
    __m128d s  = _mm_add_sd(v, sh);
    return _mm_cvtsd_f64(s);
}

static inline __m128d mm_abs_pd(__m128d x) {
    const __m128d sign = _mm_set1_pd(-0.0);
    return _mm_andnot_pd(sign, x);
}

static inline __m128 f16x4_to_f32x4_loadu(const uint16_t* p) {
    float tmp[4];
    tmp[0] = float16_to_float32(p[0]);
    tmp[1] = float16_to_float32(p[1]);
    tmp[2] = float16_to_float32(p[2]);
    tmp[3] = float16_to_float32(p[3]);
    return _mm_loadu_ps(tmp);
}

/* load 4 bf16 -> __m128 of f32 */
static inline __m128 bf16x4_to_f32x4_loadu(const uint16_t* p) {
    float tmp[4];
    tmp[0] = bfloat16_to_float32(p[0]);
    tmp[1] = bfloat16_to_float32(p[1]);
    tmp[2] = bfloat16_to_float32(p[2]);
    tmp[3] = bfloat16_to_float32(p[3]);
    return _mm_loadu_ps(tmp);
}


// MARK: - FLOAT32 -

static inline float float32_distance_l2_impl_sse2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m128 acc = _mm_setzero_ps();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 sq = _mm_mul_ps(diff, diff);
        acc = _mm_add_ps(acc, sq);
    }

    float partial[4];
    _mm_storeu_ps(partial, acc);
    float total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }

    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float float32_distance_l2_sse2 (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_sse2(v1, v2, n, true);
}

float float32_distance_l2_squared_sse2 (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_sse2(v1, v2, n, false);
}

float float32_distance_l1_sse2 (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m128 acc = _mm_setzero_ps();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff); // abs using bitmask
        acc = _mm_add_ps(acc, abs_diff);
    }

    float partial[4];
    _mm_storeu_ps(partial, acc);
    float total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        total += fabsf(a[i] - b[i]);
    }

    return total;
}

float float32_distance_dot_sse2 (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m128 acc = _mm_setzero_ps();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(va, vb);
        acc = _mm_add_ps(acc, prod);
    }

    float partial[4];
    _mm_storeu_ps(partial, acc);
    float total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        total += a[i] * b[i];
    }

    return -total;
}

float float32_distance_cosine_sse2 (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m128 acc_dot = _mm_setzero_ps();
    __m128 acc_a2 = _mm_setzero_ps();
    __m128 acc_b2 = _mm_setzero_ps();

    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);

        acc_dot = _mm_add_ps(acc_dot, _mm_mul_ps(va, vb));
        acc_a2  = _mm_add_ps(acc_a2, _mm_mul_ps(va, va));
        acc_b2  = _mm_add_ps(acc_b2, _mm_mul_ps(vb, vb));
    }

    float dot[4], a2[4], b2[4];
    _mm_storeu_ps(dot, acc_dot);
    _mm_storeu_ps(a2, acc_a2);
    _mm_storeu_ps(b2, acc_b2);

    float total_dot = dot[0] + dot[1] + dot[2] + dot[3];
    float total_a2  = a2[0] + a2[1] + a2[2] + a2[3];
    float total_b2  = b2[0] + b2[1] + b2[2] + b2[3];

    for (; i < n; ++i) {
        total_dot += a[i] * b[i];
        total_a2  += a[i] * a[i];
        total_b2  += b[i] * b[i];
    }

    float denom = sqrtf(total_a2 * total_b2);
    if (denom == 0.0f) return 1.0f;
    float cosine_sim = total_dot / denom;
    return 1.0f - cosine_sim;
}

// MARK: - FLOAT16 -

static inline float float16_distance_l2_impl_sse2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        /* Inf mismatch? (a_inf ^ b_inf) || (both inf with different signs) */
        for (int k=0;k<4;++k) {
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=f16_is_inf(ak), bi=f16_is_inf(bk);
            if ((ai ^ bi) || (ai && bi && (f16_sign(ak) != f16_sign(bk))))
                return INFINITY;
        }

        __m128 af = f16x4_to_f32x4_loadu(a + i);
        __m128 bf = f16x4_to_f32x4_loadu(b + i);

        /* widen to f64 and subtract in f64 to avoid f32 overflow */
        __m128d a_lo = _mm_cvtps_pd(af);
        __m128d b_lo = _mm_cvtps_pd(bf);
        __m128  af_hi4 = _mm_movehl_ps(af, af);  /* [a3 a2 .. ..] */
        __m128  bf_hi4 = _mm_movehl_ps(bf, bf);
        __m128d a_hi = _mm_cvtps_pd(af_hi4);
        __m128d b_hi = _mm_cvtps_pd(bf_hi4);

        __m128d d0 = _mm_sub_pd(a_lo, b_lo);
        __m128d d1 = _mm_sub_pd(a_hi, b_hi);

        /* zero-out NaNs: m = (d==d) */
        __m128d m0 = _mm_cmpeq_pd(d0, d0);
        __m128d m1 = _mm_cmpeq_pd(d1, d1);
        d0 = _mm_and_pd(d0, m0);
        d1 = _mm_and_pd(d1, m1);

        acc0 = _mm_add_pd(acc0, _mm_mul_pd(d0, d0));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(d1, d1));
    }

    double sum = hsum128d(acc0) + hsum128d(acc1);

    /* tail */
    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((f16_is_inf(ai) || f16_is_inf(bi)) &&
            !(f16_is_inf(ai) && f16_is_inf(bi) && f16_sign(ai)==f16_sign(bi)))
            return INFINITY;
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        double d = (double)float16_to_float32(ai) - (double)float16_to_float32(bi);
        sum = fma(d, d, sum);
    }

    return use_sqrt ? (float)sqrt(sum) : (float)sum;
}
float float16_distance_l2_sse2 (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_sse2(v1, v2, n, true);
}

float float16_distance_l2_squared_sse2 (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_sse2(v1, v2, n, false);
}

float float16_distance_l1_sse2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m128d acc = _mm_setzero_pd();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        for (int k=0;k<4;++k) {
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=f16_is_inf(ak), bi=f16_is_inf(bk);
            if ((ai ^ bi) || (ai && bi && (f16_sign(ak) != f16_sign(bk))))
                return INFINITY;
        }

        __m128 af = f16x4_to_f32x4_loadu(a + i);
        __m128 bf = f16x4_to_f32x4_loadu(b + i);

        __m128d a_lo = _mm_cvtps_pd(af);
        __m128d b_lo = _mm_cvtps_pd(bf);
        __m128  af_hi4 = _mm_movehl_ps(af, af);
        __m128  bf_hi4 = _mm_movehl_ps(bf, bf);
        __m128d a_hi = _mm_cvtps_pd(af_hi4);
        __m128d b_hi = _mm_cvtps_pd(bf_hi4);

        __m128d d0 = mm_abs_pd(_mm_sub_pd(a_lo, b_lo));
        __m128d d1 = mm_abs_pd(_mm_sub_pd(a_hi, b_hi));

        __m128d m0 = _mm_cmpeq_pd(d0, d0);
        __m128d m1 = _mm_cmpeq_pd(d1, d1);
        d0 = _mm_and_pd(d0, m0);
        d1 = _mm_and_pd(d1, m1);

        acc = _mm_add_pd(acc, d0);
        acc = _mm_add_pd(acc, d1);
    }

    double sum = hsum128d(acc);

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((f16_is_inf(ai) || f16_is_inf(bi)) &&
            !(f16_is_inf(ai) && f16_is_inf(bi) && f16_sign(ai)==f16_sign(bi)))
            return INFINITY;
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        sum += fabs((double)float16_to_float32(ai) - (double)float16_to_float32(bi));
    }

    return (float)sum;
}

float float16_distance_dot_sse2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        /* Handle Inf*X lanes up-front for correct ±Inf sign */
        for (int k=0;k<4;++k) {
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=f16_is_inf(ak), bi=f16_is_inf(bk);
            if (ai || bi) {
                if ((ai && f16_is_zero(bk)) || (bi && f16_is_zero(ak))) {
                    /* Inf * 0 => NaN: ignore lane */
                } else {
                    int s = (f16_sign(ak) ^ f16_sign(bk)) ? -1 : +1;
                    return s < 0 ? INFINITY : -INFINITY; /* function returns -dot */
                }
            }
        }

        __m128 af = f16x4_to_f32x4_loadu(a + i);
        __m128 bf = f16x4_to_f32x4_loadu(b + i);

        /* zero-out NaNs */
        __m128 mask_a = _mm_cmpeq_ps(af, af);
        __m128 mask_b = _mm_cmpeq_ps(bf, bf);
        af = _mm_and_ps(af, mask_a);
        bf = _mm_and_ps(bf, mask_b);

        /* widen and accumulate in f64 */
        __m128d a_lo = _mm_cvtps_pd(af);
        __m128d b_lo = _mm_cvtps_pd(bf);
        __m128  af_hi4 = _mm_movehl_ps(af, af);
        __m128  bf_hi4 = _mm_movehl_ps(bf, bf);
        __m128d a_hi = _mm_cvtps_pd(af_hi4);
        __m128d b_hi = _mm_cvtps_pd(bf_hi4);

        acc0 = _mm_add_pd(acc0, _mm_mul_pd(a_lo, b_lo));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(a_hi, b_hi));
    }

    double dot = hsum128d(acc0) + hsum128d(acc1);

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        bool aiinf=f16_is_inf(ai), biinf=f16_is_inf(bi);
        if (aiinf || biinf) {
            if ((aiinf && f16_is_zero(bi)) || (biinf && f16_is_zero(ai))) {
                /* Inf*0 -> NaN: ignore */
            } else {
                int s = (f16_sign(ai) ^ f16_sign(bi)) ? -1 : +1;
                return s < 0 ? INFINITY : -INFINITY; /* returns -dot */
            }
        } else {
            dot += (double)float16_to_float32(ai) * (double)float16_to_float32(bi);
        }
    }

    return (float)(-dot);
}

float float16_distance_cosine_sse2 (const void *v1, const void *v2, int n) {
    float dot    = -float16_distance_dot_sse2(v1, v2, n);
    float norm_a =  sqrtf(-float16_distance_dot_sse2(v1, v1, n));
    float norm_b =  sqrtf(-float16_distance_dot_sse2(v2, v2, n));
    if (!(norm_a > 0.0f) || !(norm_b > 0.0f) || !isfinite(norm_a) || !isfinite(norm_b) || !isfinite(dot))
        return 1.0f;
    float cs = dot / (norm_a * norm_b);
    if (cs > 1.0f) cs = 1.0f; else if (cs < -1.0f) cs = -1.0f;
    return 1.0f - cs;
}

// MARK: - BFLOAT16 -

static inline float bfloat16_distance_l2_impl_sse2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        for (int k=0;k<4;++k) {
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=bfloat16_is_inf(ak), bi=bfloat16_is_inf(bk);
            if ((ai ^ bi) || (ai && bi && (bfloat16_sign(ak) != bfloat16_sign(bk))))
                return INFINITY;
        }

        __m128 af = bf16x4_to_f32x4_loadu(a + i);
        __m128 bf = bf16x4_to_f32x4_loadu(b + i);

        __m128d a_lo = _mm_cvtps_pd(af);
        __m128d b_lo = _mm_cvtps_pd(bf);
        __m128  af_hi4 = _mm_movehl_ps(af, af);
        __m128  bf_hi4 = _mm_movehl_ps(bf, bf);
        __m128d a_hi = _mm_cvtps_pd(af_hi4);
        __m128d b_hi = _mm_cvtps_pd(bf_hi4);

        __m128d d0 = _mm_sub_pd(a_lo, b_lo);
        __m128d d1 = _mm_sub_pd(a_hi, b_hi);

        __m128d m0 = _mm_cmpeq_pd(d0, d0);
        __m128d m1 = _mm_cmpeq_pd(d1, d1);
        d0 = _mm_and_pd(d0, m0);
        d1 = _mm_and_pd(d1, m1);

        acc0 = _mm_add_pd(acc0, _mm_mul_pd(d0, d0));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(d1, d1));
    }

    double sum = hsum128d(acc0) + hsum128d(acc1);

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((bfloat16_is_inf(ai) || bfloat16_is_inf(bi)) &&
            !(bfloat16_is_inf(ai) && bfloat16_is_inf(bi) && bfloat16_sign(ai)==bfloat16_sign(bi)))
            return INFINITY;
        if (bfloat16_is_nan(ai) || bfloat16_is_nan(bi)) continue;
        double d = (double)bfloat16_to_float32(ai) - (double)bfloat16_to_float32(bi);
        sum = fma(d, d, sum);
    }

    return use_sqrt ? (float)sqrt(sum) : (float)sum;
}
float bfloat16_distance_l2_sse2 (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_sse2(v1, v2, n, true);
}

float bfloat16_distance_l2_squared_sse2 (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_sse2(v1, v2, n, false);
}

float bfloat16_distance_l1_sse2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m128d acc = _mm_setzero_pd();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        for (int k=0;k<4;++k) {
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=bfloat16_is_inf(ak), bi=bfloat16_is_inf(bk);
            if ((ai ^ bi) || (ai && bi && (bfloat16_sign(ak) != bfloat16_sign(bk))))
                return INFINITY;
        }

        __m128 af = bf16x4_to_f32x4_loadu(a + i);
        __m128 bf = bf16x4_to_f32x4_loadu(b + i);

        __m128d a_lo = _mm_cvtps_pd(af);
        __m128d b_lo = _mm_cvtps_pd(bf);
        __m128  af_hi4 = _mm_movehl_ps(af, af);
        __m128  bf_hi4 = _mm_movehl_ps(bf, bf);
        __m128d a_hi = _mm_cvtps_pd(af_hi4);
        __m128d b_hi = _mm_cvtps_pd(bf_hi4);

        __m128d d0 = mm_abs_pd(_mm_sub_pd(a_lo, b_lo));
        __m128d d1 = mm_abs_pd(_mm_sub_pd(a_hi, b_hi));

        __m128d m0 = _mm_cmpeq_pd(d0, d0);
        __m128d m1 = _mm_cmpeq_pd(d1, d1);
        d0 = _mm_and_pd(d0, m0);
        d1 = _mm_and_pd(d1, m1);

        acc = _mm_add_pd(acc, d0);
        acc = _mm_add_pd(acc, d1);
    }

    double sum = hsum128d(acc);

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((bfloat16_is_inf(ai) || bfloat16_is_inf(bi)) &&
            !(bfloat16_is_inf(ai) && bfloat16_is_inf(bi) && bfloat16_sign(ai)==bfloat16_sign(bi)))
            return INFINITY;
        if (bfloat16_is_nan(ai) || bfloat16_is_nan(bi)) continue;
        sum += fabs((double)bfloat16_to_float32(ai) - (double)bfloat16_to_float32(bi));
    }

    return (float)sum;
}

float bfloat16_distance_dot_sse2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    int i = 0;

    for (; i <= n - 4; i += 4) {
        /* handle Inf*X per-lane for correct sign */
        for (int k=0;k<4;++k){
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=bfloat16_is_inf(ak), bi=bfloat16_is_inf(bk);
            if (ai || bi) {
                if ((ai && bfloat16_is_zero(bk)) || (bi && bfloat16_is_zero(ak))) {
                    /* Inf*0 -> NaN: ignore lane */
                } else {
                    int s = (bfloat16_sign(ak) ^ bfloat16_sign(bk)) ? -1 : +1;
                    return s < 0 ? INFINITY : -INFINITY;  /* function returns -dot */
                }
            }
        }

        __m128 af = bf16x4_to_f32x4_loadu(a + i);
        __m128 bf = bf16x4_to_f32x4_loadu(b + i);

        /* zero NaNs */
        __m128 ma = _mm_cmpeq_ps(af, af);
        __m128 mb = _mm_cmpeq_ps(bf, bf);
        af = _mm_and_ps(af, ma);
        bf = _mm_and_ps(bf, mb);

        /* widen and accumulate in f64 */
        __m128d a_lo = _mm_cvtps_pd(af);
        __m128d b_lo = _mm_cvtps_pd(bf);
        __m128  af_hi4 = _mm_movehl_ps(af, af);
        __m128  bf_hi4 = _mm_movehl_ps(bf, bf);
        __m128d a_hi = _mm_cvtps_pd(af_hi4);
        __m128d b_hi = _mm_cvtps_pd(bf_hi4);

        acc0 = _mm_add_pd(acc0, _mm_mul_pd(a_lo, b_lo));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(a_hi, b_hi));
    }

    double dot = hsum128d(acc0) + hsum128d(acc1);

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if (bfloat16_is_nan(ai) || bfloat16_is_nan(bi)) continue;
        bool aiinf=bfloat16_is_inf(ai), biinf=bfloat16_is_inf(bi);
        if (aiinf || biinf) {
            if ((aiinf && bfloat16_is_zero(bi)) || (biinf && bfloat16_is_zero(ai))) {
                /* Inf*0 -> NaN: ignore */
            } else {
                int s = (bfloat16_sign(ai) ^ bfloat16_sign(bi)) ? -1 : +1;
                return s < 0 ? INFINITY : -INFINITY;  /* returns -dot */
            }
        } else {
            dot += (double)bfloat16_to_float32(ai) * (double)bfloat16_to_float32(bi);
        }
    }

    return (float)(-dot);
}

float bfloat16_distance_cosine_sse2 (const void *v1, const void *v2, int n) {
    float dot    = -bfloat16_distance_dot_sse2(v1, v2, n);
    float norm_a =  sqrtf(-bfloat16_distance_dot_sse2(v1, v1, n));
    float norm_b =  sqrtf(-bfloat16_distance_dot_sse2(v2, v2, n));
    if (!(norm_a > 0.0f) || !(norm_b > 0.0f) || !isfinite(norm_a) || !isfinite(norm_b) || !isfinite(dot)) return 1.0f;
    float cs = dot / (norm_a * norm_b);
    if (cs > 1.0f) cs = 1.0f; else if (cs < -1.0f) cs = -1.0f;
    return 1.0f - cs;
}

// MARK: - UINT8 -

static inline float uint8_distance_l2_impl_sse2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m128i acc = _mm_setzero_si128();  // 4 x 32-bit accumulator
    int i = 0;
    
    // process 16 bytes per iteration
    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));
        
        // unpack to 16-bit integers
        __m128i va_lo = _mm_unpacklo_epi8(va, _mm_setzero_si128()); // Lower 8 bytes to u16
        __m128i vb_lo = _mm_unpacklo_epi8(vb, _mm_setzero_si128());
        
        __m128i va_hi = _mm_unpackhi_epi8(va, _mm_setzero_si128()); // Upper 8 bytes to u16
        __m128i vb_hi = _mm_unpackhi_epi8(vb, _mm_setzero_si128());
        
        // compute differences
        __m128i diff_lo = _mm_sub_epi16(va_lo, vb_lo);
        __m128i diff_hi = _mm_sub_epi16(va_hi, vb_hi);
        
        // square differences
        __m128i mul_lo = _mm_mullo_epi16(diff_lo, diff_lo);
        __m128i mul_hi = _mm_mullo_epi16(diff_hi, diff_hi);
        
        // accumulate using widening add
        __m128i sum_32_lo = _mm_unpacklo_epi16(mul_lo, _mm_setzero_si128());
        __m128i sum_32_hi = _mm_unpackhi_epi16(mul_lo, _mm_setzero_si128());
        acc = _mm_add_epi32(acc, sum_32_lo);
        acc = _mm_add_epi32(acc, sum_32_hi);
        
        sum_32_lo = _mm_unpacklo_epi16(mul_hi, _mm_setzero_si128());
        sum_32_hi = _mm_unpackhi_epi16(mul_hi, _mm_setzero_si128());
        acc = _mm_add_epi32(acc, sum_32_lo);
        acc = _mm_add_epi32(acc, sum_32_hi);
    }
    
    // horizontal add the 4 lanes of acc
    int32_t partial[4];
    _mm_storeu_si128((__m128i *)partial, acc);
    int32_t total = partial[0] + partial[1] + partial[2] + partial[3];
    
    // handle remaining elements
    for (; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        total += diff * diff;
    }
    
    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float uint8_distance_l2_sse2 (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_sse2(v1, v2, n, true);
}

float uint8_distance_l2_squared_sse2 (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_sse2(v1, v2, n, false);
}

float uint8_distance_dot_sse2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m128i acc = _mm_setzero_si128();
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        __m128i va_lo = _mm_unpacklo_epi8(va, _mm_setzero_si128());
        __m128i vb_lo = _mm_unpacklo_epi8(vb, _mm_setzero_si128());

        __m128i va_hi = _mm_unpackhi_epi8(va, _mm_setzero_si128());
        __m128i vb_hi = _mm_unpackhi_epi8(vb, _mm_setzero_si128());

        __m128i mul_lo = _mm_mullo_epi16(va_lo, vb_lo);
        __m128i mul_hi = _mm_mullo_epi16(va_hi, vb_hi);

        __m128i sum_lo = _mm_unpacklo_epi16(mul_lo, _mm_setzero_si128());
        __m128i sum_hi = _mm_unpackhi_epi16(mul_lo, _mm_setzero_si128());
        acc = _mm_add_epi32(acc, sum_lo);
        acc = _mm_add_epi32(acc, sum_hi);

        sum_lo = _mm_unpacklo_epi16(mul_hi, _mm_setzero_si128());
        sum_hi = _mm_unpackhi_epi16(mul_hi, _mm_setzero_si128());
        acc = _mm_add_epi32(acc, sum_lo);
        acc = _mm_add_epi32(acc, sum_hi);
    }

    int32_t partial[4];
    _mm_storeu_si128((__m128i *)partial, acc);
    int32_t total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        total += (int)a[i] * (int)b[i];
    }

    return -(float)total;
}

float uint8_distance_l1_sse2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m128i acc = _mm_setzero_si128();
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        __m128i va_lo = _mm_unpacklo_epi8(va, _mm_setzero_si128());
        __m128i vb_lo = _mm_unpacklo_epi8(vb, _mm_setzero_si128());

        __m128i va_hi = _mm_unpackhi_epi8(va, _mm_setzero_si128());
        __m128i vb_hi = _mm_unpackhi_epi8(vb, _mm_setzero_si128());

        __m128i diff_lo = _mm_sub_epi16(va_lo, vb_lo);
        __m128i diff_hi = _mm_sub_epi16(va_hi, vb_hi);

        diff_lo = _mm_sub_epi16(_mm_max_epi16(va_lo, vb_lo), _mm_min_epi16(va_lo, vb_lo));
        diff_hi = _mm_sub_epi16(_mm_max_epi16(va_hi, vb_hi), _mm_min_epi16(va_hi, vb_hi));
        
        // SEE3+ instructions
        // diff_lo = _mm_abs_epi16(diff_lo);
        // diff_hi = _mm_abs_epi16(diff_hi);

        __m128i sum_lo = _mm_unpacklo_epi16(diff_lo, _mm_setzero_si128());
        __m128i sum_hi = _mm_unpackhi_epi16(diff_lo, _mm_setzero_si128());
        acc = _mm_add_epi32(acc, sum_lo);
        acc = _mm_add_epi32(acc, sum_hi);

        sum_lo = _mm_unpacklo_epi16(diff_hi, _mm_setzero_si128());
        sum_hi = _mm_unpackhi_epi16(diff_hi, _mm_setzero_si128());
        acc = _mm_add_epi32(acc, sum_lo);
        acc = _mm_add_epi32(acc, sum_hi);
    }

    int32_t partial[4];
    _mm_storeu_si128((__m128i *)partial, acc);
    int32_t total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        total += abs((int)a[i] - (int)b[i]);
    }

    return (float)total;
}

float uint8_distance_cosine_sse2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m128i acc_dot = _mm_setzero_si128();
    __m128i acc_a2 = _mm_setzero_si128();
    __m128i acc_b2 = _mm_setzero_si128();

    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        __m128i va_lo = _mm_unpacklo_epi8(va, _mm_setzero_si128());
        __m128i vb_lo = _mm_unpacklo_epi8(vb, _mm_setzero_si128());

        __m128i va_hi = _mm_unpackhi_epi8(va, _mm_setzero_si128());
        __m128i vb_hi = _mm_unpackhi_epi8(vb, _mm_setzero_si128());

        // dot product
        __m128i mul_dot_lo = _mm_mullo_epi16(va_lo, vb_lo);
        __m128i mul_dot_hi = _mm_mullo_epi16(va_hi, vb_hi);

        // a^2
        __m128i mul_a2_lo = _mm_mullo_epi16(va_lo, va_lo);
        __m128i mul_a2_hi = _mm_mullo_epi16(va_hi, va_hi);

        // b^2
        __m128i mul_b2_lo = _mm_mullo_epi16(vb_lo, vb_lo);
        __m128i mul_b2_hi = _mm_mullo_epi16(vb_hi, vb_hi);

        __m128i acc_tmp;

        ACCUMULATE(mul_dot_lo, acc_dot);
        ACCUMULATE(mul_dot_hi, acc_dot);

        ACCUMULATE(mul_a2_lo, acc_a2);
        ACCUMULATE(mul_a2_hi, acc_a2);

        ACCUMULATE(mul_b2_lo, acc_b2);
        ACCUMULATE(mul_b2_hi, acc_b2);
    }

    int32_t dot[4], a2[4], b2[4];
    _mm_storeu_si128((__m128i *)dot, acc_dot);
    _mm_storeu_si128((__m128i *)a2, acc_a2);
    _mm_storeu_si128((__m128i *)b2, acc_b2);

    int32_t total_dot = dot[0] + dot[1] + dot[2] + dot[3];
    int32_t total_a2  = a2[0] + a2[1] + a2[2] + a2[3];
    int32_t total_b2  = b2[0] + b2[1] + b2[2] + b2[3];

    for (; i < n; ++i) {
        int va = (int)a[i];
        int vb = (int)b[i];
        total_dot += va * vb;
        total_a2  += va * va;
        total_b2  += vb * vb;
    }

    float denom = sqrtf((float)total_a2 * (float)total_b2);
    if (denom == 0.0f) return 1.0f; // orthogonal or zero
    float cosine_sim = total_dot / denom;
    return 1.0f - cosine_sim; // cosine distance
}

// MARK: - INT8 -

// SSE2 does not support 8-bit integer multiplication directly
// Unpack to 16-bit signed integers
// Multiply using _mm_mullo_epi16, and accumulate in 32-bit lanes

static inline float int8_distance_l2_impl_sse2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    __m128i acc = _mm_setzero_si128();
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        // unpack to 16-bit signed integers
        __m128i va_lo = _mm_unpacklo_epi8(va, _mm_cmpgt_epi8(_mm_setzero_si128(), va));
        __m128i vb_lo = _mm_unpacklo_epi8(vb, _mm_cmpgt_epi8(_mm_setzero_si128(), vb));
        __m128i va_hi = _mm_unpackhi_epi8(va, _mm_cmpgt_epi8(_mm_setzero_si128(), va));
        __m128i vb_hi = _mm_unpackhi_epi8(vb, _mm_cmpgt_epi8(_mm_setzero_si128(), vb));

        // compute (a - b)
        __m128i diff_lo = _mm_sub_epi16(va_lo, vb_lo);
        __m128i diff_hi = _mm_sub_epi16(va_hi, vb_hi);

        // square differences
        __m128i sq_lo = _mm_mullo_epi16(diff_lo, diff_lo);
        __m128i sq_hi = _mm_mullo_epi16(diff_hi, diff_hi);

        // widen and accumulate
        acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(sq_lo, _mm_setzero_si128()));
        acc = _mm_add_epi32(acc, _mm_unpackhi_epi16(sq_lo, _mm_setzero_si128()));
        acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(sq_hi, _mm_setzero_si128()));
        acc = _mm_add_epi32(acc, _mm_unpackhi_epi16(sq_hi, _mm_setzero_si128()));
    }

    int32_t partial[4];
    _mm_storeu_si128((__m128i *)partial, acc);
    int32_t total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        int diff = (int)a[i] - (int)b[i];
        total += diff * diff;
    }

    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float int8_distance_l2_sse2 (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_sse2(v1, v2, n, true);
}

float int8_distance_l2_squared_sse2 (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_sse2(v1, v2, n, false);
}

float int8_distance_dot_sse2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    __m128i acc = _mm_setzero_si128();
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        // Manual sign-extension: int8_t → int16_t
        __m128i zero = _mm_setzero_si128();
        __m128i va_sign = _mm_cmpgt_epi8(zero, va);
        __m128i vb_sign = _mm_cmpgt_epi8(zero, vb);

        __m128i va_lo = _mm_unpacklo_epi8(va, va_sign);
        __m128i va_hi = _mm_unpackhi_epi8(va, va_sign);
        __m128i vb_lo = _mm_unpacklo_epi8(vb, vb_sign);
        __m128i vb_hi = _mm_unpackhi_epi8(vb, vb_sign);

        // Multiply int16 × int16 → int16 (overflow-safe because dot products are small)
        __m128i mul_lo = _mm_mullo_epi16(va_lo, vb_lo);
        __m128i mul_hi = _mm_mullo_epi16(va_hi, vb_hi);

        // Correct signed extension: int16 → int32
        acc = _mm_add_epi32(acc, SIGN_EXTEND_EPI16_TO_EPI32_LO(mul_lo));
        acc = _mm_add_epi32(acc, SIGN_EXTEND_EPI16_TO_EPI32_HI(mul_lo));
        acc = _mm_add_epi32(acc, SIGN_EXTEND_EPI16_TO_EPI32_LO(mul_hi));
        acc = _mm_add_epi32(acc, SIGN_EXTEND_EPI16_TO_EPI32_HI(mul_hi));
    }

    int32_t partial[4];
    _mm_storeu_si128((__m128i *)partial, acc);
    int32_t total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        total += (int)a[i] * (int)b[i];
    }

    return -(float)total;
}

float int8_distance_l1_sse2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    __m128i acc = _mm_setzero_si128();
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        __m128i va_lo = _mm_unpacklo_epi8(va, _mm_cmpgt_epi8(_mm_setzero_si128(), va));
        __m128i vb_lo = _mm_unpacklo_epi8(vb, _mm_cmpgt_epi8(_mm_setzero_si128(), vb));
        __m128i va_hi = _mm_unpackhi_epi8(va, _mm_cmpgt_epi8(_mm_setzero_si128(), va));
        __m128i vb_hi = _mm_unpackhi_epi8(vb, _mm_cmpgt_epi8(_mm_setzero_si128(), vb));

        __m128i diff_lo = _mm_sub_epi16(va_lo, vb_lo);
        __m128i diff_hi = _mm_sub_epi16(va_hi, vb_hi);

        // Absolute value via max/min since _mm_abs_epi16 is SSE3+
        diff_lo = _mm_sub_epi16(_mm_max_epi16(diff_lo, _mm_sub_epi16(_mm_setzero_si128(), diff_lo)),
                                _mm_setzero_si128());
        diff_hi = _mm_sub_epi16(_mm_max_epi16(diff_hi, _mm_sub_epi16(_mm_setzero_si128(), diff_hi)),
                                _mm_setzero_si128());

        acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(diff_lo, _mm_setzero_si128()));
        acc = _mm_add_epi32(acc, _mm_unpackhi_epi16(diff_lo, _mm_setzero_si128()));
        acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(diff_hi, _mm_setzero_si128()));
        acc = _mm_add_epi32(acc, _mm_unpackhi_epi16(diff_hi, _mm_setzero_si128()));
    }

    int32_t partial[4];
    _mm_storeu_si128((__m128i *)partial, acc);
    int32_t total = partial[0] + partial[1] + partial[2] + partial[3];

    for (; i < n; ++i) {
        total += abs((int)a[i] - (int)b[i]);
    }

    return (float)total;
}

float int8_distance_cosine_sse2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    __m128i acc_dot = _mm_setzero_si128();
    __m128i acc_a2  = _mm_setzero_si128();
    __m128i acc_b2  = _mm_setzero_si128();

    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        // Manual sign extension from int8_t → int16_t
        __m128i zero = _mm_setzero_si128();
        __m128i va_sign = _mm_cmpgt_epi8(zero, va);
        __m128i vb_sign = _mm_cmpgt_epi8(zero, vb);

        __m128i va_lo = _mm_unpacklo_epi8(va, va_sign);  // lower 8 int8_t → int16_t
        __m128i va_hi = _mm_unpackhi_epi8(va, va_sign);  // upper 8 int8_t → int16_t
        __m128i vb_lo = _mm_unpacklo_epi8(vb, vb_sign);
        __m128i vb_hi = _mm_unpackhi_epi8(vb, vb_sign);

        // Multiply and accumulate
        __m128i dot_lo = _mm_mullo_epi16(va_lo, vb_lo);
        __m128i dot_hi = _mm_mullo_epi16(va_hi, vb_hi);
        __m128i a2_lo  = _mm_mullo_epi16(va_lo, va_lo);
        __m128i a2_hi  = _mm_mullo_epi16(va_hi, va_hi);
        __m128i b2_lo  = _mm_mullo_epi16(vb_lo, vb_lo);
        __m128i b2_hi  = _mm_mullo_epi16(vb_hi, vb_hi);

        // Unpack 16-bit to 32-bit and accumulate
        acc_dot = _mm_add_epi32(acc_dot, SIGN_EXTEND_EPI16_TO_EPI32_LO(dot_lo));
        acc_dot = _mm_add_epi32(acc_dot, SIGN_EXTEND_EPI16_TO_EPI32_HI(dot_lo));
        acc_dot = _mm_add_epi32(acc_dot, SIGN_EXTEND_EPI16_TO_EPI32_LO(dot_hi));
        acc_dot = _mm_add_epi32(acc_dot, SIGN_EXTEND_EPI16_TO_EPI32_HI(dot_hi));

        acc_a2 = _mm_add_epi32(acc_a2, SIGN_EXTEND_EPI16_TO_EPI32_LO(a2_lo));
        acc_a2 = _mm_add_epi32(acc_a2, SIGN_EXTEND_EPI16_TO_EPI32_HI(a2_lo));
        acc_a2 = _mm_add_epi32(acc_a2, SIGN_EXTEND_EPI16_TO_EPI32_LO(a2_hi));
        acc_a2 = _mm_add_epi32(acc_a2, SIGN_EXTEND_EPI16_TO_EPI32_HI(a2_hi));

        acc_b2 = _mm_add_epi32(acc_b2, SIGN_EXTEND_EPI16_TO_EPI32_LO(b2_lo));
        acc_b2 = _mm_add_epi32(acc_b2, SIGN_EXTEND_EPI16_TO_EPI32_HI(b2_lo));
        acc_b2 = _mm_add_epi32(acc_b2, SIGN_EXTEND_EPI16_TO_EPI32_LO(b2_hi));
        acc_b2 = _mm_add_epi32(acc_b2, SIGN_EXTEND_EPI16_TO_EPI32_HI(b2_hi));
    }

    // Horizontal sum of SIMD accumulators
    int32_t _d[4], _a[4], _b[4];
    _mm_storeu_si128((__m128i *)_d, acc_dot);
    _mm_storeu_si128((__m128i *)_a, acc_a2);
    _mm_storeu_si128((__m128i *)_b, acc_b2);

    int32_t total_dot = _d[0] + _d[1] + _d[2] + _d[3];
    int32_t total_a2  = _a[0] + _a[1] + _a[2] + _a[3];
    int32_t total_b2  = _b[0] + _b[1] + _b[2] + _b[3];

    // Handle tail
    for (; i < n; ++i) {
        int va = a[i];
        int vb = b[i];
        total_dot += va * vb;
        total_a2  += va * va;
        total_b2  += vb * vb;
    }

    float denom = sqrtf((float)total_a2 * (float)total_b2);
    if (denom == 0.0f) return 1.0f;
    float cosine_sim = total_dot / denom;
    return 1.0f - cosine_sim;
}

#endif

// MARK: -

void init_distance_functions_sse2 (void) {
#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || (_M_IX86_FP >= 2)))
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_sse2;
    
    distance_backend_name = "SSE2";
#endif
}
