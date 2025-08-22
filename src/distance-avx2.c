//
//  distance-avx2.c
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#include "distance-avx2.h"
#include "distance-cpu.h"

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#include <stdint.h>
#include <math.h>

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern char *distance_backend_name;

#define _mm256_abs_ps(x) _mm256_andnot_ps(_mm256_set1_ps(-0.0f), (x))

static inline __m256 mm256_abs_ps(__m256 x) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    return _mm256_and_ps(x, mask);
}

static inline double hsum256d(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s, s);
    __m128d ss = _mm_add_sd(s, sh);
    return _mm_cvtsd_f64(ss);
}

// per-block Inf mismatch test on 8 lanes (returns true if L1/L2 should be +Inf)
static inline bool block_has_l2_inf_mismatch_8(const uint16_t *a, const uint16_t *b) {
    /* mismatch if (a_inf ^ b_inf) OR (both Inf and signs differ) */
    for (int k = 0; k < 8; ++k) {
        uint16_t ak = a[k], bk = b[k];
        bool ai = f16_is_inf(ak), bi = f16_is_inf(bk);
        if ((ai ^ bi) || (ai && bi && (f16_sign(ak) != f16_sign(bk)))) return true;
    }
    return false;
}

/* 8×bf16 -> 8×f32: widen to u32, shift <<16, reinterpret as f32 */
static inline __m256 bf16x8_to_f32x8_loadu(const uint16_t* p) {
    __m128i v16 = _mm_loadu_si128((const __m128i*)p);             // 8×u16
    __m256i v32 = _mm256_cvtepu16_epi32(v16);                     // 8×u32
    v32 = _mm256_slli_epi32(v32, 16);                             // <<16
    return _mm256_castsi256_ps(v32);                              // bitcast to f32
}

/* Any lane has infinite difference?  (a_inf ^ b_inf) || (both inf and signs differ) */
static inline bool block_has_l2_inf_mismatch_bf16_8(const uint16_t* a, const uint16_t* b) {
    for (int k = 0; k < 8; ++k) {
        uint16_t ak = a[k], bk = b[k];
        bool ai = bfloat16_is_inf(ak), bi = bfloat16_is_inf(bk);
        if ((ai ^ bi) || (ai && bi && (bfloat16_sign(ak) != bfloat16_sign(bk)))) return true;
    }
    return false;
}


// MARK: - FLOAT32 -

static inline float float32_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m256 acc = _mm256_setzero_ps();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }

    float temp[8];
    _mm256_storeu_ps(temp, acc);
    float total = temp[0] + temp[1] + temp[2] + temp[3] +
                  temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }

    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float float32_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_avx2(v1, v2, n, true);
}

float float32_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_avx2(v1, v2, n, false);
}

float float32_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m256 acc = _mm256_setzero_ps();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_abs_ps(diff));
    }

    float temp[8];
    _mm256_storeu_ps(temp, acc);
    float total = temp[0] + temp[1] + temp[2] + temp[3] +
                  temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        total += fabsf(a[i] - b[i]);
    }

    return total;
}

float float32_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    __m256 acc = _mm256_setzero_ps();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    float temp[8];
    _mm256_storeu_ps(temp, acc);
    float total = temp[0] + temp[1] + temp[2] + temp[3] +
                  temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        total += a[i] * b[i];
    }

    return -total;
}

float float32_distance_cosine_avx2 (const void *a, const void *b, int n) {
    float dot = -float32_distance_dot_avx2(a, b, n);
    float norm_a = sqrtf(-float32_distance_dot_avx2(a, a, n));
    float norm_b = sqrtf(-float32_distance_dot_avx2(b, b, n));

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (norm_a * norm_b);
    return 1.0f - cosine_similarity;
}

// MARK: - FLOAT16 -

static inline float float16_distance_l2_impl_avx2(const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m256d acc0 = _mm256_setzero_pd();  /* sum of squares for low 4 lanes */
    __m256d acc1 = _mm256_setzero_pd();  /* sum of squares for high 4 lanes */
    int i = 0;

    for (; i <= n - 8; i += 8) {
        /* Inf mismatch => distance is +Inf */
        if (block_has_l2_inf_mismatch_8(a + i, b + i)) return INFINITY;

        /* convert 8 f16 -> 8 f32 and zero-out NaN diffs */
        float diff_f32[8];
        for (int k = 0; k < 8; ++k) {
            uint16_t ak = a[i + k], bk = b[i + k];
            if (f16_is_nan(ak) || f16_is_nan(bk)) {
                diff_f32[k] = 0.0f;
            } else {
                float ax = float16_to_float32(ak);
                float bx = float16_to_float32(bk);
                float d  = ax - bx;
                diff_f32[k] = isnan(d) ? 0.0f : d;  /* defensive */
            }
        }

        __m256 d = _mm256_loadu_ps(diff_f32);
        /* widen to f64 and accumulate squares */
        __m128 lo = _mm256_castps256_ps128(d);
        __m128 hi = _mm256_extractf128_ps(d, 1);
        __m256d dlo = _mm256_cvtps_pd(lo);
        __m256d dhi = _mm256_cvtps_pd(hi);
#if defined(__FMA__)
        acc0 = _mm256_fmadd_pd(dlo, dlo, acc0);
        acc1 = _mm256_fmadd_pd(dhi, dhi, acc1);
#else
        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(dlo, dlo));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(dhi, dhi));
#endif
    }

    double sum = hsum256d(acc0) + hsum256d(acc1);

    /* scalar tail with same NaN/Inf policy */
    for (; i < n; ++i) {
        uint16_t ai = a[i], bi = b[i];
        if ((f16_is_inf(ai) || f16_is_inf(bi)) && !(f16_is_inf(ai) && f16_is_inf(bi) && f16_sign(ai) == f16_sign(bi))) return INFINITY;
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        double d = (double)float16_to_float32(ai) - (double)float16_to_float32(bi);
        sum = fma(d, d, sum);
    }

    return use_sqrt ? (float)sqrt(sum) : (float)sum;
}

float float16_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_avx2(v1, v2, n, true);
}

float float16_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_avx2(v1, v2, n, false);
}

float float16_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m256d acc = _mm256_setzero_pd();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        if (block_has_l2_inf_mismatch_8(a + i, b + i)) return INFINITY;

        float absdiff_f32[8];
        for (int k = 0; k < 8; ++k) {
            uint16_t ak = a[i + k], bk = b[i + k];
            if (f16_is_nan(ak) || f16_is_nan(bk)) {
                absdiff_f32[k] = 0.0f;
            } else {
                float ax = float16_to_float32(ak);
                float bx = float16_to_float32(bk);
                float d  = fabsf(ax - bx);
                absdiff_f32[k] = isnan(d) ? 0.0f : d;
            }
        }

        __m256 d = _mm256_loadu_ps(absdiff_f32);
        __m128 lo = _mm256_castps256_ps128(d);
        __m128 hi = _mm256_extractf128_ps(d, 1);
        __m256d dlo = _mm256_cvtps_pd(lo);
        __m256d dhi = _mm256_cvtps_pd(hi);
        acc = _mm256_add_pd(acc, dlo);
        acc = _mm256_add_pd(acc, dhi);
    }

    double sum = hsum256d(acc);

    for (; i < n; ++i) {
        uint16_t ai = a[i], bi = b[i];
        if ((f16_is_inf(ai) || f16_is_inf(bi)) && !(f16_is_inf(ai) && f16_is_inf(bi) && f16_sign(ai) == f16_sign(bi))) return INFINITY;
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        sum += fabs((double)float16_to_float32(ai) - (double)float16_to_float32(bi));
    }

    return (float)sum;
}

float float16_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        /* convert 8 f16 -> 8 f32; skip NaNs; detect Inf products */
        float prod_f32[8];
        for (int k = 0; k < 8; ++k) {
            uint16_t ak = a[i + k], bk = b[i + k];
            if (f16_is_nan(ak) || f16_is_nan(bk)) {
                prod_f32[k] = 0.0f;
                continue;
            }
            /* if any lane yields ±Inf product, return immediately with sign */
            bool ai = f16_is_inf(ak), bi = f16_is_inf(bk);
            if (ai || bi) {
                /* Infinity * zero -> NaN (ignore); else sign = sign(a) ^ sign(b) */
                if ((ai && f16_is_zero(bk)) || (bi && f16_is_zero(ak))) {
                    prod_f32[k] = 0.0f;  /* treat NaN as 0 contribution */
                } else {
                    int s = (f16_sign(ak) ^ f16_sign(bk)) ? -1 : +1;
                    return s < 0 ? INFINITY : -INFINITY; /* function returns -dot */
                }
            } else {
                float ax = float16_to_float32(ak);
                float bx = float16_to_float32(bk);
                float p  = ax * bx;
                if (isinf(p)) return (p > 0) ? -INFINITY : INFINITY;
                prod_f32[k] = isnan(p) ? 0.0f : p;
            }
        }

        __m256 p = _mm256_loadu_ps(prod_f32);
        __m128 lo = _mm256_castps256_ps128(p);
        __m128 hi = _mm256_extractf128_ps(p, 1);
        __m256d dlo = _mm256_cvtps_pd(lo);
        __m256d dhi = _mm256_cvtps_pd(hi);
        acc0 = _mm256_add_pd(acc0, dlo);
        acc1 = _mm256_add_pd(acc1, dhi);
    }

    double dot = hsum256d(acc0) + hsum256d(acc1);

    for (; i < n; ++i) {
        uint16_t ai = a[i], bi = b[i];
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        bool aiinf = f16_is_inf(ai), biinf = f16_is_inf(bi);
        if (aiinf || biinf) {
            if ((aiinf && f16_is_zero(bi)) || (biinf && f16_is_zero(ai))) {
                /* Inf * 0 -> NaN: ignore */
            } else {
                int s = (f16_sign(ai) ^ f16_sign(bi)) ? -1 : +1;
                return s < 0 ? INFINITY : -INFINITY;  /* returns -dot */
            }
        } else {
            float x = float16_to_float32(ai);
            float y = float16_to_float32(bi);
            double p = (double)x * (double)y;
            if (isinf(p)) return (p > 0) ? -INFINITY : INFINITY;
            if (!isnan(p)) dot += p;
        }
    }

    return (float)(-dot);
}

float float16_distance_cosine_avx2 (const void *va, const void *vb, int n) {
    const uint16_t *a = (const uint16_t *)va;
    const uint16_t *b = (const uint16_t *)vb;

    /* If either vector contains any ±Inf, return max distance */
    for (int i = 0; i < n; ++i) {
        if (f16_is_inf(a[i]) || f16_is_inf(b[i])) return 1.0f;
    }

    /* reuse dot for dot, and norms as sqrt(-dot(self,self)) */
    float dot    = -float16_distance_dot_avx2(a, b, n);
    float norm_a =  sqrtf(-float16_distance_dot_avx2(a, a, n));
    float norm_b =  sqrtf(-float16_distance_dot_avx2(b, b, n));

    if (!(norm_a > 0.0f) || !(norm_b > 0.0f) || !isfinite(norm_a) || !isfinite(norm_b) || !isfinite(dot))
        return 1.0f;

    float cosine = dot / (norm_a * norm_b);
    if (cosine > 1.0f)  cosine = 1.0f;
    if (cosine < -1.0f) cosine = -1.0f;
    return 1.0f - cosine;
}

// MARK: - BFLOAT16 -

static inline float bfloat16_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m256d acc0 = _mm256_setzero_pd();   // low 4 lanes
    __m256d acc1 = _mm256_setzero_pd();   // high 4 lanes
    int i = 0;

    for (; i <= n - 8; i += 8) {
        if (block_has_l2_inf_mismatch_bf16_8(a + i, b + i)) return INFINITY;

        __m256 af = bf16x8_to_f32x8_loadu(a + i);
        __m256 bf = bf16x8_to_f32x8_loadu(b + i);

        /* widen each half to f64 and subtract in f64 to avoid f32 overflow */
        __m128 af_lo = _mm256_castps256_ps128(af);
        __m128 af_hi = _mm256_extractf128_ps(af, 1);
        __m128 bf_lo = _mm256_castps256_ps128(bf);
        __m128 bf_hi = _mm256_extractf128_ps(bf, 1);

        __m256d a0 = _mm256_cvtps_pd(af_lo);
        __m256d a1 = _mm256_cvtps_pd(af_hi);
        __m256d b0 = _mm256_cvtps_pd(bf_lo);
        __m256d b1 = _mm256_cvtps_pd(bf_hi);

        __m256d d0 = _mm256_sub_pd(a0, b0);
        __m256d d1 = _mm256_sub_pd(a1, b1);

        /* zero-out NaNs: mask = (d==d) */
        __m256d z  = _mm256_setzero_pd();
        __m256d m0 = _mm256_cmp_pd(d0, d0, _CMP_ORD_Q);  // true if not NaN
        __m256d m1 = _mm256_cmp_pd(d1, d1, _CMP_ORD_Q);
        d0 = _mm256_blendv_pd(z, d0, m0);
        d1 = _mm256_blendv_pd(z, d1, m1);

    #if defined(__FMA__)
        acc0 = _mm256_fmadd_pd(d0, d0, acc0);
        acc1 = _mm256_fmadd_pd(d1, d1, acc1);
    #else
        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(d0, d0));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(d1, d1));
    #endif
    }

    double sum = hsum256d(acc0) + hsum256d(acc1);

    /* scalar tail */
    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((bfloat16_is_inf(ai) || bfloat16_is_inf(bi)) && !(bfloat16_is_inf(ai) && bfloat16_is_inf(bi) && bfloat16_sign(ai)==bfloat16_sign(bi))) return INFINITY;
        if (bfloat16_is_nan(ai) || bfloat16_is_nan(bi)) continue;
        double d = (double)bfloat16_to_float32(ai) - (double)bfloat16_to_float32(bi);
        sum = fma(d, d, sum);
    }

    return use_sqrt ? (float)sqrt(sum) : (float)sum;
}

float bfloat16_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_avx2(v1, v2, n, true);
}

float bfloat16_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_avx2(v1, v2, n, false);
}

float bfloat16_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m256d acc = _mm256_setzero_pd();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        if (block_has_l2_inf_mismatch_bf16_8(a + i, b + i)) return INFINITY;

        __m256 af = bf16x8_to_f32x8_loadu(a + i);
        __m256 bf = bf16x8_to_f32x8_loadu(b + i);

        __m128 af_lo = _mm256_castps256_ps128(af);
        __m128 af_hi = _mm256_extractf128_ps(af, 1);
        __m128 bf_lo = _mm256_castps256_ps128(bf);
        __m128 bf_hi = _mm256_extractf128_ps(bf, 1);

        __m256d a0 = _mm256_cvtps_pd(af_lo);
        __m256d a1 = _mm256_cvtps_pd(af_hi);
        __m256d b0 = _mm256_cvtps_pd(bf_lo);
        __m256d b1 = _mm256_cvtps_pd(bf_hi);

        __m256d d0 = _mm256_sub_pd(a0, b0);
        __m256d d1 = _mm256_sub_pd(a1, b1);

        /* |d| and NaN→0 */
        __m256d sign = _mm256_set1_pd(-0.0);
        d0 = _mm256_andnot_pd(sign, d0);
        d1 = _mm256_andnot_pd(sign, d1);
        __m256d z  = _mm256_setzero_pd();
        __m256d m0 = _mm256_cmp_pd(d0, d0, _CMP_ORD_Q);
        __m256d m1 = _mm256_cmp_pd(d1, d1, _CMP_ORD_Q);
        d0 = _mm256_blendv_pd(z, d0, m0);
        d1 = _mm256_blendv_pd(z, d1, m1);

        acc = _mm256_add_pd(acc, d0);
        acc = _mm256_add_pd(acc, d1);
    }

    /* sum */
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s, s);
    __m128d ss = _mm_add_sd(s, sh);
    double sum = _mm_cvtsd_f64(ss);

    /* tail */
    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((bfloat16_is_inf(ai) || bfloat16_is_inf(bi)) && !(bfloat16_is_inf(ai) && bfloat16_is_inf(bi) && bfloat16_sign(ai)==bfloat16_sign(bi))) return INFINITY;
        if (bfloat16_is_nan(ai) || bfloat16_is_nan(bi)) continue;
        sum += fabs((double)bfloat16_to_float32(ai) - (double)bfloat16_to_float32(bi));
    }

    return (float)sum;
}

float bfloat16_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int i = 0;

    for (; i <= n - 8; i += 8) {
        /* quick Inf product check per-lane */
        for (int k=0;k<8;++k){
            uint16_t ak=a[i+k], bk=b[i+k];
            bool ai=bfloat16_is_inf(ak), bi=bfloat16_is_inf(bk);
            if (ai || bi) {
                if ((ai && bfloat16_is_zero(bk)) || (bi && bfloat16_is_zero(ak))) {
                    /* Inf * 0 ⇒ NaN: ignore lane */
                    continue;
                } else {
                    int s = (bfloat16_sign(ak) ^ bfloat16_sign(bk)) ? -1 : +1;
                    return s < 0 ? INFINITY : -INFINITY;   /* function returns -dot */
                }
            }
        }

        __m256 af = bf16x8_to_f32x8_loadu(a + i);
        __m256 bf = bf16x8_to_f32x8_loadu(b + i);

        /* zero-out NaNs before multiply */
        __m256 mask_a = _mm256_cmp_ps(af, af, _CMP_ORD_Q);
        __m256 mask_b = _mm256_cmp_ps(bf, bf, _CMP_ORD_Q);
        af = _mm256_blendv_ps(_mm256_setzero_ps(), af, mask_a);
        bf = _mm256_blendv_ps(_mm256_setzero_ps(), bf, mask_b);

        __m256 prod = _mm256_mul_ps(af, bf);

        __m128 lo = _mm256_castps256_ps128(prod);
        __m128 hi = _mm256_extractf128_ps(prod, 1);
        __m256d d0 = _mm256_cvtps_pd(lo);
        __m256d d1 = _mm256_cvtps_pd(hi);

        acc0 = _mm256_add_pd(acc0, d0);
        acc1 = _mm256_add_pd(acc1, d1);
    }

    /* sum */
    __m128d lo = _mm256_castpd256_pd128(acc0);
    __m128d hi = _mm256_extractf128_pd(acc0, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s, s);
    double dot = _mm_cvtsd_f64(_mm_add_sd(s, sh));
    lo = _mm256_castpd256_pd128(acc1);
    hi = _mm256_extractf128_pd(acc1, 1);
    s  = _mm_add_pd(lo, hi);
    sh = _mm_unpackhi_pd(s, s);
    dot += _mm_cvtsd_f64(_mm_add_sd(s, sh));

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if (bfloat16_is_nan(ai) || bfloat16_is_nan(bi)) continue;
        bool aiinf=bfloat16_is_inf(ai), biinf=bfloat16_is_inf(bi);
        if (aiinf || biinf) {
            if ((aiinf && bfloat16_is_zero(bi)) || (biinf && bfloat16_is_zero(ai))) {
                /* Inf*0 -> NaN: ignore */
            } else {
                int sgn = (bfloat16_sign(ai) ^ bfloat16_sign(bi)) ? -1 : +1;
                return sgn < 0 ? INFINITY : -INFINITY;      /* returns -dot */
            }
        } else {
            double p = (double)bfloat16_to_float32(ai) * (double)bfloat16_to_float32(bi);
            dot += p;
        }
    }

    return (float)(-dot);
}

float bfloat16_distance_cosine_avx2 (const void *v1, const void *v2, int n) {
    /* reuse dot routine like your original float32 version */
    float dot    = -bfloat16_distance_dot_avx2(v1, v2, n);
    float norm_a =  sqrtf(-bfloat16_distance_dot_avx2(v1, v1, n));
    float norm_b =  sqrtf(-bfloat16_distance_dot_avx2(v2, v2, n));

    if (!(norm_a > 0.0f) || !(norm_b > 0.0f) || !isfinite(norm_a) || !isfinite(norm_b) || !isfinite(dot))
        return 1.0f;

    float cs = dot / (norm_a * norm_b);
    if (cs >  1.0f) cs =  1.0f;
    if (cs < -1.0f) cs = -1.0f;
    return 1.0f - cs;
}

// MARK: - UINT8 -

static inline float uint8_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    
    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        
        // Split into 2 x 128-bit chunks
        __m128i va_lo = _mm256_extracti128_si256(va, 0);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_extracti128_si256(vb, 0);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);
        
        // Unpack to 16-bit integers
        __m128i va_lo_u16 = _mm_unpacklo_epi8(va_lo, _mm_setzero_si128());
        __m128i va_hi_u16 = _mm_unpackhi_epi8(va_lo, _mm_setzero_si128());
        __m128i vb_lo_u16 = _mm_unpacklo_epi8(vb_lo, _mm_setzero_si128());
        __m128i vb_hi_u16 = _mm_unpackhi_epi8(vb_lo, _mm_setzero_si128());
        
        __m128i va_lo_u16_hi = _mm_unpacklo_epi8(va_hi, _mm_setzero_si128());
        __m128i va_hi_u16_hi = _mm_unpackhi_epi8(va_hi, _mm_setzero_si128());
        __m128i vb_lo_u16_hi = _mm_unpacklo_epi8(vb_hi, _mm_setzero_si128());
        __m128i vb_hi_u16_hi = _mm_unpackhi_epi8(vb_hi, _mm_setzero_si128());
        
        // Compute diffs
        __m128i d0 = _mm_sub_epi16(va_lo_u16, vb_lo_u16);
        __m128i d1 = _mm_sub_epi16(va_hi_u16, vb_hi_u16);
        __m128i d2 = _mm_sub_epi16(va_lo_u16_hi, vb_lo_u16_hi);
        __m128i d3 = _mm_sub_epi16(va_hi_u16_hi, vb_hi_u16_hi);
        
        // Square diffs
        __m128i s0 = _mm_mullo_epi16(d0, d0);
        __m128i s1 = _mm_mullo_epi16(d1, d1);
        __m128i s2 = _mm_mullo_epi16(d2, d2);
        __m128i s3 = _mm_mullo_epi16(d3, d3);
        
        // Widen to 32-bit and accumulate
        __m256i w0 = _mm256_cvtepu16_epi32(s0);
        __m256i w1 = _mm256_cvtepu16_epi32(s1);
        __m256i w2 = _mm256_cvtepu16_epi32(s2);
        __m256i w3 = _mm256_cvtepu16_epi32(s3);
        
        acc = _mm256_add_epi32(acc, w0);
        acc = _mm256_add_epi32(acc, w1);
        acc = _mm256_add_epi32(acc, w2);
        acc = _mm256_add_epi32(acc, w3);
    }
    
    // Horizontal sum of 8 x 32-bit integers
    uint32_t temp[8];
    _mm256_storeu_si256((__m256i *)temp, acc);
    uint32_t total = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    
    // Tail loop
    for (; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        total += d * d;
    }
    
    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float uint8_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_avx2(v1, v2, n, true);
}

float uint8_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_avx2(v1, v2, n, false);
}

float uint8_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m256i acc = _mm256_setzero_si256();
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        __m256i a_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        __m256i a_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        __m256i b_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        __m256i b_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        __m256i prod_lo = _mm256_mullo_epi16(a_lo, b_lo);
        __m256i prod_hi = _mm256_mullo_epi16(a_hi, b_hi);

        __m256i prod_lo_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_lo, 0));
        __m256i prod_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_lo, 1));
        acc = _mm256_add_epi32(acc, prod_lo_32);
        acc = _mm256_add_epi32(acc, prod_hi_32);

        prod_lo_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_hi, 0));
        prod_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_hi, 1));
        acc = _mm256_add_epi32(acc, prod_lo_32);
        acc = _mm256_add_epi32(acc, prod_hi_32);
    }

    uint32_t temp[8];
    _mm256_storeu_si256((__m256i *)temp, acc);
    uint32_t total = temp[0] + temp[1] + temp[2] + temp[3] +
                     temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        total += a[i] * b[i];
    }

    return -(float)total;
}

float uint8_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    __m256i acc = _mm256_setzero_si256();
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        __m256i a_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        __m256i a_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        __m256i b_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        __m256i b_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        __m256i diff_lo = _mm256_abs_epi16(_mm256_sub_epi16(a_lo, b_lo));
        __m256i diff_hi = _mm256_abs_epi16(_mm256_sub_epi16(a_hi, b_hi));

        __m256i diff_lo_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(diff_lo, 0));
        __m256i diff_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(diff_lo, 1));
        acc = _mm256_add_epi32(acc, diff_lo_32);
        acc = _mm256_add_epi32(acc, diff_hi_32);

        diff_lo_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(diff_hi, 0));
        diff_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(diff_hi, 1));
        acc = _mm256_add_epi32(acc, diff_lo_32);
        acc = _mm256_add_epi32(acc, diff_hi_32);
    }

    uint32_t temp[8];
    _mm256_storeu_si256((__m256i *)temp, acc);
    uint32_t total = temp[0] + temp[1] + temp[2] + temp[3] +
                     temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        total += abs((int)a[i] - (int)b[i]);
    }

    return (float)total;
}

float uint8_distance_cosine_avx2 (const void *a, const void *b, int n) {
    float dot = -uint8_distance_dot_avx2(a, b, n);
    float norm_a = sqrtf(-uint8_distance_dot_avx2(a, a, n));
    float norm_b = sqrtf(-uint8_distance_dot_avx2(b, b, n));

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (norm_a * norm_b);
    return 1.0f - cosine_similarity;
}

// MARK: - INT8 -

static inline float int8_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    __m256i acc = _mm256_setzero_si256();
    int i = 0;

    for (; i <= n - 32; i += 32) {
        // Load 32 int8_t elements from each input
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        // Extract 128-bit halves
        __m128i va_lo = _mm256_extracti128_si256(va, 0);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_extracti128_si256(vb, 0);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);

        // Sign-extend int8_t to int16_t
        __m128i va_lo_s16 = _mm_cvtepi8_epi16(va_lo);
        __m128i va_hi_s16 = _mm_cvtepi8_epi16(_mm_srli_si128(va_lo, 8));
        __m128i vb_lo_s16 = _mm_cvtepi8_epi16(vb_lo);
        __m128i vb_hi_s16 = _mm_cvtepi8_epi16(_mm_srli_si128(vb_lo, 8));

        __m128i va_lo_s16_hi = _mm_cvtepi8_epi16(va_hi);
        __m128i va_hi_s16_hi = _mm_cvtepi8_epi16(_mm_srli_si128(va_hi, 8));
        __m128i vb_lo_s16_hi = _mm_cvtepi8_epi16(vb_hi);
        __m128i vb_hi_s16_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vb_hi, 8));

        // Compute differences
        __m128i d0 = _mm_sub_epi16(va_lo_s16, vb_lo_s16);
        __m128i d1 = _mm_sub_epi16(va_hi_s16, vb_hi_s16);
        __m128i d2 = _mm_sub_epi16(va_lo_s16_hi, vb_lo_s16_hi);
        __m128i d3 = _mm_sub_epi16(va_hi_s16_hi, vb_hi_s16_hi);

        // Square differences
        __m128i s0 = _mm_mullo_epi16(d0, d0);
        __m128i s1 = _mm_mullo_epi16(d1, d1);
        __m128i s2 = _mm_mullo_epi16(d2, d2);
        __m128i s3 = _mm_mullo_epi16(d3, d3);

        // Extend to 32-bit and accumulate
        __m256i w0 = _mm256_cvtepu16_epi32(s0);
        __m256i w1 = _mm256_cvtepu16_epi32(s1);
        __m256i w2 = _mm256_cvtepu16_epi32(s2);
        __m256i w3 = _mm256_cvtepu16_epi32(s3);

        acc = _mm256_add_epi32(acc, w0);
        acc = _mm256_add_epi32(acc, w1);
        acc = _mm256_add_epi32(acc, w2);
        acc = _mm256_add_epi32(acc, w3);
    }

    // Horizontal sum
    uint32_t temp[8];
    _mm256_storeu_si256((__m256i *)temp, acc);
    uint32_t total = temp[0] + temp[1] + temp[2] + temp[3] +
                     temp[4] + temp[5] + temp[6] + temp[7];

    // Scalar tail
    for (; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        total += d * d;
    }

    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float int8_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_avx2(v1, v2, n, true);
}

float int8_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_avx2(v1, v2, n, false);
}

float int8_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    __m256i acc = _mm256_setzero_si256();
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        __m128i va_lo = _mm256_extracti128_si256(va, 0);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_extracti128_si256(vb, 0);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);

        __m128i a0 = _mm_cvtepi8_epi16(va_lo);
        __m128i a1 = _mm_cvtepi8_epi16(_mm_srli_si128(va_lo, 8));
        __m128i b0 = _mm_cvtepi8_epi16(vb_lo);
        __m128i b1 = _mm_cvtepi8_epi16(_mm_srli_si128(vb_lo, 8));

        __m128i a2 = _mm_cvtepi8_epi16(va_hi);
        __m128i a3 = _mm_cvtepi8_epi16(_mm_srli_si128(va_hi, 8));
        __m128i b2 = _mm_cvtepi8_epi16(vb_hi);
        __m128i b3 = _mm_cvtepi8_epi16(_mm_srli_si128(vb_hi, 8));

        __m128i p0 = _mm_mullo_epi16(a0, b0);
        __m128i p1 = _mm_mullo_epi16(a1, b1);
        __m128i p2 = _mm_mullo_epi16(a2, b2);
        __m128i p3 = _mm_mullo_epi16(a3, b3);

        __m256i w0 = _mm256_cvtepi16_epi32(p0);
        __m256i w1 = _mm256_cvtepi16_epi32(p1);
        __m256i w2 = _mm256_cvtepi16_epi32(p2);
        __m256i w3 = _mm256_cvtepi16_epi32(p3);

        acc = _mm256_add_epi32(acc, w0);
        acc = _mm256_add_epi32(acc, w1);
        acc = _mm256_add_epi32(acc, w2);
        acc = _mm256_add_epi32(acc, w3);
    }

    uint32_t temp[8];
    _mm256_storeu_si256((__m256i *)temp, acc);
    int32_t total = temp[0] + temp[1] + temp[2] + temp[3] +
                    temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        total += (int)a[i] * (int)b[i];
    }

    return -(float)total;
}

float int8_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    __m256i acc = _mm256_setzero_si256();
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));

        __m128i va_lo = _mm256_extracti128_si256(va, 0);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_extracti128_si256(vb, 0);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);

        __m128i a0 = _mm_cvtepi8_epi16(va_lo);
        __m128i a1 = _mm_cvtepi8_epi16(_mm_srli_si128(va_lo, 8));
        __m128i b0 = _mm_cvtepi8_epi16(vb_lo);
        __m128i b1 = _mm_cvtepi8_epi16(_mm_srli_si128(vb_lo, 8));

        __m128i a2 = _mm_cvtepi8_epi16(va_hi);
        __m128i a3 = _mm_cvtepi8_epi16(_mm_srli_si128(va_hi, 8));
        __m128i b2 = _mm_cvtepi8_epi16(vb_hi);
        __m128i b3 = _mm_cvtepi8_epi16(_mm_srli_si128(vb_hi, 8));

        __m128i d0 = _mm_abs_epi16(_mm_sub_epi16(a0, b0));
        __m128i d1 = _mm_abs_epi16(_mm_sub_epi16(a1, b1));
        __m128i d2 = _mm_abs_epi16(_mm_sub_epi16(a2, b2));
        __m128i d3 = _mm_abs_epi16(_mm_sub_epi16(a3, b3));

        __m256i w0 = _mm256_cvtepu16_epi32(d0);
        __m256i w1 = _mm256_cvtepu16_epi32(d1);
        __m256i w2 = _mm256_cvtepu16_epi32(d2);
        __m256i w3 = _mm256_cvtepu16_epi32(d3);

        acc = _mm256_add_epi32(acc, w0);
        acc = _mm256_add_epi32(acc, w1);
        acc = _mm256_add_epi32(acc, w2);
        acc = _mm256_add_epi32(acc, w3);
    }

    uint32_t temp[8];
    _mm256_storeu_si256((__m256i *)temp, acc);
    int32_t total = temp[0] + temp[1] + temp[2] + temp[3] +
                    temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i) {
        total += abs((int)a[i] - (int)b[i]);
    }

    return (float)total;
}

float int8_distance_cosine_avx2 (const void *a, const void *b, int n) {
    float dot = -int8_distance_dot_avx2(a, b, n);
    float norm_a = sqrtf(-int8_distance_dot_avx2(a, a, n));
    float norm_b = sqrtf(-int8_distance_dot_avx2(b, b, n));

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (norm_a * norm_b);
    return 1.0f - cosine_similarity;
}

#endif

// MARK: -

void init_distance_functions_avx2 (void) {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_avx2;
    
    distance_backend_name = "AVX2";
#endif
}
