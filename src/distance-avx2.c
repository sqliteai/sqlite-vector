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
extern const char *distance_backend_name;
extern turbo_lut_dot_function_t turbo_lut_dot_function;
extern const char *turbo_lut_backend_name;

#define _mm256_abs_ps(x) _mm256_andnot_ps(_mm256_set1_ps(-0.0f), (x))

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

// A single accumulator makes the loop one dependency chain: an FMA has around four cycles
// of latency, so it retires one vector every four cycles however many FMA ports the core
// has. Four independent accumulators keep them fed and still fit sixteen YMM registers.
#if defined(__FMA__)
#define MM256_FMA_PS(_acc, _x, _y)          _mm256_fmadd_ps((_x), (_y), (_acc))
#else
#define MM256_FMA_PS(_acc, _x, _y)          _mm256_add_ps((_acc), _mm256_mul_ps((_x), (_y)))
#endif

static inline float hsum256_ps (__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

static inline float float32_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i     ), _mm256_loadu_ps(b + i     ));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = MM256_FMA_PS(acc0, d0, d0);
        acc1 = MM256_FMA_PS(acc1, d1, d1);
        acc2 = MM256_FMA_PS(acc2, d2, d2);
        acc3 = MM256_FMA_PS(acc3, d3, d3);
    }
    for (; i <= n - 8; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = MM256_FMA_PS(acc0, d, d);
    }

    float total = hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));

    for (; i < n; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }

    return use_sqrt ? sqrtf(total) : total;
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

    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i     ), _mm256_loadu_ps(b + i     ));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_add_ps(acc0, _mm256_abs_ps(d0));
        acc1 = _mm256_add_ps(acc1, _mm256_abs_ps(d1));
        acc2 = _mm256_add_ps(acc2, _mm256_abs_ps(d2));
        acc3 = _mm256_add_ps(acc3, _mm256_abs_ps(d3));
    }
    for (; i <= n - 8; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_add_ps(acc0, _mm256_abs_ps(d));
    }

    float total = hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));

    for (; i < n; ++i) {
        total += fabsf(a[i] - b[i]);
    }

    return total;
}

float float32_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int i = 0;

    for (; i <= n - 32; i += 32) {
        acc0 = MM256_FMA_PS(acc0, _mm256_loadu_ps(a + i     ), _mm256_loadu_ps(b + i     ));
        acc1 = MM256_FMA_PS(acc1, _mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8));
        acc2 = MM256_FMA_PS(acc2, _mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        acc3 = MM256_FMA_PS(acc3, _mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
    }
    for (; i <= n - 8; i += 8) {
        acc0 = MM256_FMA_PS(acc0, _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    }

    float total = hsum256_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));

    for (; i < n; ++i) {
        total += a[i] * b[i];
    }

    return -total;
}

float float32_distance_cosine_avx2 (const void *a, const void *b, int n) {
    const float *x = (const float *)a;
    const float *y = (const float *)b;

    // one fused pass, not three calls to the dot kernel: the data is read once instead of
    // three times, which is what actually costs on anything larger than L1
    __m256 dot0 = _mm256_setzero_ps(), dot1 = dot0;
    __m256 na0 = dot0, na1 = dot0;
    __m256 nb0 = dot0, nb1 = dot0;
    int i = 0;

    for (; i <= n - 16; i += 16) {
        __m256 a0 = _mm256_loadu_ps(x + i), a1 = _mm256_loadu_ps(x + i + 8);
        __m256 b0 = _mm256_loadu_ps(y + i), b1 = _mm256_loadu_ps(y + i + 8);
        dot0 = MM256_FMA_PS(dot0, a0, b0);  dot1 = MM256_FMA_PS(dot1, a1, b1);
        na0  = MM256_FMA_PS(na0,  a0, a0);  na1  = MM256_FMA_PS(na1,  a1, a1);
        nb0  = MM256_FMA_PS(nb0,  b0, b0);  nb1  = MM256_FMA_PS(nb1,  b1, b1);
    }
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(x + i), vb = _mm256_loadu_ps(y + i);
        dot0 = MM256_FMA_PS(dot0, va, vb);
        na0  = MM256_FMA_PS(na0,  va, va);
        nb0  = MM256_FMA_PS(nb0,  vb, vb);
    }

    float dot = hsum256_ps(_mm256_add_ps(dot0, dot1));
    float norm_a = hsum256_ps(_mm256_add_ps(na0, na1));
    float norm_b = hsum256_ps(_mm256_add_ps(nb0, nb1));

    for (; i < n; ++i) {
        float ai = x[i];
        float bi = y[i];
        dot    += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (sqrtf(norm_a) * sqrtf(norm_b));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
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

// The integer kernels used to widen every byte to 32 bits before multiplying, on a single
// accumulator chain. x86 has instructions for exactly this shape: PSADBW sums absolute
// differences of 32 bytes in one go (L1), and PMADDWD multiplies 16-bit pairs and adds
// adjacent products into 32-bit lanes, which is a squared difference or a dot product
// depending on what you feed it. Two byte-sized factors always fit 16 bits, and PMADDWD's
// pairwise add keeps the running value inside 32 bits.
//
// The signed kernels bias by 0x80 where the arithmetic allows it: that maps int8 onto
// uint8 without changing any difference between two elements, so L2 and L1 become the
// unsigned kernel plus one XOR per vector. Dot and cosine need the true signed values.
static inline __m256i abs_diff_epu8 (__m256i a, __m256i b) {
    return _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
}

static inline uint64_t hsum256_epi64 (__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi64(lo, hi);
    return (uint64_t)_mm_cvtsi128_si64(s) + (uint64_t)_mm_extract_epi64(s, 1);
}

// The reductions widen to 64 bits before summing the lanes. Each lane is itself a running
// total, so folding eight of them in 32 bits would cap the usable dimension far below
// what the per-lane accumulators can hold.
static inline uint64_t hsum256_epi32 (__m256i v) {
    const __m256i zero = _mm256_setzero_si256();
    return hsum256_epi64(_mm256_add_epi64(_mm256_unpacklo_epi32(v, zero), _mm256_unpackhi_epi32(v, zero)));
}

static inline int64_t hsum256_epi32_signed (__m256i v) {
    __m256i wl = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
    __m256i wh = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
    __m256i s = _mm256_add_epi64(wl, wh);
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i r = _mm_add_epi64(lo, hi);
    return (int64_t)_mm_cvtsi128_si64(r) + (int64_t)_mm_extract_epi64(r, 1);
}

// sum of squared differences of 32 bytes, accumulated 32 bits wide
static inline __m256i sqdiff_epu8 (__m256i a, __m256i b) {
    __m256i d = abs_diff_epu8(a, b);
    __m256i lo = _mm256_unpacklo_epi8(d, _mm256_setzero_si256());
    __m256i hi = _mm256_unpackhi_epi8(d, _mm256_setzero_si256());
    return _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
}

static inline float uint8_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;

    for (; i <= n - 64; i += 64) {
        acc0 = _mm256_add_epi32(acc0, sqdiff_epu8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                                  _mm256_loadu_si256((const __m256i *)(b + i))));
        acc1 = _mm256_add_epi32(acc1, sqdiff_epu8(_mm256_loadu_si256((const __m256i *)(a + i + 32)),
                                                  _mm256_loadu_si256((const __m256i *)(b + i + 32))));
    }
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_add_epi32(acc0, sqdiff_epu8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                                  _mm256_loadu_si256((const __m256i *)(b + i))));
    }

    uint64_t total = hsum256_epi32(_mm256_add_epi32(acc0, acc1));

    for (; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        total += (uint64_t)(d * d);
    }

    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float uint8_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_avx2(v1, v2, n, true);
}

float uint8_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_avx2(v1, v2, n, false);
}

// widening dot product of 32 unsigned bytes
static inline __m256i dot_epu8 (__m256i a, __m256i b) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i al = _mm256_unpacklo_epi8(a, zero), ah = _mm256_unpackhi_epi8(a, zero);
    __m256i bl = _mm256_unpacklo_epi8(b, zero), bh = _mm256_unpackhi_epi8(b, zero);
    return _mm256_add_epi32(_mm256_madd_epi16(al, bl), _mm256_madd_epi16(ah, bh));
}

float uint8_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;

    for (; i <= n - 64; i += 64) {
        acc0 = _mm256_add_epi32(acc0, dot_epu8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                               _mm256_loadu_si256((const __m256i *)(b + i))));
        acc1 = _mm256_add_epi32(acc1, dot_epu8(_mm256_loadu_si256((const __m256i *)(a + i + 32)),
                                               _mm256_loadu_si256((const __m256i *)(b + i + 32))));
    }
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_add_epi32(acc0, dot_epu8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                               _mm256_loadu_si256((const __m256i *)(b + i))));
    }

    uint64_t dot = hsum256_epi32(_mm256_add_epi32(acc0, acc1));

    for (; i < n; ++i) dot += (uint64_t)((uint32_t)a[i] * (uint32_t)b[i]);

    return -(float)dot;
}

float uint8_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // PSADBW already sums absolute differences into 64-bit lanes: one instruction per
    // 32 bytes, and it cannot overflow for any vector length we support
    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;

    for (; i <= n - 64; i += 64) {
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                                      _mm256_loadu_si256((const __m256i *)(b + i))));
        acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(a + i + 32)),
                                                      _mm256_loadu_si256((const __m256i *)(b + i + 32))));
    }
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                                      _mm256_loadu_si256((const __m256i *)(b + i))));
    }

    uint64_t sum = hsum256_epi64(_mm256_add_epi64(acc0, acc1));

    for (; i < n; ++i) sum += (uint64_t)abs((int)a[i] - (int)b[i]);

    return (float)sum;
}

float uint8_distance_cosine_avx2 (const void *a, const void *b, int n) {
    const uint8_t *x = (const uint8_t *)a;
    const uint8_t *y = (const uint8_t *)b;

    // one fused pass rather than three calls to the dot kernel
    __m256i dacc = _mm256_setzero_si256(), aacc = dacc, bacc = dacc;
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(y + i));
        dacc = _mm256_add_epi32(dacc, dot_epu8(va, vb));
        aacc = _mm256_add_epi32(aacc, dot_epu8(va, va));
        bacc = _mm256_add_epi32(bacc, dot_epu8(vb, vb));
    }

    uint64_t dot = hsum256_epi32(dacc);
    uint64_t norm_a = hsum256_epi32(aacc);
    uint64_t norm_b = hsum256_epi32(bacc);

    for (; i < n; ++i) {
        uint32_t p = x[i], q = y[i];
        dot += (uint64_t)(p * q);
        norm_a += (uint64_t)(p * p);
        norm_b += (uint64_t)(q * q);
    }

    if (norm_a == 0 || norm_b == 0) return 1.0f;

    float cosine_similarity = (float)((double)dot / (sqrt((double)norm_a) * sqrt((double)norm_b)));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - INT8 -

#define S8_TO_BIASED_U8_AVX2(_v)            _mm256_xor_si256((_v), _mm256_set1_epi8((char)0x80))

static inline float int8_distance_l2_impl_avx2 (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;

    for (; i <= n - 64; i += 64) {
        acc0 = _mm256_add_epi32(acc0, sqdiff_epu8(S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(a + i))),
                                                  S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(b + i)))));
        acc1 = _mm256_add_epi32(acc1, sqdiff_epu8(S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(a + i + 32))),
                                                  S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(b + i + 32)))));
    }
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_add_epi32(acc0, sqdiff_epu8(S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(a + i))),
                                                  S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(b + i)))));
    }

    uint64_t total = hsum256_epi32(_mm256_add_epi32(acc0, acc1));

    for (; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        total += (uint64_t)(d * d);
    }

    return use_sqrt ? sqrtf((float)total) : (float)total;
}

float int8_distance_l2_avx2 (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_avx2(v1, v2, n, true);
}

float int8_distance_l2_squared_avx2 (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_avx2(v1, v2, n, false);
}

// widening dot product of 32 signed bytes: sign-extend each half, then PMADDWD
static inline __m256i dot_epi8 (__m256i a, __m256i b) {
    __m256i al = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
    __m256i ah = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
    __m256i bl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
    __m256i bh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
    return _mm256_add_epi32(_mm256_madd_epi16(al, bl), _mm256_madd_epi16(ah, bh));
}

float int8_distance_dot_avx2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;

    for (; i <= n - 64; i += 64) {
        acc0 = _mm256_add_epi32(acc0, dot_epi8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                               _mm256_loadu_si256((const __m256i *)(b + i))));
        acc1 = _mm256_add_epi32(acc1, dot_epi8(_mm256_loadu_si256((const __m256i *)(a + i + 32)),
                                               _mm256_loadu_si256((const __m256i *)(b + i + 32))));
    }
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_add_epi32(acc0, dot_epi8(_mm256_loadu_si256((const __m256i *)(a + i)),
                                               _mm256_loadu_si256((const __m256i *)(b + i))));
    }

    int64_t dot = hsum256_epi32_signed(_mm256_add_epi32(acc0, acc1));

    for (; i < n; ++i) dot += (int64_t)((int32_t)a[i] * (int32_t)b[i]);

    return -(float)dot;
}

float int8_distance_l1_avx2 (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // same biasing trick as L2, so PSADBW applies unchanged
    __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
    int i = 0;

    for (; i <= n - 64; i += 64) {
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(a + i))),
                                                      S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(b + i)))));
        acc1 = _mm256_add_epi64(acc1, _mm256_sad_epu8(S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(a + i + 32))),
                                                      S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(b + i + 32)))));
    }
    for (; i <= n - 32; i += 32) {
        acc0 = _mm256_add_epi64(acc0, _mm256_sad_epu8(S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(a + i))),
                                                      S8_TO_BIASED_U8_AVX2(_mm256_loadu_si256((const __m256i *)(b + i)))));
    }

    uint64_t sum = hsum256_epi64(_mm256_add_epi64(acc0, acc1));

    for (; i < n; ++i) sum += (uint64_t)abs((int)a[i] - (int)b[i]);

    return (float)sum;
}

float int8_distance_cosine_avx2 (const void *a, const void *b, int n) {
    const int8_t *x = (const int8_t *)a;
    const int8_t *y = (const int8_t *)b;

    __m256i dacc = _mm256_setzero_si256(), aacc = dacc, bacc = dacc;
    int i = 0;

    for (; i <= n - 32; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(y + i));
        dacc = _mm256_add_epi32(dacc, dot_epi8(va, vb));
        aacc = _mm256_add_epi32(aacc, dot_epi8(va, va));
        bacc = _mm256_add_epi32(bacc, dot_epi8(vb, vb));
    }

    int64_t dot = hsum256_epi32_signed(dacc);
    int64_t norm_a = hsum256_epi32_signed(aacc);
    int64_t norm_b = hsum256_epi32_signed(bacc);

    for (; i < n; ++i) {
        int32_t p = x[i], q = y[i];
        dot += (int64_t)(p * q);
        norm_a += (int64_t)(p * p);
        norm_b += (int64_t)(q * q);
    }

    if (norm_a == 0 || norm_b == 0) return 1.0f;

    float cosine_similarity = (float)((double)dot / (sqrt((double)norm_a) * sqrt((double)norm_b)));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - BIT -

// lookup table for popcount of 4-bit values (plain byte array — avoids GCC static-init restriction on intrinsics)
static const char popcount_lut_bytes[32] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

static inline __m256i popcount_avx2(__m256i v) {
    __m256i popcount_lut = _mm256_loadu_si256((const __m256i*)popcount_lut_bytes);
    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i cnt_lo = _mm256_shuffle_epi8(popcount_lut, lo);
    __m256i cnt_hi = _mm256_shuffle_epi8(popcount_lut, hi);
    return _mm256_add_epi8(cnt_lo, cnt_hi);
}

float bit1_distance_hamming_avx2 (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    
    // Process 32 bytes at a time
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i xored = _mm256_xor_si256(va, vb);
        __m256i popcnt = popcount_avx2(xored);
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(popcnt, _mm256_setzero_si256()));
    }
    
    // Horizontal sum
    __m128i sum128 = _mm_add_epi64(_mm256_extracti128_si256(acc, 0), _mm256_extracti128_si256(acc, 1));
    int distance = _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);
    
    // Handle remainder with scalar
    for (; i < n; i++) {
        distance += __builtin_popcount(a[i] ^ b[i]);
    }
    
    return (float)distance;
}



#endif

// MARK: -

bool init_distance_functions_avx2 (void) {
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
    
    dispatch_distance_table[VECTOR_DISTANCE_HAMMING][VECTOR_TYPE_BIT] = bit1_distance_hamming_avx2;
    
    distance_backend_name = "AVX2";
    // the TurboQuant lookup scan is gather-bound and shared by every backend
    turbo_lut_dot_function = turbo_lut_dot_cpu;
    turbo_lut_backend_name = "AVX2";
    return true;
#else
    return false;
#endif
}
