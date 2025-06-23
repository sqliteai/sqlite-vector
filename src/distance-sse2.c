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
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_sse2;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_sse2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_sse2;
    
    distance_backend_name = "SSE2";
#endif
}
