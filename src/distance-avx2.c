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
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_avx2;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_avx2;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_avx2;
    
    distance_backend_name = "AVX2";
#endif
}
