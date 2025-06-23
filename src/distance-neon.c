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

float float32_distance_l2_imp_neon (const void *v1, const void *v2, int n, bool use_sqrt) {
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

    float tmp[4];
    vst1q_f32(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }

    return use_sqrt ? sqrtf(sum) : sum;
}

float float32_distance_l2_neon (const void *v1, const void *v2, int n) {
    return float32_distance_l2_imp_neon(v1, v2, n, true);
}

float float32_distance_l2_squared_neon (const void *v1, const void *v2, int n) {
    return float32_distance_l2_imp_neon(v1, v2, n, false);
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
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_neon;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_neon;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_neon;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_neon;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_neon;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_neon;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_neon;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_neon;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_neon;
    
    distance_backend_name = "NEON";
#endif
}
