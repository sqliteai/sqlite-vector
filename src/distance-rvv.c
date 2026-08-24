//
//  distance-rvv.c
//  sqlitevector
//
//  Created by Afonso Bordado on 2026/02/19.
//

#include "distance-rvv.h"
#include "distance-cpu.h"

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern const char *distance_backend_name;
extern turbo_lut_dot_function_t turbo_lut_dot_function;
extern const char *turbo_lut_backend_name;

// MARK: - UTILS -

// Reduces a vector by summing all of it's elements into a single scalar float
static inline float float32_sum_vector_f32m8 (vfloat32m8_t vec, size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vl = __riscv_vsetvl_e32m8(vl);
    acc = __riscv_vfredusum_vs_f32m8_f32m1(vec, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar float
static inline float float32_sum_vector_f32m4 (vfloat32m4_t vec, size_t vl) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vl = __riscv_vsetvl_e32m4(vl);
    acc = __riscv_vfredusum_vs_f32m4_f32m1(vec, acc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar double
static inline double float64_sum_vector_f64m4 (vfloat64m4_t vec, size_t vl) {
    vfloat64m1_t acc = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vl = __riscv_vsetvl_e64m4(vl);
    acc = __riscv_vfredusum_vs_f64m4_f64m1(vec, acc, vl);
    return __riscv_vfmv_f_s_f64m1_f64(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar integer
static inline uint64_t uint64_sum_vector_u64m8 (vuint64m8_t vec, size_t vl) {
    vuint64m1_t acc = __riscv_vmv_s_x_u64m1(0, 1);
    vl = __riscv_vsetvl_e64m8(vl);
    acc = __riscv_vredsum_vs_u64m8_u64m1(vec, acc, vl);
    return __riscv_vmv_x_s_u64m1_u64(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar integer
static inline uint32_t uint32_sum_vector_u32m8 (vuint32m8_t vec, size_t vl) {
    vuint32m1_t acc = __riscv_vmv_s_x_u32m1(0, 1);
    vl = __riscv_vsetvl_e32m8(vl);
    acc = __riscv_vredsum_vs_u32m8_u32m1(vec, acc, vl);
    return __riscv_vmv_x_s_u32m1_u32(acc);
}

// Reduces a vector by summing all of it's elements into a single scalar integer
static inline int32_t int32_sum_vector_i32m8 (vint32m8_t vec, size_t vl) {
    vint32m1_t acc = __riscv_vmv_s_x_i32m1(0, 1);
    vl = __riscv_vsetvl_e32m8(vl);
    acc = __riscv_vredsum_vs_i32m8_i32m1(vec, acc, vl);
    return __riscv_vmv_x_s_i32m1_i32(acc);
}

// Scalar-load fp16 payloads, convert to fp32, and pack as an f32m2 vector.
static inline vfloat32m2_t rvv_load_f16_as_f32m2 (const uint16_t *src, size_t n) {
    size_t vl = __riscv_vsetvl_e32m2(n);
    float lanes[vl];
    for (size_t i = 0; i < vl; ++i) lanes[i] = float16_to_float32(src[i]);
    return __riscv_vle32_v_f32m2(lanes, vl);
}

// Scalar-load bf16 payloads, convert to fp32, and pack as an f32m8 vector.
static inline vfloat32m8_t rvv_load_bf16_as_f32m8 (const uint16_t *src, size_t n) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    float lanes[vl];
    for (size_t i = 0; i < vl; ++i) lanes[i] = bfloat16_to_float32(src[i]);
    return __riscv_vle32_v_f32m8(lanes, vl);
}

// Scalar-load bf16 payloads, convert to fp32, and pack as an f32m4 vector.
static inline vfloat32m4_t rvv_load_bf16_as_f32m4 (const uint16_t *src, size_t n) {
    size_t vl = __riscv_vsetvl_e32m4(n);
    float lanes[vl];
    for (size_t i = 0; i < vl; ++i) lanes[i] = bfloat16_to_float32(src[i]);
    return __riscv_vle32_v_f32m4(lanes, vl);
}

// Scalar-load bf16 payloads, convert to fp32, and pack as an f32m2 vector.
static inline vfloat32m2_t rvv_load_bf16_as_f32m2 (const uint16_t *src, size_t n) {
    size_t vl = __riscv_vsetvl_e32m2(n);
    float lanes[vl];
    for (size_t i = 0; i < vl; ++i) lanes[i] = bfloat16_to_float32(src[i]);
    return __riscv_vle32_v_f32m2(lanes, vl);
}

// Returns true if any lane has an fp16-style infinity mismatch:
// one side is Inf and the other is not, or both are Inf with different signs.
static inline bool rvv_has_f16_inf_mismatch_f64m4 (vfloat64m4_t va, vfloat64m4_t vb, size_t vl) {
    vuint64m4_t a_class = __riscv_vfclass_v_u64m4(va, vl);
    vuint64m4_t b_class = __riscv_vfclass_v_u64m4(vb, vl);
    vuint64m4_t a_inf_bits = __riscv_vand_vx_u64m4(a_class, 0x81u, vl);
    vuint64m4_t b_inf_bits = __riscv_vand_vx_u64m4(b_class, 0x81u, vl);
    vbool16_t inf_mismatch = __riscv_vmsne_vv_u64m4_b16(a_inf_bits, b_inf_bits, vl);
    return __riscv_vfirst_m_b16(inf_mismatch, vl) >= 0;
}

// Returns mask of lanes where both vectors are not NaN.
static inline vbool16_t rvv_both_not_nan_f64m4 (vfloat64m4_t va, vfloat64m4_t vb, size_t vl) {
    vbool16_t a_not_nan = __riscv_vmfeq_vv_f64m4_b16(va, va, vl);
    vbool16_t b_not_nan = __riscv_vmfeq_vv_f64m4_b16(vb, vb, vl);
    return __riscv_vmand_mm_b16(a_not_nan, b_not_nan, vl);
}


// MARK: - FLOAT32 -

float float32_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vl2 = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e32m8(i);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);

        // L2 = (a[i] - b[i]) + acc
        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(va, vb, vl);
        vl2 = __riscv_vfmacc_vv_f32m8(vl2, vdiff, vdiff, vl);
        
        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l2 = float32_sum_vector_f32m8(vl2, n);
    return use_sqrt ? sqrtf(l2) : l2;
}

float float32_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_rvv(v1, v2, n, true);
}

float float32_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_rvv(v1, v2, n, false);
}

float float32_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vsad = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e32m8(i);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);


        // SAD = abs(a[i] - b[i]) + acc
        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(va, vb, vl);
        vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vdiff, vl);
        vsad = __riscv_vfadd_vv_f32m8(vsad, vabs, vl);
        
        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    return float32_sum_vector_f32m8(vsad, n);
}

float float32_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vdot = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e32m8(i);

        // Load the vectors into the registers
        vfloat32m8_t va = __riscv_vle32_v_f32m8(a, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b, vl);

        // Compute the dot product for the entire register, and sum the
        // results into the accumuating register
        vdot = __riscv_vfmacc_vv_f32m8(vdot, va, vb, vl);
        
        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float dot = float32_sum_vector_f32m8(vdot, n);
    return -dot;
}

float float32_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;

    // Use LMUL=4, we have 8 registers to work with.
    size_t vl = __riscv_vsetvlmax_e32m4();

    // Zero out the starting registers
    vfloat32m4_t vdot = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmagn_a = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vmagn_b = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Update VL with the remaining elements
        vl = __riscv_vsetvl_e32m4(i);

        // Load the vectors into the registers
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b, vl);

        // Compute the dot product for the entire register
        vdot = __riscv_vfmacc_vv_f32m4(vdot, va, vb, vl);

        // Also calculate the magnitude value for both a and b
        vmagn_a = __riscv_vfmacc_vv_f32m4(vmagn_a, va, va, vl);
        vmagn_b = __riscv_vfmacc_vv_f32m4(vmagn_b, vb, vb, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Now do a final reduction on the registers to sum the remaining elements
    // TODO: With default flags this does not always use the fsqrt.s/fmin.s/fmax.s instruction, we should fix that
    float dot = float32_sum_vector_f32m4(vdot, n);
    float magn_a = sqrtf(float32_sum_vector_f32m4(vmagn_a, n));
    float magn_b = sqrtf(float32_sum_vector_f32m4(vmagn_b, n));

    if (magn_a == 0.0f || magn_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (magn_a * magn_b);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - FLOAT16 -

static inline float float16_distance_l2_impl_rvv(const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vsum = __riscv_vfmv_v_f_f64m4(0.0, vl);

    for (size_t i = n; i > 0;) {
        // Scalar-load fp16, convert to f32, then widen to f64.
        vl = __riscv_vsetvl_e32m2(i);
        vfloat32m2_t va32 = rvv_load_f16_as_f32m2(a, vl);
        vfloat32m2_t vb32 = rvv_load_f16_as_f32m2(b, vl);
        vfloat64m4_t va = __riscv_vfwcvt_f_f_v_f64m4(va32, vl);
        vfloat64m4_t vb = __riscv_vfwcvt_f_f_v_f64m4(vb32, vl);

        vl = __riscv_vsetvl_e64m4(vl);

        // Return +Inf if there is an infinity mismatch.
        if (rvv_has_f16_inf_mismatch_f64m4(va, vb, vl)) return INFINITY;

        // Skip NaN lanes in accumulation path.
        vbool16_t not_nan = rvv_both_not_nan_f64m4(va, vb, vl);

        vfloat64m4_t vdiff = __riscv_vfsub_vv_f64m4(va, vb, vl);
        vsum = __riscv_vfmacc_vv_f64m4_m(not_nan, vsum, vdiff, vdiff, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    double l2sq = float64_sum_vector_f64m4(vsum, n);
    return use_sqrt ? sqrtf((float)l2sq) : (float)l2sq;
}

float float16_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_rvv(v1, v2, n, true);
}

float float16_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_rvv(v1, v2, n, false);
}

float float16_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vsum = __riscv_vfmv_v_f_f64m4(0.0, vl);

    for (size_t i = n; i > 0;) {
        // Scalar-load fp16, convert to f32, then widen to f64.
        vl = __riscv_vsetvl_e32m2(i);
        vfloat32m2_t va32 = rvv_load_f16_as_f32m2(a, vl);
        vfloat32m2_t vb32 = rvv_load_f16_as_f32m2(b, vl);
        vfloat64m4_t va = __riscv_vfwcvt_f_f_v_f64m4(va32, vl);
        vfloat64m4_t vb = __riscv_vfwcvt_f_f_v_f64m4(vb32, vl);

        vl = __riscv_vsetvl_e64m4(vl);

        // Return +Inf if there is an infinity mismatch.
        if (rvv_has_f16_inf_mismatch_f64m4(va, vb, vl)) return INFINITY;

        // Skip NaN lanes in accumulation path.
        vbool16_t not_nan = rvv_both_not_nan_f64m4(va, vb, vl);

        vfloat64m4_t vdiff = __riscv_vfsub_vv_f64m4(va, vb, vl);
        vfloat64m4_t vabs = __riscv_vfabs_v_f64m4(vdiff, vl);
        vsum = __riscv_vfadd_vv_f64m4_m(not_nan, vsum, vabs, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    return (float)float64_sum_vector_f64m4(vsum, n);
}

float float16_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    // Keep accumulation vectorized while preserving CPU NaN/Inf semantics.
    size_t vl = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vdot = __riscv_vfmv_v_f_f64m4(0.0, vl);

    for (size_t i = n; i > 0;) {
        // Scalar-load fp16, convert to f32, then widen to f64.
        vl = __riscv_vsetvl_e32m2(i);
        vfloat32m2_t va32 = rvv_load_f16_as_f32m2(a, vl);
        vfloat32m2_t vb32 = rvv_load_f16_as_f32m2(b, vl);
        vfloat64m4_t va = __riscv_vfwcvt_f_f_v_f64m4(va32, vl);
        vfloat64m4_t vb = __riscv_vfwcvt_f_f_v_f64m4(vb32, vl);
        
        vl = __riscv_vsetvl_e64m4(vl);

        // not_nan = lanes where both sides are not NaN.
        vbool16_t not_nan = rvv_both_not_nan_f64m4(va, vb, vl);

        // Multiply once, then classify the product only.
        vfloat64m4_t vprod = __riscv_vfmul_vv_f64m4(va, vb, vl);

        // Try to find infinite values, if there are any, exit early
        vuint64m4_t p_class = __riscv_vfclass_v_u64m4(vprod, vl);
        vbool16_t inf_pos = __riscv_vmsne_vx_u64m4_b16_m(not_nan, __riscv_vand_vx_u64m4_m(not_nan, p_class, 0x80u, vl), 0u, vl);
        vbool16_t inf_neg = __riscv_vmsne_vx_u64m4_b16_m(not_nan, __riscv_vand_vx_u64m4_m(not_nan, p_class, 0x01u, vl), 0u, vl);
        long first_pos = __riscv_vfirst_m_b16(inf_pos, vl);
        long first_neg = __riscv_vfirst_m_b16(inf_neg, vl);
        if (first_pos >= 0 || first_neg >= 0) {
            if (first_pos >= 0 && (first_neg < 0 || first_pos < first_neg)) return -INFINITY;
            return INFINITY;
        }

        // Accumulate only valid lanes; NaN lanes are skipped.
        vdot = __riscv_vfadd_vv_f64m4_m(not_nan, vdot, vprod, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    double dot = float64_sum_vector_f64m4(vdot, n);
    return (float)(-dot);
}

float float16_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vdot = __riscv_vfmv_v_f_f64m4(0.0, vl);
    vfloat64m4_t vnx = __riscv_vfmv_v_f_f64m4(0.0, vl);
    vfloat64m4_t vny = __riscv_vfmv_v_f_f64m4(0.0, vl);

    for (size_t i = n; i > 0;) {
        // Scalar-load fp16, convert to f32, then widen to f64.
        vl = __riscv_vsetvl_e32m2(i);
        vfloat32m2_t va32 = rvv_load_f16_as_f32m2(a, vl);
        vfloat32m2_t vb32 = rvv_load_f16_as_f32m2(b, vl);
        vfloat64m4_t va = __riscv_vfwcvt_f_f_v_f64m4(va32, vl);
        vfloat64m4_t vb = __riscv_vfwcvt_f_f_v_f64m4(vb32, vl);

        vl = __riscv_vsetvl_e64m4(vl);

        // Keep only lanes where both values are not NaN.
        vbool16_t not_nan = rvv_both_not_nan_f64m4(va, vb, vl);

        // Any infinity on a valid lane returns 1.0f.
        vuint64m4_t a_class = __riscv_vfclass_v_u64m4(va, vl);
        vuint64m4_t b_class = __riscv_vfclass_v_u64m4(vb, vl);
        vuint64m4_t ab_class = __riscv_vor_vv_u64m4(a_class, b_class, vl);
        vbool16_t ab_inf = __riscv_vmsne_vx_u64m4_b16(__riscv_vand_vx_u64m4(ab_class, 0x81u, vl), 0u, vl);
        vbool16_t any_inf = __riscv_vmand_mm_b16(not_nan, ab_inf, vl);
        if (__riscv_vfirst_m_b16(any_inf, vl) >= 0) return 1.0f;

        // Accumulate dot and squared norms on valid lanes.
        vfloat64m4_t vprod = __riscv_vfmul_vv_f64m4(va, vb, vl);
        vdot = __riscv_vfadd_vv_f64m4_m(not_nan, vdot, vprod, vl);
        vnx = __riscv_vfmacc_vv_f64m4_m(not_nan, vnx, va, va, vl);
        vny = __riscv_vfmacc_vv_f64m4_m(not_nan, vny, vb, vb, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    double dot = float64_sum_vector_f64m4(vdot, n);
    double nx = float64_sum_vector_f64m4(vnx, n);
    double ny = float64_sum_vector_f64m4(vny, n);
    double denom = sqrt(nx) * sqrt(ny);
    if (!(denom > 0.0) || !isfinite(denom) || !isfinite(dot)) return 1.0f;

    double cosv = dot / denom;
    if (cosv > 1.0) cosv = 1.0;
    if (cosv < -1.0) cosv = -1.0;
    return (float)(1.0 - cosv);
}

// MARK: - BFLOAT16 -

static inline float bfloat16_distance_l2_impl_rvv(const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t vsum = __riscv_vfmv_v_f_f64m4(0.0, vl);

    for (size_t i = n; i > 0;) {
        // Load as f32m2 and widen to f64m4 to avoid overflow in accumulation.
        vl = __riscv_vsetvl_e32m2(i);
        vfloat32m2_t va32 = rvv_load_bf16_as_f32m2(a, vl);
        vfloat32m2_t vb32 = rvv_load_bf16_as_f32m2(b, vl);
        vfloat64m4_t va = __riscv_vfwcvt_f_f_v_f64m4(va32, vl);
        vfloat64m4_t vb = __riscv_vfwcvt_f_f_v_f64m4(vb32, vl);
        
        vl = __riscv_vsetvl_e64m4(vl);

        vfloat64m4_t vdiff = __riscv_vfsub_vv_f64m4(va, vb, vl);

        // If any diff lane is infinite, return +INFINITY.
        vuint64m4_t d_class = __riscv_vfclass_v_u64m4(vdiff, vl);
        vbool16_t d_inf = __riscv_vmsne_vx_u64m4_b16(__riscv_vand_vx_u64m4(d_class, 0x81u, vl), 0u, vl);
        if (__riscv_vfirst_m_b16(d_inf, vl) >= 0) return INFINITY;

        // Skip NaN diff lanes.
        vbool16_t not_nan = __riscv_vmfeq_vv_f64m4_b16(vdiff, vdiff, vl);
        vsum = __riscv_vfmacc_vv_f64m4_m(not_nan, vsum, vdiff, vdiff, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    double l2sq = float64_sum_vector_f64m4(vsum, n);
    return use_sqrt ? sqrtf((float)l2sq) : (float)l2sq;
}

float bfloat16_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_rvv(v1, v2, n, true);
}

float bfloat16_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_rvv(v1, v2, n, false);
}

float bfloat16_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (size_t i = n; i > 0;) {
        vl = __riscv_vsetvl_e32m8(i);
        vfloat32m8_t va = rvv_load_bf16_as_f32m8(a, vl);
        vfloat32m8_t vb = rvv_load_bf16_as_f32m8(b, vl);

        vfloat32m8_t vdiff = __riscv_vfsub_vv_f32m8(va, vb, vl);
        vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(vdiff, vl);
        vsum = __riscv_vfadd_vv_f32m8(vsum, vabs, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    return float32_sum_vector_f32m8(vsum, n);
}

float bfloat16_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vdot = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (size_t i = n; i > 0;) {
        vl = __riscv_vsetvl_e32m8(i);
        vfloat32m8_t va = rvv_load_bf16_as_f32m8(a, vl);
        vfloat32m8_t vb = rvv_load_bf16_as_f32m8(b, vl);
        vdot = __riscv_vfmacc_vv_f32m8(vdot, va, vb, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    float dot = float32_sum_vector_f32m8(vdot, n);
    return -dot;
}

float bfloat16_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    size_t vl = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vdot = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vnx = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vny = __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (size_t i = n; i > 0;) {
        vl = __riscv_vsetvl_e32m4(i);
        vfloat32m4_t va = rvv_load_bf16_as_f32m4(a, vl);
        vfloat32m4_t vb = rvv_load_bf16_as_f32m4(b, vl);

        vdot = __riscv_vfmacc_vv_f32m4(vdot, va, vb, vl);
        vnx = __riscv_vfmacc_vv_f32m4(vnx, va, va, vl);
        vny = __riscv_vfmacc_vv_f32m4(vny, vb, vb, vl);

        a += vl;
        b += vl;
        i -= vl;
    }

    float dot = float32_sum_vector_f32m4(vdot, n);
    float norm_x = float32_sum_vector_f32m4(vnx, n);
    float norm_y = float32_sum_vector_f32m4(vny, n);
    if (norm_x == 0.0f || norm_y == 0.0f) return 1.0f;

    float cosine_similarity = dot / (sqrtf(norm_x) * sqrtf(norm_y));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - UINT8 -

float uint8_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vl2 = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Widen these values to 16bit unsigned
        vuint16m4_t va_wide = __riscv_vwcvtu_x_x_v_u16m4(va, vl);
        vuint16m4_t vb_wide = __riscv_vwcvtu_x_x_v_u16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Cast these to signed values
        vint16m4_t va_wides = __riscv_vreinterpret_v_u16m4_i16m4(va_wide);
        vint16m4_t vb_wides = __riscv_vreinterpret_v_u16m4_i16m4(vb_wide);

        // L2 = (a[i] - b[i]) + acc
        // The subtract is signed, but the accumulate is unsigned
        vint32m8_t vdiff = __riscv_vwsub_vv_i32m8(va_wides, vb_wides, vl);
        vl2 = __riscv_vmacc_vv_i32m8(vl2, vdiff, vdiff, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l2 = (float) int32_sum_vector_i32m8(vl2, n);
    return use_sqrt ? sqrtf(l2) : l2;
}

float uint8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_rvv(v1, v2, n, true);
}

float uint8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_rvv(v1, v2, n, false);
}

float uint8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vuint32m8_t vdot = __riscv_vmv_s_x_u32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Widen these vectors to 16bit
        vuint16m4_t va_wide = __riscv_vwcvtu_x_x_v_u16m4(va, vl);
        vuint16m4_t vb_wide = __riscv_vwcvtu_x_x_v_u16m4(vb, vl);

        // Now we're operating on 16 bit elements
        vl = __riscv_vsetvl_e16m4(i);

        // Do a widening multiply-accumulate to 32 bits
        vdot = __riscv_vwmaccu_vv_u32m8(vdot, va_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float dot = uint32_sum_vector_u32m8(vdot, n);
    return -dot;
}

float uint8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vuint32m8_t vl1 = __riscv_vmv_s_x_u32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Compute the absolute difference by getting the min and max and subtracting them.
        vuint8m2_t vmin = __riscv_vminu_vv_u8m2(va, vb, vl);
        vuint8m2_t vmax = __riscv_vmaxu_vv_u8m2(va, vb, vl);
        vuint16m4_t vabs = __riscv_vwsubu_vv_u16m4(vmax, vmin, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Now widen it to 32bits and add to the accumulator
        vuint32m8_t vwide = __riscv_vwcvtu_x_x_v_u32m8(vabs, vl);
        vl1 = __riscv_vadd_vv_u32m8(vl1, vwide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l1 = uint32_sum_vector_u32m8(vl1, n);
    return l1;
}

float uint8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();

    // Zero out the starting registers
    vuint32m8_t vdot = __riscv_vmv_s_x_u32m8(0, vl);
    vuint32m8_t vmagn_a = __riscv_vmv_s_x_u32m8(0, vl);
    vuint32m8_t vmagn_b = __riscv_vmv_s_x_u32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vuint8m2_t va = __riscv_vle8_v_u8m2(a, vl);
        vuint8m2_t vb = __riscv_vle8_v_u8m2(b, vl);

        // Widen these values to 16bit unsigned
        vuint16m4_t va_wide = __riscv_vwcvtu_x_x_v_u16m4(va, vl);
        vuint16m4_t vb_wide = __riscv_vwcvtu_x_x_v_u16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Compute the dot product for the entire register (widening madd)
        vdot = __riscv_vwmaccu_vv_u32m8(vdot, va_wide, vb_wide, vl);

        // Also calculate the magnitude value for both a and b (widening madd)
        vmagn_a = __riscv_vwmaccu_vv_u32m8(vmagn_a, va_wide, va_wide, vl);
        vmagn_b = __riscv_vwmaccu_vv_u32m8(vmagn_b, vb_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Now do a final reduction on the registers to sum the remaining elements
    // TODO: With default flags this does not always use the fsqrt.s/fmin.s/fmax.s instruction, we should fix that
    float dot = uint32_sum_vector_u32m8(vdot, n);
    float magn_a = sqrtf(uint32_sum_vector_u32m8(vmagn_a, n));
    float magn_b = sqrtf(uint32_sum_vector_u32m8(vmagn_b, n));

    if (magn_a == 0.0f || magn_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (magn_a * magn_b);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - INT8 -

float int8_distance_l2_impl_rvv (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vl2 = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Widen these values to 16bit signed
        vint16m4_t va_wide = __riscv_vwcvt_x_x_v_i16m4(va, vl);
        vint16m4_t vb_wide = __riscv_vwcvt_x_x_v_i16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // L2 = (a[i] - b[i]) + acc
        vint32m8_t vdiff = __riscv_vwsub_vv_i32m8(va_wide, vb_wide, vl);
        vl2 = __riscv_vmacc_vv_i32m8(vl2, vdiff, vdiff, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l2 = (float) int32_sum_vector_i32m8(vl2, n);
    return use_sqrt ? sqrtf(l2) : l2;
}

float int8_distance_l2_rvv (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_rvv(v1, v2, n, true);
}

float int8_distance_l2_squared_rvv (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_rvv(v1, v2, n, false);
}

float int8_distance_dot_rvv (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vdot = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Widen these vectors to 16bit
        vint16m4_t va_wide = __riscv_vwcvt_x_x_v_i16m4(va, vl);
        vint16m4_t vb_wide = __riscv_vwcvt_x_x_v_i16m4(vb, vl);

        // Now we're operating on 16 bit elements
        vl = __riscv_vsetvl_e16m4(i);

        // Do a widening multiply-accumulate to 32 bits
        vdot = __riscv_vwmacc_vv_i32m8(vdot, va_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float dot = (float) int32_sum_vector_i32m8(vdot, n);
    return -dot;
}

float int8_distance_l1_rvv (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();
    vint32m8_t vl1 = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Compute the absolute difference by getting the min and max and subtracting them.
        vint8m2_t vmin = __riscv_vmin_vv_i8m2(va, vb, vl);
        vint8m2_t vmax = __riscv_vmax_vv_i8m2(va, vb, vl);
        vint16m4_t vabs = __riscv_vwsub_vv_i16m4(vmax, vmin, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Now widen it to 32bits and add to the accumulator
        vint32m8_t vwide = __riscv_vwcvt_x_x_v_i32m8(vabs, vl);
        vl1 = __riscv_vadd_vv_i32m8(vl1, vwide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Copy the accumulators back into a scalar register
    float l1 = (float) int32_sum_vector_i32m8(vl1, n);
    return l1;
}

float int8_distance_cosine_rvv (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e32m8();

    // Zero out the starting registers
    vint32m8_t vdot = __riscv_vmv_s_x_i32m8(0, vl);
    vint32m8_t vmagn_a = __riscv_vmv_s_x_i32m8(0, vl);
    vint32m8_t vmagn_b = __riscv_vmv_s_x_i32m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=2 to start off, but we're going to widen this
        vl = __riscv_vsetvl_e8m2(i);

        // Load the vectors into the registers
        vint8m2_t va = __riscv_vle8_v_i8m2(a, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b, vl);

        // Widen these values to 16bit signed
        vint16m4_t va_wide = __riscv_vwcvt_x_x_v_i16m4(va, vl);
        vint16m4_t vb_wide = __riscv_vwcvt_x_x_v_i16m4(vb, vl);
        vl = __riscv_vsetvl_e16m4(i);

        // Compute the dot product for the entire register (widening madd)
        vdot = __riscv_vwmacc_vv_i32m8(vdot, va_wide, vb_wide, vl);

        // Also calculate the magnitude value for both a and b (widening madd)
        vmagn_a = __riscv_vwmacc_vv_i32m8(vmagn_a, va_wide, va_wide, vl);
        vmagn_b = __riscv_vwmacc_vv_i32m8(vmagn_b, vb_wide, vb_wide, vl);

        // Advance the a and b pointers to the next offset
        a = &a[vl];
        b = &b[vl];
    }

    // Now do a final reduction on the registers to sum the remaining elements
    float dot = (float) int32_sum_vector_i32m8(vdot, n);
    float magn_a = sqrtf((float) int32_sum_vector_i32m8(vmagn_a, n));
    float magn_b = sqrtf((float) int32_sum_vector_i32m8(vmagn_b, n));

    if (magn_a == 0.0f || magn_b == 0.0f) return 1.0f;

    float cosine_similarity = dot / (magn_a * magn_b);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

// MARK: - BIT -


// Counts the number of set bits on each element of a vector register
//
// TODO: RISC-V natively supports vcpop.v for population count, but only with the
// Zvbb extension, which we don't support yet. For everyone else, do a fallback implemetation.
vuint64m8_t vpopcnt_u64m8(vuint64m8_t v, size_t vl) {
    // v = v - ((v >> 1) & 0x5555555555555555ULL);
    vuint64m8_t shr1 = __riscv_vsrl_vx_u64m8(v, 1, vl);
    vuint64m8_t and1 = __riscv_vand_vx_u64m8(shr1, 0x5555555555555555ULL, vl);
    v = __riscv_vsub_vv_u64m8(v, and1, vl);
    
    // v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    vuint64m8_t shr2 = __riscv_vsrl_vx_u64m8(v, 2, vl);
    vuint64m8_t and2 = __riscv_vand_vx_u64m8(shr2, 0x3333333333333333ULL, vl);
    vuint64m8_t and3 = __riscv_vand_vx_u64m8(v, 0x3333333333333333ULL, vl);
    v = __riscv_vadd_vv_u64m8(and2, and3, vl);

    // v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    vuint64m8_t shr4 = __riscv_vsrl_vx_u64m8(v, 4, vl);
    vuint64m8_t add = __riscv_vadd_vv_u64m8(v, shr4, vl);
    v = __riscv_vand_vx_u64m8(add, 0x0f0f0f0f0f0f0f0fULL, vl);

    // v = (v * 0x0101010101010101ULL) >> 56;
    vuint64m8_t mul = __riscv_vmul_vx_u64m8(v, 0x0101010101010101ULL, vl);
    v = __riscv_vsrl_vx_u64m8(mul, 56, vl);

    return v;
}

float bit1_distance_hamming_rvv (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;

    // We accumulate the results into a vector register
    size_t vl = __riscv_vsetvlmax_e64m8();
    vuint64m8_t vdistance = __riscv_vmv_s_x_u64m8(0, vl);

    // Iterate by VL elements
    for (size_t i = n; i > 0; i -= vl) {
        // Use LMUL=8, we have 4 registers to work with.
        vl = __riscv_vsetvl_e64m8(i);

        // Load the vectors into the registers and cast them into a u64 inplace
        vuint64m8_t va = __riscv_vreinterpret_v_u8m8_u64m8(__riscv_vle8_v_u8m8(a, vl));
        vuint64m8_t vb = __riscv_vreinterpret_v_u8m8_u64m8(__riscv_vle8_v_u8m8(b, vl));

        vuint64m8_t xor = __riscv_vxor_vv_u64m8(va, vb, vl);
        vuint64m8_t popcnt = vpopcnt_u64m8(xor, vl);
        vdistance = __riscv_vadd_vv_u64m8(vdistance, popcnt, vl);

        // Advance the a and b pointers to the next offset. Here we multiply by 8 because
        // the vectors are defined as u8, but VL is defined in elements of 64bits.
        a = &a[vl * 8];
        b = &b[vl * 8];
    }

    // Copy the accumulator back into a scalar register
    return (float) uint64_sum_vector_u64m8(vdistance, vl);
}


#endif

// MARK: -

bool init_distance_functions_rvv (void) {
#if defined(__riscv_v_intrinsic)
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F32] = float32_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_F16] = float16_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_U8] = uint8_distance_l2_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L2][VECTOR_TYPE_I8] = int8_distance_l2_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F32] = float32_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_F16] = float16_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_U8] = uint8_distance_l2_squared_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_SQUARED_L2][VECTOR_TYPE_I8] = int8_distance_l2_squared_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F32] = float32_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_F16] = float16_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_BF16] = bfloat16_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_U8] = uint8_distance_cosine_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_COSINE][VECTOR_TYPE_I8] = int8_distance_cosine_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] = float32_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F16] = float16_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_BF16] = bfloat16_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_U8] = uint8_distance_dot_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_I8] = int8_distance_dot_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F32] = float32_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_F16] = float16_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_BF16] = bfloat16_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_U8] = uint8_distance_l1_rvv;
    dispatch_distance_table[VECTOR_DISTANCE_L1][VECTOR_TYPE_I8] = int8_distance_l1_rvv;
    
    dispatch_distance_table[VECTOR_DISTANCE_HAMMING][VECTOR_TYPE_BIT] = bit1_distance_hamming_rvv;
    
    distance_backend_name = "RVV";
    // the TurboQuant lookup scan is gather-bound and shared by every backend
    turbo_lut_dot_function = turbo_lut_dot_cpu;
    turbo_lut_backend_name = "RVV";
    return true;
#else
    return false;
#endif
}
