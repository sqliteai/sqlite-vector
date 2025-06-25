//
//  distance-cpu.c
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#include "distance-cpu.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "distance-neon.h"
#include "distance-sse2.h"
#include "distance-avx2.h"

char *distance_backend_name = "CPU";
distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX] = {0};

// MARK: FLOAT32 -

float float32_distance_l2_impl_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float sum_sq = 0.0f;
    int i = 0;
    
    if (n >= 4) {
        // unroll the loop 4 times
        for (; i <= n - 4; i += 4) {
            float d0 = a[i] - b[i];
            float d1 = a[i+1] - b[i+1];
            float d2 = a[i+2] - b[i+2];
            float d3 = a[i+3] - b[i+3];
            sum_sq += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
    }
    
    // tail loop
    for (; i < n; i++) {
        float d = a[i] - b[i];
        sum_sq += d * d;
    }
    
    return use_sqrt ? sqrtf(sum_sq) : sum_sq;
}

float float32_distance_l2_cpu (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_cpu(v1, v2, n, true);
}

float float32_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return float32_distance_l2_impl_cpu(v1, v2, n, false);
}

float float32_distance_cosine_cpu (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float dot = 0.0f;
    float norm_x = 0.0f;
    float norm_y = 0.0f;
    int i = 0;
    
    // unroll the loop 4 times
    for (; i <= n - 4; i += 4) {
        float x0 = a[i],     y0 = b[i];
        float x1 = a[i + 1], y1 = b[i + 1];
        float x2 = a[i + 2], y2 = b[i + 2];
        float x3 = a[i + 3], y3 = b[i + 3];
        
        dot     += x0*y0 + x1*y1 + x2*y2 + x3*y3;
        norm_x  += x0*x0 + x1*x1 + x2*x2 + x3*x3;
        norm_y  += y0*y0 + y1*y1 + y2*y2 + y3*y3;
    }
    
    // tail loop
    for (; i < n; i++) {
        float x = a[i];
        float y = b[i];
        dot    += x * y;
        norm_x += x * x;
        norm_y += y * y;
    }
    
    // max distance if one vector is zero
    if (norm_x == 0.0f || norm_y == 0.0f) {
        return 1.0f;
    }
    
    return 1.0f - (dot / (sqrtf(norm_x) * sqrtf(norm_y)));
}

float float32_distance_dot_cpu (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float dot = 0.0f;
    int i = 0;
    
    // unroll the loop 4 times
    for (; i <= n - 4; i += 4) {
        float x0 = a[i],     y0 = b[i];
        float x1 = a[i + 1], y1 = b[i + 1];
        float x2 = a[i + 2], y2 = b[i + 2];
        float x3 = a[i + 3], y3 = b[i + 3];
        dot += x0*y0 + x1*y1 + x2*y2 + x3*y3;
    }
    
    // tail loop
    for (; i < n; i++) {
        float x = a[i];
        float y = b[i];
        dot += x * y;
    }
    
    return -dot;
}

float float32_distance_l1_cpu (const void *v1, const void *v2, int n) {
    const float *a = (const float *)v1;
    const float *b = (const float *)v2;
    
    float sum = 0.0f;
    int i = 0;

    // unroll the loop 4 times
    for (; i <= n - 4; i += 4) {
        sum += fabsf(a[i]     - b[i]);
        sum += fabsf(a[i + 1] - b[i + 1]);
        sum += fabsf(a[i + 2] - b[i + 2]);
        sum += fabsf(a[i + 3] - b[i + 3]);
    }

    // tail loop
    for (; i < n; ++i) {
        sum += fabsf(a[i] - b[i]);
    }

    return sum;
}

// MARK: - UINT8 -

static inline float uint8_distance_l2_imp_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    float sum = 0.0f;
    int i = 0;
    
    // unrolled loop
    for (; i <= n - 4; i += 4) {
        int d0 = (int)a[i + 0] - (int)b[i + 0];
        int d1 = (int)a[i + 1] - (int)b[i + 1];
        int d2 = (int)a[i + 2] - (int)b[i + 2];
        int d3 = (int)a[i + 3] - (int)b[i + 3];
        
        sum += (float)(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
    }
    
    // tail loop
    for (; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        sum += (float)(d * d);
    }
    
    return use_sqrt ? sqrtf(sum) : sum;
}

float uint8_distance_l2_cpu (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_imp_cpu(v1, v2, n, true);
}

float uint8_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_imp_cpu(v1, v2, n, false);
}

float uint8_distance_cosine_cpu (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    uint32_t dot = 0;
    uint32_t norm_a2 = 0;
    uint32_t norm_b2 = 0;

    int i = 0;
    for (; i <= n - 4; i += 4) {
        uint32_t a0 = a[i + 0], b0 = b[i + 0];
        uint32_t a1 = a[i + 1], b1 = b[i + 1];
        uint32_t a2 = a[i + 2], b2 = b[i + 2];
        uint32_t a3 = a[i + 3], b3 = b[i + 3];

        dot     += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a2 += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b2 += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    // tail loop
    for (; i < n; ++i) {
        uint32_t ai = a[i];
        uint32_t bi = b[i];
        dot     += ai * bi;
        norm_a2 += ai * ai;
        norm_b2 += bi * bi;
    }

    if (norm_a2 == 0 || norm_b2 == 0) {
        return 1.0f;
    }

    float cosine_similarity = dot / (sqrtf((float)norm_a2) * sqrtf((float)norm_b2));
    return 1.0f - cosine_similarity;
}

float uint8_distance_dot_cpu (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    float dot = 0.0f;
    
    int i = 0;
    for (; i <= n - 4; i += 4) {
        dot += (float)(a[i + 0]) * b[i + 0];
        dot += (float)(a[i + 1]) * b[i + 1];
        dot += (float)(a[i + 2]) * b[i + 2];
        dot += (float)(a[i + 3]) * b[i + 3];
    }
    for (; i < n; ++i) {
        dot += (float)(a[i]) * b[i];
    }
    
    return -dot;  // dot distance = negative dot product
}

float uint8_distance_l1_cpu (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    float sum = 0.0f;

    int i = 0;
    for (; i <= n - 4; i += 4) {
        sum += fabsf((float)a[i + 0] - (float)b[i + 0]);
        sum += fabsf((float)a[i + 1] - (float)b[i + 1]);
        sum += fabsf((float)a[i + 2] - (float)b[i + 2]);
        sum += fabsf((float)a[i + 3] - (float)b[i + 3]);
    }

    for (; i < n; ++i) {
        sum += fabsf((float)a[i] - (float)b[i]);
    }

    return sum;
}

// MARK: - INT8 -

float int8_distance_l2_imp_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    float sum = 0.0f;
    int i = 0;
    
    // unrolled loop
    for (; i <= n - 4; i += 4) {
        int d0 = (int)a[i + 0] - (int)b[i + 0];
        int d1 = (int)a[i + 1] - (int)b[i + 1];
        int d2 = (int)a[i + 2] - (int)b[i + 2];
        int d3 = (int)a[i + 3] - (int)b[i + 3];
        
        sum += (float)(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
    }
    
    // tail loop
    for (; i < n; ++i) {
        int d = (int)a[i] - (int)b[i];
        sum += (float)(d * d);
    }
    
    return use_sqrt ? sqrtf(sum) : sum;
}

float int8_distance_l2_cpu (const void *v1, const void *v2, int n) {
    return int8_distance_l2_imp_cpu(v1, v2, n, true);
}

float int8_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return int8_distance_l2_imp_cpu(v1, v2, n, false);
}

float int8_distance_cosine_cpu (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    float dot = 0.0f, norm_a2 = 0.0f, norm_b2 = 0.0f;
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float a0 = (float)a[i + 0], b0 = (float)b[i + 0];
        float a1 = (float)a[i + 1], b1 = (float)b[i + 1];
        float a2 = (float)a[i + 2], b2 = (float)b[i + 2];
        float a3 = (float)a[i + 3], b3 = (float)b[i + 3];

        dot     += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a2 += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b2 += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    for (; i < n; ++i) {
        float ai = (float)a[i];
        float bi = (float)b[i];
        dot     += ai * bi;
        norm_a2 += ai * ai;
        norm_b2 += bi * bi;
    }

    if (norm_a2 == 0.0f || norm_b2 == 0.0f) {
        return 1.0f;
    }

    float cosine_sim = dot / (sqrtf(norm_a2) * sqrtf(norm_b2));
    return 1.0f - cosine_sim;
}

float int8_distance_dot_cpu (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    float dot = 0.0f;
    int i = 0;

    for (; i <= n - 4; i += 4) {
        dot += (float)a[i + 0] * b[i + 0];
        dot += (float)a[i + 1] * b[i + 1];
        dot += (float)a[i + 2] * b[i + 2];
        dot += (float)a[i + 3] * b[i + 3];
    }

    for (; i < n; ++i) {
        dot += (float)a[i] * b[i];
    }

    return -dot;
}

float int8_distance_l1_cpu (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    float sum = 0.0f;
    int i = 0;

    for (; i <= n - 4; i += 4) {
        sum += fabsf((float)a[i + 0] - (float)b[i + 0]);
        sum += fabsf((float)a[i + 1] - (float)b[i + 1]);
        sum += fabsf((float)a[i + 2] - (float)b[i + 2]);
        sum += fabsf((float)a[i + 3] - (float)b[i + 3]);
    }

    for (; i < n; ++i) {
        sum += fabsf((float)a[i] - (float)b[i]);
    }

    return sum;
}

// MARK: - ENTRYPOINT -

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <cpuid.h>

    static void x86_cpuid(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
        #if defined(_MSC_VER)
            int regs[4];
            __cpuidex(regs, leaf, subleaf);
            *eax = regs[0]; *ebx = regs[1]; *ecx = regs[2]; *edx = regs[3];
        #else
            __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
        #endif
    }

    bool cpu_supports_avx2 (void) {
        #if FORCE_AVX2
        return true;
        #else
        int eax, ebx, ecx, edx;
        x86_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
        if (eax < 7) return false;
        x86_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & (1 << 5)) != 0;  // AVX2
        #endif
    }

    bool cpu_supports_sse2 (void) {
        int eax, ebx, ecx, edx;
        x86_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
        return (edx & (1 << 26)) != 0;  // SSE2
    }

#else
    // For ARM (NEON is always present on aarch64, runtime detection rarely needed)
    #if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    bool cpu_supports_neon (void) {
        return true;
    }
    #else
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
    bool cpu_supports_neon (void) {
        #ifdef AT_HWCAP
        return (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
        #else
        return false;
        #endif
    }
    #endif
#endif

// MARK: -

void init_cpu_functions (void) {
    distance_function_t cpu_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX] = {
        [VECTOR_DISTANCE_L2] = {
                [VECTOR_TYPE_F32] = float32_distance_l2_cpu,
                [VECTOR_TYPE_F16] = NULL,
                [VECTOR_TYPE_BF16] = NULL,
                [VECTOR_TYPE_U8]  = uint8_distance_l2_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_l2_cpu,
            },
            [VECTOR_DISTANCE_SQUARED_L2] = {
                [VECTOR_TYPE_F32] = float32_distance_l2_squared_cpu,
                [VECTOR_TYPE_F16] = NULL,
                [VECTOR_TYPE_BF16] = NULL,
                [VECTOR_TYPE_U8]  = uint8_distance_l2_squared_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_l2_squared_cpu,
            },
            [VECTOR_DISTANCE_COSINE] = {
                [VECTOR_TYPE_F32] = float32_distance_cosine_cpu,
                [VECTOR_TYPE_F16] = NULL,
                [VECTOR_TYPE_BF16] = NULL,
                [VECTOR_TYPE_U8]  = uint8_distance_cosine_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_cosine_cpu,
            },
            [VECTOR_DISTANCE_DOT] = {
                [VECTOR_TYPE_F32] = float32_distance_dot_cpu,
                [VECTOR_TYPE_F16] = NULL,
                [VECTOR_TYPE_BF16] = NULL,
                [VECTOR_TYPE_U8]  = uint8_distance_dot_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_dot_cpu,
            },
            [VECTOR_DISTANCE_L1] = {
                [VECTOR_TYPE_F32] = float32_distance_l1_cpu,
                [VECTOR_TYPE_F16] = NULL,
                [VECTOR_TYPE_BF16] = NULL,
                [VECTOR_TYPE_U8]  = uint8_distance_l1_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_l1_cpu,
            }
    };
    
    memcpy(dispatch_distance_table, cpu_table, sizeof(cpu_table));
}

void init_distance_functions (bool force_cpu) {
    init_cpu_functions();
    if (force_cpu) return;
    
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (1/*cpu_supports_avx2()*/) {
        init_distance_functions_avx2();
    } else if (cpu_supports_sse2()) {
        init_distance_functions_sse2();
    }
    #elif defined(__ARM_NEON) || defined(__aarch64__)
    if (cpu_supports_neon()) {
        init_distance_functions_neon();
    }
    #endif
}

