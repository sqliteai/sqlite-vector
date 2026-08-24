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
#include "distance-avx512.h"
#include "distance-rvv.h"

const char *distance_backend_name = "CPU";
distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX] = {0};
const char *turbo_lut_backend_name = "CPU";
turbo_lut_dot_function_t turbo_lut_dot_function = NULL;

#define LASSQ_UPDATE(ad_) do {                            \
        double _ad = (ad_);                               \
        if (_ad != 0.0) {                                 \
            if (scale < _ad) {                            \
                double r = scale / _ad;                   \
                ssq = 1.0 + ssq * (r * r);                \
                scale = _ad;                              \
            } else {                                      \
                double r = _ad / scale;                   \
                ssq += r * r;                             \
            }                                             \
        }                                                 \
    } while (0)

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

    float cosine_similarity = dot / (sqrtf(norm_x) * sqrtf(norm_y));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
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

// MARK: - BFLOAT16 -

// Overflow/underflow-safe L2 using LASSQ, unrolled by 4
static inline float bfloat16_distance_l2_impl_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;
    
    double scale = 0.0;
    double ssq   = 1.0;
    int i = 0;
    
    // unrolled main loop (x4)
    for (; i <= n - 4; i += 4) {
        float x0 = bfloat16_to_float32(a[i    ]), y0 = bfloat16_to_float32(b[i    ]);
        float x1 = bfloat16_to_float32(a[i + 1]), y1 = bfloat16_to_float32(b[i + 1]);
        float x2 = bfloat16_to_float32(a[i + 2]), y2 = bfloat16_to_float32(b[i + 2]);
        float x3 = bfloat16_to_float32(a[i + 3]), y3 = bfloat16_to_float32(b[i + 3]);
        
        float d0f = x0 - y0, d1f = x1 - y1, d2f = x2 - y2, d3f = x3 - y3;
        
        // If any difference is NaN, ignore that lane (treat contribution as 0)
        if (isinf(d0f)) return INFINITY; if (!isnan(d0f)) LASSQ_UPDATE(fabs((double)d0f));
        if (isinf(d1f)) return INFINITY; if (!isnan(d1f)) LASSQ_UPDATE(fabs((double)d1f));
        if (isinf(d2f)) return INFINITY; if (!isnan(d2f)) LASSQ_UPDATE(fabs((double)d2f));
        if (isinf(d3f)) return INFINITY; if (!isnan(d3f)) LASSQ_UPDATE(fabs((double)d3f));
    }
    
    for (; i < n; ++i) {
        float d = bfloat16_to_float32(a[i]) - bfloat16_to_float32(b[i]);
        if (isinf(d)) return INFINITY;
        if (!isnan(d)) LASSQ_UPDATE(fabs((double)d));
    }
    
    double sum_sq = (scale == 0.0) ? 0.0 : (scale * scale * ssq);
    double out = use_sqrt ? sqrt(sum_sq) : sum_sq;
    return (float)out;
}

float bfloat16_distance_l2_cpu (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_cpu(v1, v2, n, true);
}

float bfloat16_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return bfloat16_distance_l2_impl_cpu(v1, v2, n, false);
}

float bfloat16_distance_cosine_cpu(const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    float dot = 0.0f, norm_x = 0.0f, norm_y = 0.0f;
    int i = 0;

    // unroll the loop 4 times
    for (; i <= n - 4; i += 4) {
        float x0 = bfloat16_to_float32(a[i    ]), y0 = bfloat16_to_float32(b[i    ]);
        float x1 = bfloat16_to_float32(a[i + 1]), y1 = bfloat16_to_float32(b[i + 1]);
        float x2 = bfloat16_to_float32(a[i + 2]), y2 = bfloat16_to_float32(b[i + 2]);
        float x3 = bfloat16_to_float32(a[i + 3]), y3 = bfloat16_to_float32(b[i + 3]);

        // accumulate (fmaf may fuse on capable CPUs)
        dot    = fmaf(x0, y0, dot);
        dot    = fmaf(x1, y1, dot);
        dot    = fmaf(x2, y2, dot);
        dot    = fmaf(x3, y3, dot);

        norm_x = fmaf(x0, x0, norm_x);
        norm_x = fmaf(x1, x1, norm_x);
        norm_x = fmaf(x2, x2, norm_x);
        norm_x = fmaf(x3, x3, norm_x);

        norm_y = fmaf(y0, y0, norm_y);
        norm_y = fmaf(y1, y1, norm_y);
        norm_y = fmaf(y2, y2, norm_y);
        norm_y = fmaf(y3, y3, norm_y);
    }

    // tail loop
    for (; i < n; ++i) {
        float x = bfloat16_to_float32(a[i]);
        float y = bfloat16_to_float32(b[i]);
        dot    = fmaf(x, y, dot);
        norm_x = fmaf(x, x, norm_x);
        norm_y = fmaf(y, y, norm_y);
    }

    // max distance if one vector is zero
    if (norm_x == 0.0f || norm_y == 0.0f) {
        return 1.0f;
    }

    float cosine_similarity = dot / (sqrtf(norm_x) * sqrtf(norm_y));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}

float bfloat16_distance_dot_cpu (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;
    
    float dot = 0.0f;
    int i = 0;
    
    // unroll the loop 4 times
    for (; i <= n - 4; i += 4) {
        float x0 = bfloat16_to_float32(a[i    ]), y0 = bfloat16_to_float32(b[i    ]);
        float x1 = bfloat16_to_float32(a[i + 1]), y1 = bfloat16_to_float32(b[i + 1]);
        float x2 = bfloat16_to_float32(a[i + 2]), y2 = bfloat16_to_float32(b[i + 2]);
        float x3 = bfloat16_to_float32(a[i + 3]), y3 = bfloat16_to_float32(b[i + 3]);
        
        // fmaf often maps to a fused multiply-add, improving precision/speed
        dot = fmaf(x0, y0, dot);
        dot = fmaf(x1, y1, dot);
        dot = fmaf(x2, y2, dot);
        dot = fmaf(x3, y3, dot);
    }
    
    // tail loop
    for (; i < n; ++i) {
        float x = bfloat16_to_float32(a[i]);
        float y = bfloat16_to_float32(b[i]);
        dot = fmaf(x, y, dot);
    }
    
    return -dot;
}

float bfloat16_distance_l1_cpu (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;
    
    float sum = 0.0f;
    int i = 0;
    
    // unroll the loop 4 times
    for (; i <= n - 4; i += 4) {
        float a0 = bfloat16_to_float32(a[i    ]), b0 = bfloat16_to_float32(b[i    ]);
        float a1 = bfloat16_to_float32(a[i + 1]), b1 = bfloat16_to_float32(b[i + 1]);
        float a2 = bfloat16_to_float32(a[i + 2]), b2 = bfloat16_to_float32(b[i + 2]);
        float a3 = bfloat16_to_float32(a[i + 3]), b3 = bfloat16_to_float32(b[i + 3]);
        
        sum += fabsf(a0 - b0);
        sum += fabsf(a1 - b1);
        sum += fabsf(a2 - b2);
        sum += fabsf(a3 - b3);
    }
    
    // tail loop
    for (; i < n; ++i) {
        float da = bfloat16_to_float32(a[i]);
        float db = bfloat16_to_float32(b[i]);
        sum += fabsf(da - db);
    }
    
    return sum;
}

// MARK: - FLOAT16 -

static inline float float16_distance_l2_impl_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
    const uint16_t *a = (const uint16_t *)v1;  /* float16 bits */
    const uint16_t *b = (const uint16_t *)v2;
    
    double scale = 0.0;
    double ssq   = 1.0;
    int i = 0;
    
    /* main loop, unrolled by 4 */
    for (; i <= n - 4; i += 4) {
        uint16_t a0=a[i], a1=a[i+1], a2=a[i+2], a3=a[i+3];
        uint16_t b0=b[i], b1=b[i+1], b2=b[i+2], b3=b[i+3];
        
        /* If any pair involves an infinity not matched with same-signed infinity → +Inf */
        if ((f16_is_inf(a0)||f16_is_inf(b0)) && !(f16_is_inf(a0)&&f16_is_inf(b0)&&f16_sign(a0)==f16_sign(b0))) return INFINITY;
        if ((f16_is_inf(a1)||f16_is_inf(b1)) && !(f16_is_inf(a1)&&f16_is_inf(b1)&&f16_sign(a1)==f16_sign(b1))) return INFINITY;
        if ((f16_is_inf(a2)||f16_is_inf(b2)) && !(f16_is_inf(a2)&&f16_is_inf(b2)&&f16_sign(a2)==f16_sign(b2))) return INFINITY;
        if ((f16_is_inf(a3)||f16_is_inf(b3)) && !(f16_is_inf(a3)&&f16_is_inf(b3)&&f16_sign(a3)==f16_sign(b3))) return INFINITY;
        
        /* NaN lanes contribute 0 */
        if (!f16_is_nan(a0) && !f16_is_nan(b0)) { double d = (double)float16_to_float32(a0) - (double)float16_to_float32(b0); LASSQ_UPDATE(fabs(d)); }
        if (!f16_is_nan(a1) && !f16_is_nan(b1)) { double d = (double)float16_to_float32(a1) - (double)float16_to_float32(b1); LASSQ_UPDATE(fabs(d)); }
        if (!f16_is_nan(a2) && !f16_is_nan(b2)) { double d = (double)float16_to_float32(a2) - (double)float16_to_float32(b2); LASSQ_UPDATE(fabs(d)); }
        if (!f16_is_nan(a3) && !f16_is_nan(b3)) { double d = (double)float16_to_float32(a3) - (double)float16_to_float32(b3); LASSQ_UPDATE(fabs(d)); }
    }
    
    /* tail */
    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((f16_is_inf(ai)||f16_is_inf(bi)) && !(f16_is_inf(ai)&&f16_is_inf(bi)&&f16_sign(ai)==f16_sign(bi))) return INFINITY;
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        double d = (double)float16_to_float32(ai) - (double)float16_to_float32(bi);
        LASSQ_UPDATE(fabs(d));
    }
    
    double sum_sq = (scale == 0.0) ? 0.0 : (scale * scale * ssq);
    double out = use_sqrt ? sqrt(sum_sq) : sum_sq;
    return (float)out;
}

float float16_distance_l2_cpu (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_cpu(v1, v2, n, true);
}

float float16_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return float16_distance_l2_impl_cpu(v1, v2, n, false);
}

float float16_distance_l1_cpu (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    double sum = 0.0;
    int i = 0;

    for (; i <= n - 4; i += 4) {
        uint16_t a0=a[i], a1=a[i+1], a2=a[i+2], a3=a[i+3];
        uint16_t b0=b[i], b1=b[i+1], b2=b[i+2], b3=b[i+3];

        /* Inf differences yield +Inf */
        if ((f16_is_inf(a0)||f16_is_inf(b0)) && !(f16_is_inf(a0)&&f16_is_inf(b0)&&f16_sign(a0)==f16_sign(b0))) return INFINITY;
        if ((f16_is_inf(a1)||f16_is_inf(b1)) && !(f16_is_inf(a1)&&f16_is_inf(b1)&&f16_sign(a1)==f16_sign(b1))) return INFINITY;
        if ((f16_is_inf(a2)||f16_is_inf(b2)) && !(f16_is_inf(a2)&&f16_is_inf(b2)&&f16_sign(a2)==f16_sign(b2))) return INFINITY;
        if ((f16_is_inf(a3)||f16_is_inf(b3)) && !(f16_is_inf(a3)&&f16_is_inf(b3)&&f16_sign(a3)==f16_sign(b3))) return INFINITY;

        if (!f16_is_nan(a0) && !f16_is_nan(b0)) sum += fabs((double)float16_to_float32(a0) - (double)float16_to_float32(b0));
        if (!f16_is_nan(a1) && !f16_is_nan(b1)) sum += fabs((double)float16_to_float32(a1) - (double)float16_to_float32(b1));
        if (!f16_is_nan(a2) && !f16_is_nan(b2)) sum += fabs((double)float16_to_float32(a2) - (double)float16_to_float32(b2));
        if (!f16_is_nan(a3) && !f16_is_nan(b3)) sum += fabs((double)float16_to_float32(a3) - (double)float16_to_float32(b3));
    }

    for (; i < n; ++i) {
        uint16_t ai=a[i], bi=b[i];
        if ((f16_is_inf(ai)||f16_is_inf(bi)) && !(f16_is_inf(ai)&&f16_is_inf(bi)&&f16_sign(ai)==f16_sign(bi))) return INFINITY;
        if (f16_is_nan(ai) || f16_is_nan(bi)) continue;
        sum += fabs((double)float16_to_float32(ai) - (double)float16_to_float32(bi));
    }

    return (float)sum;
}

float float16_distance_dot_cpu (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    double dot = 0.0;
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float x0 = float16_to_float32(a[i    ]), y0 = float16_to_float32(b[i    ]);
        float x1 = float16_to_float32(a[i + 1]), y1 = float16_to_float32(b[i + 1]);
        float x2 = float16_to_float32(a[i + 2]), y2 = float16_to_float32(b[i + 2]);
        float x3 = float16_to_float32(a[i + 3]), y3 = float16_to_float32(b[i + 3]);

        /* Skip NaN lanes */
        if (!isnan(x0) && !isnan(y0)) { double p = (double)x0 * (double)y0; if (isinf(p)) return (p>0)? -INFINITY : INFINITY; dot += p; }
        if (!isnan(x1) && !isnan(y1)) { double p = (double)x1 * (double)y1; if (isinf(p)) return (p>0)? -INFINITY : INFINITY; dot += p; }
        if (!isnan(x2) && !isnan(y2)) { double p = (double)x2 * (double)y2; if (isinf(p)) return (p>0)? -INFINITY : INFINITY; dot += p; }
        if (!isnan(x3) && !isnan(y3)) { double p = (double)x3 * (double)y3; if (isinf(p)) return (p>0)? -INFINITY : INFINITY; dot += p; }
    }

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

float float16_distance_cosine_cpu (const void *v1, const void *v2, int n) {
    const uint16_t *a = (const uint16_t *)v1;
    const uint16_t *b = (const uint16_t *)v2;

    double dot = 0.0, nx = 0.0, ny = 0.0;
    int i = 0;

    for (; i <= n - 4; i += 4) {
        float x0 = float16_to_float32(a[i    ]), y0 = float16_to_float32(b[i    ]);
        float x1 = float16_to_float32(a[i + 1]), y1 = float16_to_float32(b[i + 1]);
        float x2 = float16_to_float32(a[i + 2]), y2 = float16_to_float32(b[i + 2]);
        float x3 = float16_to_float32(a[i + 3]), y3 = float16_to_float32(b[i + 3]);

        if (!isnan(x0) && !isnan(y0)) { double xd=x0, yd=y0; if (isinf(xd)||isinf(yd)) return 1.0f; dot += xd*yd; nx += xd*xd; ny += yd*yd; }
        if (!isnan(x1) && !isnan(y1)) { double xd=x1, yd=y1; if (isinf(xd)||isinf(yd)) return 1.0f; dot += xd*yd; nx += xd*xd; ny += yd*yd; }
        if (!isnan(x2) && !isnan(y2)) { double xd=x2, yd=y2; if (isinf(xd)||isinf(yd)) return 1.0f; dot += xd*yd; nx += xd*xd; ny += yd*yd; }
        if (!isnan(x3) && !isnan(y3)) { double xd=x3, yd=y3; if (isinf(xd)||isinf(yd)) return 1.0f; dot += xd*yd; nx += xd*xd; ny += yd*yd; }
    }

    for (; i < n; ++i) {
        float x = float16_to_float32(a[i]);
        float y = float16_to_float32(b[i]);
        if (isnan(x) || isnan(y)) continue;
        if (isinf((double)x) || isinf((double)y)) return 1.0f;
        double xd=x, yd=y;
        dot += xd*yd; nx += xd*xd; ny += yd*yd;
    }

    double denom = sqrt(nx) * sqrt(ny);
    if (!(denom > 0.0) || !isfinite(denom) || !isfinite(dot)) return 1.0f;

    double cosv = dot / denom;
    if (cosv > 1.0) cosv = 1.0;
    if (cosv < -1.0) cosv = -1.0;
    return (float)(1.0 - cosv);
}

// MARK: - UINT8 -

static inline float uint8_distance_l2_impl_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
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
    return uint8_distance_l2_impl_cpu(v1, v2, n, true);
}

float uint8_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return uint8_distance_l2_impl_cpu(v1, v2, n, false);
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
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
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

float int8_distance_l2_impl_cpu (const void *v1, const void *v2, int n, bool use_sqrt) {
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
    return int8_distance_l2_impl_cpu(v1, v2, n, true);
}

float int8_distance_l2_squared_cpu (const void *v1, const void *v2, int n) {
    return int8_distance_l2_impl_cpu(v1, v2, n, false);
}

float int8_distance_cosine_cpu (const void *v1, const void *v2, int n) {
    const int8_t *a = (const int8_t *)v1;
    const int8_t *b = (const int8_t *)v2;
    
    int32_t dot = 0;
    int32_t norm_a2 = 0;
    int32_t norm_b2 = 0;
    
    int i = 0;
    for (; i <= n - 4; i += 4) {
        int32_t a0 = a[i + 0], b0 = b[i + 0];
        int32_t a1 = a[i + 1], b1 = b[i + 1];
        int32_t a2 = a[i + 2], b2 = b[i + 2];
        int32_t a3 = a[i + 3], b3 = b[i + 3];

        dot     += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a2 += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b2 += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
    }

    // tail loop
    for (; i < n; ++i) {
        int32_t ai = a[i];
        int32_t bi = b[i];
        dot     += ai * bi;
        norm_a2 += ai * ai;
        norm_b2 += bi * bi;
    }

    if (norm_a2 == 0 || norm_b2 == 0) {
        return 1.0f;
    }

    float cosine_similarity = dot / (sqrtf((float)norm_a2) * sqrtf((float)norm_b2));
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
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

// MARK: - BIT -

static inline int popcount64(uint64_t x) {
    #if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
    #else
    // fallback: bit manipulation
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (x * 0x0101010101010101ULL) >> 56;
    #endif
}

float bit1_distance_hamming_cpu (const void *v1, const void *v2, int n) {
    const uint8_t *a = (const uint8_t *)v1;
    const uint8_t *b = (const uint8_t *)v2;
    
    int distance = 0;
    int i = 0;
    
    // process 8 bytes at a time
    for (; i + 8 <= n; i += 8) {
        uint64_t xa, xb;
        memcpy(&xa, a + i, sizeof(uint64_t));
        memcpy(&xb, b + i, sizeof(uint64_t));
        distance += popcount64(xa ^ xb);
    }
    
    // handle remainder
    for (; i < n; i++) {
        distance += popcount64(a[i] ^ b[i]);
    }
    
    return (float)distance;
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

    void run_cpuid(int leaf, int subleaf, int result[4]) {
        #if defined(_MSC_VER)
                __cpuidex(result, leaf, subleaf);
        #else
                __cpuid_count(leaf, subleaf, result[0], result[1], result[2], result[3]);
        #endif
    }

    uint64_t run_xgetbv(uint32_t xcr) {
        #if defined(_MSC_VER)
        return _xgetbv(xcr);
        #else
        uint32_t eax, edx;
        // xgetbv instruction: reads XCR specified by ecx into edx:eax
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
        return ((uint64_t)edx << 32) | eax;
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

    bool cpu_supports_avx512(void) {
        #if FORCE_AVX512
                return true;
        #else
            int cpu_info[4];

            // 1. Check maximum CPUID leaf
            run_cpuid(0, 0, cpu_info);
            if (cpu_info[0] < 7) return false; // CPU too old

            // 2. Check for OSXSAVE (Leaf 1, ECX bit 27)
            // This implies the processor supports XSAVE/XRSTOR
            run_cpuid(1, 0, cpu_info);
            if (!(cpu_info[2] & (1 << 27))) return false;

            // 3. Check XCR0 for OS support of ZMM registers
            // We need bits 5 (opmask), 6 (ZMM_Hi256), and 7 (Hi16_ZMM) to be 1.
            // Also usually need bit 1 (SSE) and 2 (AVX)
            uint64_t xcr0 = run_xgetbv(0);
            uint64_t avx512_state = (1 << 5) | (1 << 6) | (1 << 7);
            if ((xcr0 & avx512_state) != avx512_state) return false;

            // 4. Check hardware feature bits (Leaf 7, Subleaf 0)
            run_cpuid(7, 0, cpu_info);

            // EBX Bit 16: AVX512F (Foundation)
            bool has_avx512f = (cpu_info[1] & (1 << 16));

            // Optional: Check for BW (Byte/Word) and VL (Vector Length)
            // Many algorithms (like your integer distance code) need these.
            bool has_avx512bw = (cpu_info[1] & (1 << 30));
            bool has_avx512vl = (cpu_info[1] & (1 << 31));

            // EBX Bit 17: AVX512DQ, needed by _mm512_extractf32x8_ps in the f16/bf16 kernels
            bool has_avx512dq = (cpu_info[1] & (1 << 17));

            return has_avx512f && has_avx512bw && has_avx512vl && has_avx512dq;
        #endif
    }

    bool cpu_supports_sse2 (void) {
        int eax, ebx, ecx, edx;
        x86_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
        return (edx & (1 << 26)) != 0;  // SSE2
    }
#elif defined(__riscv) || defined(__riscv__)
    #include <sys/auxv.h>
    #define ISA_V_HWCAP (1 << ('v' - 'a'))
    
    bool cpu_supports_rvv (void) {    
        unsigned long hw_cap = getauxval(AT_HWCAP);
        return (hw_cap & ISA_V_HWCAP) != 0;
    }
#else
    // For ARM (NEON is always present on aarch64, runtime detection rarely needed)
    #if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    bool cpu_supports_neon (void) {
        return true;
    }
    #else
        #ifdef SQLITE_WASM_EXTRA_INIT
        bool cpu_supports_neon (void) {
            return false;
        }
        #else
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
        bool cpu_supports_neon (void) {
            #if defined(AT_HWCAP) && defined(HWCAP_NEON)
            return (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
            #else
            return false;
            #endif
        }
        #endif
    #endif
#endif

// MARK: -

static inline uint16_t turbo_lut3_index_cpu (const uint8_t *packed, int row, int packed_bytes) {
    size_t bit_pos = (size_t)row * 12u;
    size_t byte_pos = bit_pos / 8u;
    int shift = (int)(bit_pos % 8u);
    uint32_t word = 0;
    if ((int)byte_pos < packed_bytes) word |= packed[byte_pos];
    if ((int)byte_pos + 1 < packed_bytes) word |= (uint32_t)packed[byte_pos + 1] << 8;
    return (uint16_t)((word >> shift) & 0x0fffu);
}

// The TurboQuant scan is a chain of table lookups: one gather per row, and on any
// machine that is already about one load per cycle. There is nothing for SIMD to do -
// NEON has no gather at all, and the four per-backend copies this replaces were scalar
// gathers into a stack array plus a single vector add. What they were really buying was
// four parallel float lanes instead of one serial double accumulator, and four
// independent double accumulators buy the same parallelism without the accuracy loss:
// measured within 2% of the NEON version at every bit width, and identical on every
// backend rather than differing by up to 1.5e-4 relative depending on which one ran.
float turbo_lut_dot_cpu (const uint8_t *packed, float scale, const float *query_lut, int lut_rows, int bits, int packed_bytes) {
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    int r = 0;

    if (bits == 3) {
        for (; r + 3 < lut_rows; r += 4) {
            acc0 += (double)query_lut[(size_t)(r + 0) * 4096u + turbo_lut3_index_cpu(packed, r + 0, packed_bytes)];
            acc1 += (double)query_lut[(size_t)(r + 1) * 4096u + turbo_lut3_index_cpu(packed, r + 1, packed_bytes)];
            acc2 += (double)query_lut[(size_t)(r + 2) * 4096u + turbo_lut3_index_cpu(packed, r + 2, packed_bytes)];
            acc3 += (double)query_lut[(size_t)(r + 3) * 4096u + turbo_lut3_index_cpu(packed, r + 3, packed_bytes)];
        }
        for (; r < lut_rows; ++r) {
            acc0 += (double)query_lut[(size_t)r * 4096u + turbo_lut3_index_cpu(packed, r, packed_bytes)];
        }
    } else {
        (void)packed_bytes;
        for (; r + 3 < lut_rows; r += 4) {
            acc0 += (double)query_lut[(size_t)(r + 0) * 256u + packed[r + 0]];
            acc1 += (double)query_lut[(size_t)(r + 1) * 256u + packed[r + 1]];
            acc2 += (double)query_lut[(size_t)(r + 2) * 256u + packed[r + 2]];
            acc3 += (double)query_lut[(size_t)(r + 3) * 256u + packed[r + 3]];
        }
        for (; r < lut_rows; ++r) {
            acc0 += (double)query_lut[(size_t)r * 256u + packed[r]];
        }
    }

    return (float)(((acc0 + acc1) + (acc2 + acc3)) * (double)scale);
}

void init_cpu_functions (void) {
    distance_function_t cpu_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX] = {
        [VECTOR_DISTANCE_L2] = {
                [VECTOR_TYPE_F32] = float32_distance_l2_cpu,
                [VECTOR_TYPE_F16] = float16_distance_l2_cpu,
                [VECTOR_TYPE_BF16] = bfloat16_distance_l2_cpu,
                [VECTOR_TYPE_U8]  = uint8_distance_l2_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_l2_cpu,
            },
            [VECTOR_DISTANCE_SQUARED_L2] = {
                [VECTOR_TYPE_F32] = float32_distance_l2_squared_cpu,
                [VECTOR_TYPE_F16] = float16_distance_l2_squared_cpu,
                [VECTOR_TYPE_BF16] = bfloat16_distance_l2_squared_cpu,
                [VECTOR_TYPE_U8]  = uint8_distance_l2_squared_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_l2_squared_cpu,
            },
            [VECTOR_DISTANCE_COSINE] = {
                [VECTOR_TYPE_F32] = float32_distance_cosine_cpu,
                [VECTOR_TYPE_F16] = float16_distance_cosine_cpu,
                [VECTOR_TYPE_BF16] = bfloat16_distance_cosine_cpu,
                [VECTOR_TYPE_U8]  = uint8_distance_cosine_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_cosine_cpu,
            },
            [VECTOR_DISTANCE_DOT] = {
                [VECTOR_TYPE_F32] = float32_distance_dot_cpu,
                [VECTOR_TYPE_F16] = float16_distance_dot_cpu,
                [VECTOR_TYPE_BF16] = bfloat16_distance_dot_cpu,
                [VECTOR_TYPE_U8]  = uint8_distance_dot_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_dot_cpu,
            },
            [VECTOR_DISTANCE_L1] = {
                [VECTOR_TYPE_F32] = float32_distance_l1_cpu,
                [VECTOR_TYPE_F16] = float16_distance_l1_cpu,
                [VECTOR_TYPE_BF16] = bfloat16_distance_l1_cpu,
                [VECTOR_TYPE_U8]  = uint8_distance_l1_cpu,
                [VECTOR_TYPE_I8]  = int8_distance_l1_cpu,
            },
            [VECTOR_DISTANCE_HAMMING] = {
                [VECTOR_TYPE_BIT] = bit1_distance_hamming_cpu
            }
    };
    
    memcpy(dispatch_distance_table, cpu_table, sizeof(cpu_table));
    turbo_lut_dot_function = turbo_lut_dot_cpu;
    turbo_lut_backend_name = "CPU";
}

void init_distance_functions (bool force_cpu) {
    init_cpu_functions();
    if (force_cpu) return;
    
    // each backend reports whether its kernels were actually compiled into this build:
    // a tier whose ISA was not enabled at compile time installs nothing and we must keep
    // walking down, otherwise an AVX2-capable CPU would end up on the scalar fallback
    // even though the SSE2 kernels are available
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    bool installed = false;
    if (!installed && cpu_supports_avx512()) installed = init_distance_functions_avx512();
    if (!installed && cpu_supports_avx2()) installed = init_distance_functions_avx2();
    if (!installed && cpu_supports_sse2()) installed = init_distance_functions_sse2();
    #elif defined(__ARM_NEON) || defined(__aarch64__)
    if (cpu_supports_neon()) {
        init_distance_functions_neon();
    }
    #elif defined(__riscv) || defined(__riscv__)
    if (cpu_supports_rvv()) {
        init_distance_functions_rvv();
    }
    #endif
}
