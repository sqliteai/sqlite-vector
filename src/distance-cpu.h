//
//  distance-cpu.h
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#ifndef __VECTOR_DISTANCE_CPU__
#define __VECTOR_DISTANCE_CPU__

#include "fp16/fp16.h"
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

// Detect builtin bit_cast
#ifndef HAVE_BUILTIN_BIT_CAST
  /* Only use __builtin_bit_cast if the compiler has it AND
     we're compiling as C++ (GCC 11+) or as a C standard that supports it (C23+). */
  #if defined(__has_builtin)
    #if __has_builtin(__builtin_bit_cast)
      #if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
        #define HAVE_BUILTIN_BIT_CAST 1
      #endif
    #endif
  #endif

  /* GCC note: in GCC 11–13, __builtin_bit_cast exists for C++ but NOT for C. */
  #if !defined(HAVE_BUILTIN_BIT_CAST) && defined(__GNUC__) && !defined(__clang__) && defined(__cplusplus)
    #if __GNUC__ >= 11
      #define HAVE_BUILTIN_BIT_CAST 1
    #endif
  #endif
#endif

typedef enum {
    VECTOR_TYPE_F32 = 1,
    VECTOR_TYPE_F16,
    VECTOR_TYPE_BF16,
    VECTOR_TYPE_U8,
    VECTOR_TYPE_I8
} vector_type;
#define VECTOR_TYPE_MAX         6

typedef enum {
    VECTOR_QUANT_AUTO = 0,
    VECTOR_QUANT_U8BIT = 1,
    VECTOR_QUANT_S8BIT = 2
} vector_qtype;

typedef enum {
    VECTOR_DISTANCE_L2 = 1,
    VECTOR_DISTANCE_SQUARED_L2,
    VECTOR_DISTANCE_COSINE,
    VECTOR_DISTANCE_DOT,
    VECTOR_DISTANCE_L1,
} vector_distance;
#define VECTOR_DISTANCE_MAX     6

typedef float (*distance_function_t)(const void *v1, const void *v2, int n);

// ENTRYPOINT
void init_distance_functions (bool force_cpu);

// MARK: - FLOAT16/BFLOAT16 -
// typedef uint16_t bfloat16_t;    // don't typedef to bfloat16_t to avoid mix with <arm_neon.h>’s native bfloat16_t

// float <-> uint32_t bit casts
static inline uint32_t f32_to_bits (float f) {
    #if defined(HAVE_BUILTIN_BIT_CAST)
    return __builtin_bit_cast(uint32_t, f);
    #else
    union { float f; uint32_t u; } v = { .f = f };
    return v.u;
    #endif
}

static inline float bits_to_f32 (uint32_t u) {
    #if defined(HAVE_BUILTIN_BIT_CAST)
    return __builtin_bit_cast(float, u);
    #else
    union { uint32_t u; float f; } v = { .u = u };
    return v.f;
    #endif
}

// bfloat16 (stored as uint16_t) -> float32, and back (RNE)
static inline bool bfloat16_is_nan(uint16_t h) {      /* exp==0xFF && frac!=0 */
    return ((h & 0x7F80u) == 0x7F80u) && ((h & 0x007Fu) != 0);
}
static inline bool bfloat16_is_inf(uint16_t h) {      /* exp==0xFF && frac==0 */
    return ((h & 0x7F80u) == 0x7F80u) && ((h & 0x007Fu) == 0);
}
static inline bool bfloat16_is_zero(uint16_t h) {     /* ±0 */
    return (h & 0x7FFFu) == 0;
}
static inline int bfloat16_sign(uint16_t h) {
    return (h >> 15) & 1;
}
static inline float bfloat16_to_float32(uint16_t bf) {
    return bits_to_f32((uint32_t)bf << 16);
}
static inline uint16_t float32_to_bfloat16(float f) {
    uint32_t x = f32_to_bits(f);
    uint32_t lsb = (x >> 16) & 1u;      /* ties-to-even */
    uint32_t rnd = 0x7FFFu + lsb;
    return (uint16_t)((x + rnd) >> 16);
}

// ---- float16 (binary16) classifiers (work on raw uint16_t bits)
static inline bool f16_is_nan(uint16_t h) {      /* exp==0x1F && frac!=0 */
    return ( (h & 0x7C00u) == 0x7C00u ) && ((h & 0x03FFu) != 0);
}
static inline bool f16_is_inf(uint16_t h) {      /* exp==0x1F && frac==0 */
    return ( (h & 0x7C00u) == 0x7C00u ) && ((h & 0x03FFu) == 0);
}
static inline int  f16_sign(uint16_t h) {
    return (h >> 15) & 1;
}
static inline bool f16_is_zero(uint16_t h) {     /* ±0 */
    return (h & 0x7FFFu) == 0;
}
static inline uint16_t float32_to_float16 (float f) {
    return fp16_ieee_from_fp32_value(f);
}
static inline float float16_to_float32 (uint16_t h) {
    return fp16_ieee_to_fp32_value(h);
}

#endif
