//
//  distance-cpu.h
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#ifndef __VECTOR_DISTANCE_CPU__
#define __VECTOR_DISTANCE_CPU__

#include <inttypes.h>
#include <stdbool.h>

typedef enum {
    VECTOR_TYPE_F32 = 1,
    VECTOR_TYPE_F16,
    VECTOR_TYPE_BF16,
    VECTOR_TYPE_U8,
    VECTOR_TYPE_I8
} vector_type;
#define VECTOR_TYPE_MAX         6

typedef enum {
    VECTOR_QUANT_8BIT = 1
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

#endif
