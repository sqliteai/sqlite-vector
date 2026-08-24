//
//  distance-avx512.h
//  sqlitevector
//
//  Created by Teddy Albina on 17/12/25.
//

#ifndef __VECTOR_DISTANCE_AVX512__
#define __VECTOR_DISTANCE_AVX512__

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// returns true when the AVX512 kernels were compiled into this build
bool init_distance_functions_avx512 (void);
float turbo_lut_dot_avx512 (const uint8_t *packed, float scale, const float *query_lut, int lut_rows, int bits, int packed_bytes);

#endif
