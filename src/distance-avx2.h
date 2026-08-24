//
//  distance-avx2.h
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#ifndef __VECTOR_DISTANCE_AVX2__
#define __VECTOR_DISTANCE_AVX2__

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// returns true when the AVX2 kernels were compiled into this build
bool init_distance_functions_avx2 (void);
float turbo_lut_dot_avx2 (const uint8_t *packed, float scale, const float *query_lut, int lut_rows, int bits, int packed_bytes);

#endif
