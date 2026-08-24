//
//  distance-rvv.h
//  sqlitevector
//
//  Created by Afonso Bordado on 2026/02/19.
//

#ifndef __VECTOR_DISTANCE_RVV__
#define __VECTOR_DISTANCE_RVV__

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// returns true when the RVV kernels were compiled into this build
bool init_distance_functions_rvv (void);
float turbo_lut_dot_rvv (const uint8_t *packed, float scale, const float *query_lut, int lut_rows, int bits, int packed_bytes);

#endif
