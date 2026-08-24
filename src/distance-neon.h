//
//  distance-neon.h
//  sqlitevector
//
//  Created by Marco Bambini on 20/06/25.
//

#ifndef __VECTOR_DISTANCE_NEON__
#define __VECTOR_DISTANCE_NEON__

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// returns true when the NEON kernels were compiled into this build
bool init_distance_functions_neon (void);
float turbo_lut_dot_neon (const uint8_t *packed, float scale, const float *query_lut, int lut_rows, int bits, int packed_bytes);

#endif
