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

#endif
