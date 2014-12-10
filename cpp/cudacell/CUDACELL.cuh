#include <ostream>

#ifndef __CUDACELL_CUH__
#define __CUDACELL_CUH__
#include "HOOMDMath.h"
#include "Index1D.h"
#include "cudabox.h"
#include <stdio.h>
#include <assert.h>

namespace freud { namespace cudacell {

void CallCompute(unsigned int *p_array,
                 unsigned int *c_array,
                 unsigned int np,
                 unsigned int nc,
                 trajectory::CudaBox &box,
                 Index3D& cell_idx,
                 const float3 *points);

void createIDXArray(unsigned int **IDXArray, size_t memSize);

void createPointArray(float3 **IDXArray, size_t memSize);

void createTestArray(int **IDXArray);

// just overload dumb dumb

void freePointArray(float3 *IDXArray);

void freeIDXArray(unsigned int *IDXArray);

void freeTestArray(int *IDXArray);

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

}; }; // end namespace freud::cudapmft

#endif // _CUDACELL_CUH__
