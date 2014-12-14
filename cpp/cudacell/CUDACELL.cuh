#include <ostream>

#ifndef __CUDACELL_CUH__
#define __CUDACELL_CUH__
// #include "HOOMDMath.h"
#include "Index1D.h"
#include "cudabox.h"
#include <stdio.h>
#include <assert.h>

namespace freud { namespace cudacell {

void cudaComputeCellList(unsigned int *p_array,
                         unsigned int *c_array,
                         unsigned int *it_array,
                         unsigned int np,
                         unsigned int nc,
                         trajectory::CudaBox &box,
                         Index3D& cell_idx,
                         const float3 *points);

void cudaComputeCellNeighbors(unsigned int *cell_neighbors,
                              uint3 celldim,
                              uint3 num_neighbors,
                              unsigned int nc,
                              unsigned int total_num_neighbors,
                              Index3D& cell_idx,
                              Index2D& thread_indexer);

void createArray(unsigned int **array, size_t memSize);

void freeArray(unsigned int *array);

void createArray(float3 **array, size_t memSize);

void freeArray(float3 *array);

void createArray(uint2 **array, size_t memSize);

void freeArray(uint2 *array);

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

}; }; // end namespace freud::cudapmft

#endif // _CUDACELL_CUH__
