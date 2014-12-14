#include <ostream>

#ifndef __CUDAPMFT2D_CUH__
#define __CUDAPMFT2D_CUH__
#include "HOOMDMath.h"
#include "cudabox.h"
#include <stdio.h>
#include <assert.h>

namespace freud { namespace cudapmft {

void cudaComputePCF(unsigned int *pmftArray,
                    int nbins_x,
                    int nbins_y,
                    trajectory::CudaBox &box,
                    float max_x,
                    float max_y,
                    float dx,
                    float dy,
                    const unsigned int *cc,
                    float3 *ref_points,
                    float *ref_orientations,
                    unsigned int n_ref,
                    float3 *points,
                    float *orientations,
                    unsigned int n_p);

void createPMFTArray(unsigned int **pmftArray, unsigned int &arrSize, size_t &memSize, unsigned int nbins_x, unsigned int nbins_y);

void createArray(float **array, size_t memSize);

void createArray(float3 **array, size_t memSize);

void freeArray(unsigned int *array);

void freeArray(float *array);

void freeArray(float3 *array);

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

}; }; // end namespace freud::cudapmft

#endif // _CUDAPMFTXY2D_CUH__
