#include <ostream>
#include "HOOMDMath.h"
#include "CUDACELL.cuh"

using namespace std;

namespace freud { namespace cudacell {

__global__ void computeCellList(unsigned int *p_array,
                                unsigned int *c_array,
                                unsigned int np,
                                unsigned int nc,
                                trajectory::CudaBox box,
                                Index3D cell_idx,
                                const float3 *points)
    {
    // determine particle being calculated
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // if (idx < np)
    //     {
    //     // get the point
    //     float3 point = points[idx];
    //     // determine cell for idx
    //     float3 alpha = box.makeFraction(point);
    //     uint3 c;
    //     c.x = floorf(alpha.x * float(cell_idx.getW()));
    //     c.x %= cell_idx.getW();
    //     c.y = floorf(alpha.y * float(cell_idx.getH()));
    //     c.y %= cell_idx.getH();
    //     c.z = floorf(alpha.z * float(cell_idx.getD()));
    //     c.z %= cell_idx.getD();
    //     unsigned int c_idx = cell_idx((int)c.x, (int)c.y, (int)c.z);
    //     p_array[idx] = idx;
    //     c_array[idx] = c_idx;
    //     }
    }

void CallCompute(unsigned int *p_array,
                 unsigned int *c_array,
                 unsigned int np,
                 unsigned int nc,
                 trajectory::CudaBox &box,
                 Index3D& cell_idx,
                 const float3 *points)
    {

    // define grid and block size
    int numThreadsPerBlock = 32;
    int numBlocks = (np / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

    // right now I am getting an invalid device pointer

    computeCellList<<< dimGrid, dimBlock >>>(p_array,
                                             c_array,
                                             np,
                                             nc,
                                             box,
                                             cell_idx,
                                             points);

    // block until the device has completed
    cudaDeviceSynchronize();

    // check if kernel execution generated an error
    checkCUDAError("kernel execution");
    }

void createIDXArray(unsigned int **IDXArray, size_t memSize)
    {
    cudaMallocManaged(IDXArray, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("createIDXArray");
    }

void createPointArray(float3 **IDXArray, size_t memSize)
    {
    cudaMallocManaged(IDXArray, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("createPointArray");
    }

void createTestArray(int **IDXArray)
    {
    cudaMallocManaged(IDXArray, sizeof(int));
    cudaDeviceSynchronize();
    checkCUDAError("createTestArray");
    }

void freeIDXArray(unsigned int *IDXArray)
    {
    cudaFree(IDXArray);
    checkCUDAError("freeIDXArray");
    }

void freePointArray(float3 *IDXArray)
    {
    cudaFree(IDXArray);
    checkCUDAError("freePointArray");
    }

void freeTestArray(int *IDXArray)
    {
    cudaFree(IDXArray);
    checkCUDAError("freeTestArray");
    }

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

}; }; // end namespace freud::cudacell
