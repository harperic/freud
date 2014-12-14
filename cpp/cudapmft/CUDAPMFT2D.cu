#include <ostream>
#include "CUDAPMFT2D.cuh"

using namespace std;

namespace freud { namespace cudapmft {

// Part 3 of 5: implement the kernel
__global__ void computePCF(unsigned int *pmftArray,
                           unsigned int arrSize,
                           trajectory::CudaBox box)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int test = (unsigned int)box.getLx();
    if (idx < arrSize)
        // pmftArray[idx] += (unsigned int)idx + test;
        pmftArray[idx] = (unsigned int)idx;
}

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
                    unsigned int n_p)
    {
    unsigned int arrSize = (unsigned int)(nbins_x*nbins_y);
    // define grid and block size
    int numThreadsPerBlock = 32;
    int numBlocks = (arrSize / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    computePCF<<< dimGrid, dimBlock >>>( pmftArray, arrSize, box );

    // block until the device has completed
    cudaDeviceSynchronize();

    // check if kernel execution generated an error
    checkCUDAError("kernel execution");
    }

// just write as a template function
// this appears to either be non-trivial or not possible
void createPMFTArray(unsigned int **pmftArray, unsigned int &arrSize, size_t &memSize, unsigned int nbins_x, unsigned int nbins_y)
    {
    arrSize = nbins_x * nbins_y;
    memSize = sizeof(unsigned int) * arrSize;
    cudaMallocManaged(pmftArray, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("create PMFT array");
    }

void createArray(float **array, size_t memSize)
    {
    cudaMallocManaged(array, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("create float array");
    }

void createArray(unsigned int **array, size_t memSize)
    {
    cudaMallocManaged(array, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("create unsigned int array");
    }

void createArray(float3 **array, size_t memSize)
    {
    cudaMallocManaged(array, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("create float3 array");
    }

void freeArray(unsigned int *array)
    {
    cudaFree(array);
    cudaDeviceSynchronize();
    checkCUDAError("free unsigned int array");
    }

void freeArray(float *array)
    {
    cudaFree(array);
    cudaDeviceSynchronize();
    checkCUDAError("free float array");
    }

void freeArray(float3 *array)
    {
    cudaFree(array);
    cudaDeviceSynchronize();
    checkCUDAError("free float3 array");
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

}; }; // end namespace freud::cudapmft
