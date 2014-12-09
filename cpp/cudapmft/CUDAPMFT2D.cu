#include <ostream>
#include "CUDAPMFT2D.cuh"

using namespace std;

namespace freud { namespace cudapmft {

// Part 3 of 5: implement the kernel
__global__ void myFirstKernel(unsigned int *pmftArray, unsigned int arrSize)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < arrSize)
        pmftArray[idx] += (unsigned int)idx;
}

void CallMyFirstKernel(unsigned int *pmftArray, unsigned int arrSize)
    {

    // define grid and block size
    int numThreadsPerBlock = 32;
    int numBlocks = (arrSize / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    myFirstKernel<<< dimGrid, dimBlock >>>( pmftArray, arrSize );

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
    }

void freePMFTArray(unsigned int **pmftArray)
    {
    cudaFree(pmftArray);
    }

void createCudaArray(float **cudaArray, size_t memSize)
    {
    cudaMallocManaged(cudaArray, memSize);
    cudaDeviceSynchronize();
    }

void freeCudaArray(float **cudaArray)
    {
    cudaFree(cudaArray);
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
