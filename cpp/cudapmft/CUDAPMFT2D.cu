#include <ostream>
#include "CUDAPMFT2D.cuh"

using namespace std;

namespace freud { namespace cudapmft {

// Part 3 of 5: implement the kernel
__global__ void computePCF(unsigned int *pmftArray,
                           unsigned int nbins_x,
                           unsigned int nbins_y,
                           trajectory::CudaBox box,
                           float max_x,
                           float max_y,
                           float dx,
                           float dy,
                           const unsigned int *pl,
                           const unsigned int *cl,
                           const unsigned int *nl,
                           const unsigned int *it,
                           const int total_num_neighbors,
                           Index2D neighbor_indexer,
                           float3 *ref_points,
                           float *ref_orientations,
                           unsigned int n_ref,
                           float3 *points,
                           float *orientations,
                           unsigned int n_p,
                           unsigned int n_c)
{
    // get the particle index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // precalc some values for faster computation within the loop
    float dx_inv = 1.0f / dx;
    float dy_inv = 1.0f / dy;
    float cos_theta = cosf(-ref_orientations[idx]);
    float sin_theta = sinf(-ref_orientations[idx]);

    Index2D b_i = Index2D(nbins_x, nbins_y);
    Index2D c_i = Index2D(2, n_c);
    if (idx < (int)n_ref)
        {
        // get the pos/orientation of the particle
        float3 ref_pos = ref_points[idx];
        // get the cell of the particle
        unsigned int cell_idx = pl[idx];
        // loop over cell neighbors
        for (int i = 0; i < total_num_neighbors; i++)
            {
            // currently, the retrieval of idx is not correct
            // get the neighbor of the cell
            int neigh_idx = (int)nl[neighbor_indexer((unsigned int)i, cell_idx)];
            // get the start/stop idxs of this cell
            int start_idx = (int)it[2*neigh_idx + 0];
            int stop_idx = (int)it[2*neigh_idx + 1];
            // for each particle in the neighbor cell
            for (int j = start_idx; j < stop_idx; j++)
                {
                // get the particle idx
                int comp_idx = cl[c_i(0,(unsigned int)j)];
                // get the pos/orientation of the particle
                float3 pos = points[comp_idx];
                // create the delta vector
                float3 delta = box.wrap(make_float3(pos.x - ref_pos.x,
                                                    pos.y - ref_pos.y,
                                                    pos.z - ref_pos.z));
                float rsq = (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
                if (rsq < 1e-6)
                    {
                    continue;
                    }
                // rotate the interparticle vector
                float x = delta.x*cos_theta - delta.y*sin_theta + max_x;
                float y = delta.x*sin_theta + delta.y*cos_theta + max_y;
                // find the bin to increment
                unsigned int binx = (unsigned int)floor(x * dx_inv);
                unsigned int biny = (unsigned int)floor(y * dy_inv);
                // increment the bin
                if ((binx < (int)nbins_x) && (biny < (int)nbins_y))
                    {
                    // printf("Incrementing bins\n");
                    atomicAdd(&pmftArray[(int)b_i((unsigned int)binx, (unsigned int)biny)], 1);
                    }
                }
            }
        }
}

void cudaComputePCF(unsigned int *pmftArray,
                    unsigned int nbins_x,
                    unsigned int nbins_y,
                    trajectory::CudaBox &box,
                    float max_x,
                    float max_y,
                    float dx,
                    float dy,
                    const unsigned int *pl,
                    const unsigned int *cl,
                    const unsigned int *nl,
                    const unsigned int *it,
                    const int total_num_neighbors,
                    const Index2D& neighbor_indexer,
                    float3 *ref_points,
                    float *ref_orientations,
                    unsigned int n_ref,
                    float3 *points,
                    float *orientations,
                    unsigned int n_p,
                    unsigned int n_c)
    {
    // define grid and block size
    int numThreadsPerBlock = 32;
    int numBlocks = (n_ref / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    computePCF<<< dimGrid, dimBlock >>>( pmftArray,
                                         nbins_x,
                                         nbins_y,
                                         box,
                                         max_x,
                                         max_y,
                                         dx,
                                         dy,
                                         pl,
                                         cl,
                                         nl,
                                         it,
                                         total_num_neighbors,
                                         neighbor_indexer,
                                         ref_points,
                                         ref_orientations,
                                         n_ref,
                                         points,
                                         orientations,
                                         n_p,
                                         n_c);

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
