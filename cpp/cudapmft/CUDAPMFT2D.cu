#include <ostream>
#include "CUDAPMFT2D.cuh"

using namespace std;

namespace freud { namespace cudapmft {

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
                           Index3D cell_indexer,
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
        // avoids use of point list and may/should save memory
        float3 alpha = box.makeFraction(ref_pos);
        uint3 c;
        c.x = floorf(alpha.x * float(cell_indexer.getW()));
        c.x %= cell_indexer.getW();
        c.y = floorf(alpha.y * float(cell_indexer.getH()));
        c.y %= cell_indexer.getH();
        c.z = floorf(alpha.z * float(cell_indexer.getD()));
        c.z %= cell_indexer.getD();
        // unsigned int cell_idx = pl[idx];
        unsigned int cell_idx = cell_indexer(c.x, c.y, c.z);
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
                float binx = floorf(x * dx_inv);
                float biny = floorf(y * dy_inv);
                if ((binx < 0) || (biny < 0))
                    {
                    continue;
                    }
                unsigned int ibinx = (unsigned int)binx;
                unsigned int ibiny = (unsigned int)biny;
                // increment the bin
                if ((ibinx < nbins_x) && (ibiny < nbins_y))
                    {
                    // printf("Incrementing bins\n");
                    atomicAdd(&pmftArray[b_i(ibinx, ibiny)], 1);
                    }
                }
            }
        }
    }

__global__ void cellPCF(unsigned int *pmftArray,
                          unsigned int nbins_x,
                          unsigned int nbins_y,
                          trajectory::CudaBox box,
                          float max_x,
                          float max_y,
                          float dx,
                          float dy,
                          float cell_width,
                          const unsigned int *pl,
                          const unsigned int *cl,
                          const unsigned int *nl,
                          const unsigned int *it,
                          const int total_num_neighbors,
                          Index3D cell_indexer,
                          Index2D neighbor_indexer,
                          float3 *ref_points,
                          float *ref_orientations,
                          unsigned int n_ref,
                          float3 *points,
                          float *orientations,
                          unsigned int n_p,
                          unsigned int n_c)
    {
    // get global index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // get the cell index
    int cell_idx = blockIdx.x;
    // get the local index
    int local_idx = threadIdx.x;
    // precompute inverses for faster computation
    float dx_inv = 1.0f / dx;
    float dy_inv = 1.0f / dy;
    // compute indexers
    Index2D b_i = Index2D(nbins_x, nbins_y);
    Index2D c_i = Index2D(2, n_c);
    // get the number of particles in the ref cell
    unsigned int num_ref_particles = it[2*cell_idx+1] - it[2*cell_idx+0];
    // for each cell neighbor
    for (unsigned int i = 0; i < total_num_neighbors; i++)
        {
        // get the cell to be checked
        unsigned int neigh_idx = nl[neighbor_indexer(i, cell_idx)];

        // load the cell into shared mem
        unsigned int num_particles = it[2*neigh_idx+1] - it[2*neigh_idx+0];
        // create the indexer
        Index2D thread_indexer = Index2D(num_particles, num_ref_particles);
        // now, populate the histogram
        // create the multipass indexer
        unsigned int n_pairs = num_ref_particles*num_ref_particles;
        unsigned int n_pass_threads = floorf(thread_indexer.getNumElements()/blockDim.x) + 1;
        Index2D multipass_indexer = Index2D(blockDim.x, n_pass_threads);

        for (unsigned int n_pass = 0; n_pass < n_pass_threads; n_pass++)
            {
            unsigned int my_index = multipass_indexer(local_idx, n_pass);
            // check to make sure the pair exists
            if (!(my_index < n_pairs))
                {
                break;
                }
            // get the pair index
            uint2 pair_idx = thread_indexer(my_index);
            // get the ref_point
            unsigned int ref_point = cl[it[2*cell_idx+0]+pair_idx.y];
            unsigned int check_point = cl[it[2*neigh_idx+0]+pair_idx.x];
            float3 delta = box.wrap(make_float3(points[check_point].x - ref_points[ref_point].x,
                                                points[check_point].y - ref_points[ref_point].y,
                                                points[check_point].z - ref_points[ref_point].z));
            float rsq = (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
            if (rsq < 1e-6)
                {
                continue;
                }
            // rotate the interparticle vector
            float cos_theta = cosf(-ref_orientations[ref_point]);
            float sin_theta = sinf(-ref_orientations[ref_point]);
            float x = delta.x*cos_theta - delta.y*sin_theta + max_x;
            float y = delta.x*sin_theta + delta.y*cos_theta + max_y;
            // find the bin to increment
            float binx = floorf(x * dx_inv);
            float biny = floorf(y * dy_inv);
            if ((binx < 0) || (biny < 0))
                {
                continue;
                }
            unsigned int ibinx = (unsigned int)binx;
            unsigned int ibiny = (unsigned int)biny;
            // log bin to be incremented
            if ((ibinx < nbins_x) && (ibiny < nbins_y))
                {
                atomicAdd(&pmftArray[b_i(ibinx, ibiny)], 1);
                }
            }
        __syncthreads();
        }
    }

__global__ void sharedPCF(unsigned int *pmftArray,
                          unsigned int nbins_x,
                          unsigned int nbins_y,
                          trajectory::CudaBox box,
                          float max_x,
                          float max_y,
                          float dx,
                          float dy,
                          float cell_width,
                          const unsigned int *pl,
                          const unsigned int *cl,
                          const unsigned int *nl,
                          const unsigned int *it,
                          const int total_num_neighbors,
                          Index3D cell_indexer,
                          Index2D neighbor_indexer,
                          float3 *ref_points,
                          float *ref_orientations,
                          unsigned int n_ref,
                          float3 *points,
                          float *orientations,
                          unsigned int n_p,
                          unsigned int n_c)
    {
    // get global index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // get the cell index
    int cell_idx = blockIdx.x;
    // get the local index
    int local_idx = threadIdx.x;
    // precompute inverses for faster computation
    float dx_inv = 1.0f / dx;
    float dy_inv = 1.0f / dy;
    // compute indexers
    Index2D b_i = Index2D(nbins_x, nbins_y);
    Index2D c_i = Index2D(2, n_c);
    __shared__ float3 ref_pos[1024];
    __shared__ float2 ref_theta[1024];
    __shared__ float3 pos[1024];
    // get the number of particles in the ref cell
    unsigned int num_ref_particles = it[2*cell_idx+1] - it[2*cell_idx+0];
    // load into shared mem
    // for now it must be kept lower; facilitated by increasing number of cells
    if (local_idx < num_ref_particles)
        {
        unsigned int start_idx = it[2*cell_idx+0];
        ref_pos[local_idx] = ref_points[cl[c_i(0,start_idx + local_idx)]];
        ref_theta[local_idx].x = cosf(-ref_orientations[cl[c_i(0,start_idx + local_idx)]]);
        ref_theta[local_idx].y = sinf(-ref_orientations[cl[c_i(0,start_idx + local_idx)]]);
        }
    __syncthreads();
    // for each cell neighbor
    for (unsigned int i = 0; i < total_num_neighbors; i++)
        {
        // get the cell to be checked
        unsigned int neigh_idx = nl[neighbor_indexer(i, cell_idx)];

        // load the cell into shared mem
        unsigned int num_particles = it[2*neigh_idx+1] - it[2*neigh_idx+0];
        // create the indexer
        Index2D thread_indexer = Index2D(num_particles, num_ref_particles);
        // load particles into shared mem
        if (local_idx < num_particles)
            {
            unsigned int start_idx = it[2*neigh_idx+0];
            pos[local_idx] = points[cl[c_i(0,start_idx + local_idx)]];
            }
        __syncthreads();
        // now, populate the histogram
        // create the multipass indexer
        unsigned int n_pairs = num_ref_particles*num_ref_particles;
        unsigned int n_pass_threads = floorf(thread_indexer.getNumElements()/blockDim.x) + 1;
        Index2D multipass_indexer = Index2D(blockDim.x, n_pass_threads);

        for (unsigned int n_pass = 0; n_pass < n_pass_threads; n_pass++)
            {
            unsigned int my_index = multipass_indexer(local_idx, n_pass);
            // check to make sure the pair exists
            if (!(my_index < n_pairs))
                {
                break;
                }
            // get the pair index
            uint2 pair_idx = thread_indexer(my_index);
            float3 delta = box.wrap(make_float3(pos[pair_idx.x].x - ref_pos[pair_idx.y].x,
                                                pos[pair_idx.x].y - ref_pos[pair_idx.y].y,
                                                pos[pair_idx.x].z - ref_pos[pair_idx.y].z));
            float rsq = (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
            if (rsq < 1e-6)
                {
                continue;
                }
            // rotate the interparticle vector
            float x = delta.x*ref_theta[pair_idx.y].x - delta.y*ref_theta[pair_idx.y].y + max_x;
            float y = delta.x*ref_theta[pair_idx.y].y + delta.y*ref_theta[pair_idx.y].x + max_y;
            // find the bin to increment
            float binx = floorf(x * dx_inv);
            float biny = floorf(y * dy_inv);
            if ((binx < 0) || (biny < 0))
                {
                continue;
                }
            unsigned int ibinx = (unsigned int)binx;
            unsigned int ibiny = (unsigned int)biny;
            if ((ibinx < nbins_x) && (ibiny < nbins_y))
                {
                // atomicAdd(&pmftArray[b_i(ibinx, ibiny)], 1);
                pmftArray[b_i(ibinx, ibiny)]++;
                }
            }
        __syncthreads();
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
                    const Index3D& cell_indexer,
                    const Index2D& neighbor_indexer,
                    float3 *ref_points,
                    float *ref_orientations,
                    unsigned int n_ref,
                    float3 *points,
                    float *orientations,
                    unsigned int n_p,
                    unsigned int n_c)
    {
    // printf("avg occupancy: %f\n", float(n_p)/float(n_c));
    // define grid and block size
    int numThreadsPerBlock = 64;
    int numBlocks = (n_ref / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    computePCF<<< dimGrid, dimBlock >>>(pmftArray,
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
                                        cell_indexer,
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

void cudaCellPCF(unsigned int *pmftArray,
                   unsigned int nbins_x,
                   unsigned int nbins_y,
                   trajectory::CudaBox &box,
                   float max_x,
                   float max_y,
                   float dx,
                   float dy,
                   float cell_width,
                   const unsigned int *pl,
                   const unsigned int *cl,
                   const unsigned int *nl,
                   const unsigned int *it,
                   const int total_num_neighbors,
                   const Index3D& cell_indexer,
                   const Index2D& neighbor_indexer,
                   float3 *ref_points,
                   float *ref_orientations,
                   unsigned int n_ref,
                   float3 *points,
                   float *orientations,
                   unsigned int n_p,
                   unsigned int n_c)
    {

    int numThreadsPerBlock = 64;
    // we will be just using "cells" of the pmft for this exercise
    int numBlocks = n_c;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    cellPCF<<< dimGrid, dimBlock >>>(pmftArray,
                                       nbins_x,
                                       nbins_y,
                                       box,
                                       max_x,
                                       max_y,
                                       dx,
                                       dy,
                                       cell_width,
                                       pl,
                                       cl,
                                       nl,
                                       it,
                                       total_num_neighbors,
                                       cell_indexer,
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

void cudaSharedPCF(unsigned int *pmftArray,
                   unsigned int nbins_x,
                   unsigned int nbins_y,
                   trajectory::CudaBox &box,
                   float max_x,
                   float max_y,
                   float dx,
                   float dy,
                   float cell_width,
                   const unsigned int *pl,
                   const unsigned int *cl,
                   const unsigned int *nl,
                   const unsigned int *it,
                   const int total_num_neighbors,
                   const Index3D& cell_indexer,
                   const Index2D& neighbor_indexer,
                   float3 *ref_points,
                   float *ref_orientations,
                   unsigned int n_ref,
                   float3 *points,
                   float *orientations,
                   unsigned int n_p,
                   unsigned int n_c)
    {

    int numThreadsPerBlock = 64;
    // we will be just using "cells" of the pmft for this exercise
    int numBlocks = n_c;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    sharedPCF<<< dimGrid, dimBlock >>>(pmftArray,
                                       nbins_x,
                                       nbins_y,
                                       box,
                                       max_x,
                                       max_y,
                                       dx,
                                       dy,
                                       cell_width,
                                       pl,
                                       cl,
                                       nl,
                                       it,
                                       total_num_neighbors,
                                       cell_indexer,
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
