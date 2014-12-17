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
    // get the cell index
    int cell_idx = blockIdx.x;
    int local_idx = threadIdx.x;
    // precalc some values for faster computation within the loop
    float dx_inv = 1.0f / dx;
    float dy_inv = 1.0f / dy;
    Index2D b_i = Index2D(nbins_x, nbins_y);
    // for now, let's just assume that there are fewer particles than
    // there are threads...
    // these will probably need shrunk
    __shared__ float3 ref_pos[64];
    __shared__ float2 ref_theta[64];
    __shared__ float3 pos[64];
    // currently, this is not actually working, just a toy system
    __shared__ unsigned int histogram[1024];
    unsigned int multiple = (1024/64) + 1;
    for (unsigned int times = 0; times < multiple; times++)
        {
        // avoids writing to inaccessible memory
        if (!((times*64 + local_idx) < 1024))
            {
            break;
            }
        histogram[(times*64 + local_idx)] = 0;
        }
    // get the number of particles in the ref cell
    unsigned int num_ref_particles = it[2*cell_idx+1] - it[2*cell_idx+0];
    float2 ref_cell_min = make_float2(cell_indexer(cell_idx).x*cell_width, cell_indexer(cell_idx).y*cell_width);
    float2 ref_cell_max = make_float2((cell_indexer(cell_idx).x+1)*cell_width, (cell_indexer(cell_idx).y+1)*cell_width);
    // load into shared mem
    if (local_idx < num_ref_particles)
        {
        ref_pos[local_idx] = ref_points[it[2*cell_idx+0] + local_idx];
        ref_theta[local_idx].x = cosf(-ref_orientations[it[2*cell_idx+0] + local_idx]);
        ref_theta[local_idx].y = sinf(-ref_orientations[it[2*cell_idx+0] + local_idx]);
        }
    __syncthreads();
    // this needs to be against neighbors
    for (int i = 0; i < total_num_neighbors; i++)
        {
        // get the cell to be checked
        unsigned int neigh_idx = nl[neighbor_indexer((unsigned int)i, cell_idx)];
        // first, identify the geometry of the reference cell and check cell
        uint3 ref_cell_coords = cell_indexer(cell_idx);
        uint3 check_cell_coords = cell_indexer(neigh_idx);

        // find the min/max x, y values

        // min value in x, max value in y for the histogram
        float2 min_max_x;
        float2 min_max_y;

        // the cell being checked is to the left
        if (ref_cell_coords.x > check_cell_coords.x)
            {
            min_max_x.x = check_cell_coords.x*cell_width - (ref_cell_coords.x + 1)*cell_width;
            min_max_x.y = (check_cell_coords.x+1)*cell_width - ref_cell_coords.x*cell_width;
            }
        // the cell being check is to the right
        else if (ref_cell_coords.x < check_cell_coords.x)
            {
            min_max_x.x = (check_cell_coords.x+1)*cell_width - ref_cell_coords.x*cell_width;
            min_max_x.y = check_cell_coords.x*cell_width - (ref_cell_coords.x+1)*cell_width;
            }
        // the cells are the same
        else
            {
            min_max_x.x = -cell_width;
            min_max_x.y = cell_width;
            }

        // the cell being checked is to the left
        if (ref_cell_coords.y > check_cell_coords.y)
            {
            min_max_y.x = check_cell_coords.y*cell_width - (ref_cell_coords.y + 1)*cell_width;
            min_max_y.y = (check_cell_coords.y+1)*cell_width - ref_cell_coords.y*cell_width;
            }
        // the cell being check is to the right
        else if (ref_cell_coords.y < check_cell_coords.y)
            {
            min_max_y.x = (check_cell_coords.y+1)*cell_width - ref_cell_coords.y*cell_width;
            min_max_y.y = check_cell_coords.y*cell_width - (ref_cell_coords.y+1)*cell_width;
            }
        // the cells are the same
        else
            {
            min_max_y.x = -cell_width;
            min_max_y.y = cell_width;
            }

        // this should be just cell_width
        // float x_offset = (lmax_x - lmin_x) / 2.0;
        // float y_offset = (lmax_y - lmin_y) / 2.0;
        // determine the number of bins in the smaller histogram
        unsigned int lbins_x = (unsigned int)(2*floorf(cell_width * dx_inv));
        unsigned int lbins_y = (unsigned int)(2*floorf(cell_width * dy_inv));
        Index2D l_i = Index2D(lbins_x, lbins_y);
        // now we have the min/max delta coordinates for the histogram
        // load the cell into shared mem
        unsigned int num_particles = it[2*neigh_idx+1] - it[2*neigh_idx+0];
        // create the indexer
        Index2D thread_indexer = Index2D(num_particles, num_ref_particles);
        // load particles into shared mem
        if (local_idx < num_particles)
            {
            pos[local_idx] = points[it[2*neigh_idx+0] + local_idx];
            }
        __syncthreads();
        // determine number of times for a thread to go through the list
        // for now, it's hardcoded
        multiple = (num_particles*num_ref_particles/64) + 1;
        for (unsigned int times = 0; times < multiple; times++)
            {
            if (!((times*64 + local_idx) < num_particles*num_ref_particles))
                {
                break;
                }
            // get the ref/check particle index
            uint2 p_idx = thread_indexer(times*64 + local_idx);
            float3 delta = box.wrap(make_float3(pos[p_idx.x].x - ref_pos[p_idx.y].x,
                                                pos[p_idx.x].y - ref_pos[p_idx.y].y,
                                                pos[p_idx.x].z - ref_pos[p_idx.y].z));
            float rsq = (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
            if (rsq < 1e-6)
                {
                continue;
                }
            // rotate the interparticle vector
            float x = delta.x*ref_theta[p_idx.y].x - delta.y*ref_theta[p_idx.y].y + cell_width;
            float y = delta.x*ref_theta[p_idx.y].y + delta.y*ref_theta[p_idx.y].x + cell_width;
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
            if ((ibinx < lbins_x) && (ibiny < lbins_y))
                {
                if (l_i(ibinx, ibiny) < 1024)
                    {
                    atomicAdd(&histogram[l_i(ibinx, ibiny)], 1);
                    }
                }
            }
        __syncthreads();
        // copy back to array
        multiple = (lbins_x*lbins_y/64) + 1;
        // find the global bin start
        float binx = floorf((min_max_x.x + max_x) * dx_inv);
        float biny = floorf((min_max_y.x + max_y) * dy_inv);
        unsigned int global_binx_min = (unsigned int)binx;
        unsigned int global_biny_min = (unsigned int)biny;
        // unsigned int idx_min = b_i(global_binx_min, global_biny_min);
        for (unsigned int times = 0; times < multiple; times++)
            {
            // if ((times*64 + local_idx) < lbins_x*lbins_y)
            if ((times*64 + local_idx) < 1024)
                {
                // this isn't right...I need to take into account stride...lol
                // the x bin min is fine, the y will change? or something
                // wait what about
                // find local bins
                // this would appear that it is not writing out the cells to the correct positions
                uint2 local_bins = l_i((times*64 + local_idx));
                // atomicAdd(&pmftArray[b_i(local_bins.x + global_binx_min, local_bins.y + global_biny_min)], histogram[(times*64 + local_idx)]);
                if (((global_binx_min + local_bins.x) < nbins_x) && ((global_biny_min + local_bins.y) < nbins_y))
                    {
                    atomicAdd(&pmftArray[b_i(local_bins.x + global_binx_min, local_bins.y + global_biny_min)], 1);
                    }
                }
            }
        __syncthreads();
        }
    }

__global__ void sharedTest(unsigned int *pmftArray,
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
    __shared__ unsigned int histogram[4096];
    // consider increasing in size...64 is def to small
    // maybe 512?
    __shared__ float3 ref_pos[64];
    __shared__ float2 ref_theta[64];
    __shared__ float3 pos[64];
    // get the number of particles in the ref cell
    unsigned int num_ref_particles = it[2*cell_idx+1] - it[2*cell_idx+0];
    // load into shared mem
    // for now it must be kept lower; could cycle through multiple times if necessary
    if (local_idx < num_ref_particles)
        {
        ref_pos[local_idx] = ref_points[it[2*cell_idx+0] + local_idx];
        ref_theta[local_idx].x = cosf(-ref_orientations[it[2*cell_idx+0] + local_idx]);
        ref_theta[local_idx].y = sinf(-ref_orientations[it[2*cell_idx+0] + local_idx]);
        }
    // don't sync til now; histogram not needed to be zero'd yet
    __syncthreads();
    // for each cell neighbor
    for (int i = 0; i < total_num_neighbors; i++)
        {
        // get the cell to be checked
        unsigned int neigh_idx = nl[neighbor_indexer((unsigned int)i, cell_idx)];

        // load the cell into shared mem
        unsigned int num_particles = it[2*neigh_idx+1] - it[2*neigh_idx+0];
        // create the indexer
        Index2D thread_indexer = Index2D(num_particles, num_ref_particles);
        // load particles into shared mem
        if (local_idx < num_particles)
            {
            pos[local_idx] = points[it[2*neigh_idx+0] + local_idx];
            }
        __syncthreads();
        // now, populate the histogram
        // multipass
        // determine number of times for a thread to go through the list
        // determine how many pairs to handle at a time
        // ceil wasn't behaving properly...lol

        unsigned int n_pairs = num_ref_particles*num_ref_particles;
        unsigned int n_pass_blocks = floorf(n_pairs/4096) + 1;
        unsigned int n_pass_threads = floorf(4096/blockDim.x) + 1;
        Index3D multipass_indexer = Index3D(blockDim.x, n_pass_threads, n_pass_blocks);
        for (unsigned int n_times_blocks = 0; n_times_blocks < n_pass_blocks; n_times_blocks++)
            {
            // reset the histogram
            for (unsigned int n_times = 0; n_times < (floorf(4096/blockDim.x)+1); n_times++)
                {
                // avoids writing to inaccessible memory
                if (!((n_times*blockDim.x + local_idx) < 4096))
                    {
                    break;
                    }
                histogram[(n_times*blockDim.x + local_idx)] = 0;
                }
            for (unsigned int n_times_threads = 0; n_times_threads < n_pass_threads; n_times_threads++)
                {
                // get the index to handle
                unsigned int my_index = multipass_indexer(local_idx, n_times_threads, n_times_blocks);
                // check to make sure the pair exists
                if (!(my_index < num_particles*num_ref_particles))
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
                // log bin to be incremented
                if ((ibinx < nbins_x) && (ibiny < nbins_y))
                    {
                    histogram[blockDim.x*n_pass_threads + local_idx] = b_i(ibinx, ibiny);
                    atomicAdd(&pmftArray[b_i(ibinx, ibiny)], 1);
                    // if (local_idx == 0)
                    //     printf("%d\n", histogram[blockDim.x*n_pass_threads + local_idx]);
                    }
                }
            __syncthreads();
            // now we have a complete block of 4096 values to increment
            for (unsigned int n_times = 0; n_times < (floorf(4096/blockDim.x)+1); n_times++)
                {
                // avoids writing to inaccessible memory
                if (!((n_times*blockDim.x + local_idx) < 4096))
                    {
                    break;
                    }
                // atomicAdd(&pmftArray[histogram[n_times*blockDim.x + local_idx]], 1);
                // pmftArray[histogram[n_times*blockDim.x + local_idx]] += 1;
                }
            __syncthreads();
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
    int numThreadsPerBlock = 32;
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
    int numBlocks = (n_c / numThreadsPerBlock) + 1;

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

void cudaSharedTest(unsigned int *pmftArray,
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

    int numThreadsPerBlock = 256;
    // we will be just using "cells" of the pmft for this exercise
    // printf("%d grid points divided into %d cells\n", (nbins_x*nbins_y), (nbins_x/32 + 1)*(nbins_y/32 + 1));
    // printf("that's %d by %d = %f\n", nbins_x, (nbins_x/32 + 1), (float)nbins_x/(float)(nbins_x/32 + 1));
    int numBlocks = n_c;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    checkCUDAError("prior to kernel execution");
    sharedTest<<< dimGrid, dimBlock >>>(pmftArray,
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
