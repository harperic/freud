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
    for (int check_cell = 0; check_cell < n_c; check_cell++)
        {
        float2 cell_min = make_float2(cell_indexer(check_cell).x*cell_width, cell_indexer(check_cell).y*cell_width);
        float2 cell_max = make_float2((cell_indexer(check_cell).x+1)*cell_width, (cell_indexer(check_cell).y+1)*cell_width);
        // determine which part of the histogram to load
        // start with the x; because there are only 4 comparison, we can do this super simply
        float2 min_f2 = make_float2(cell_min.x - ref_cell_min.x,cell_max.x - ref_cell_min.x);
        float2 max_f2 = make_float2(cell_min.x - ref_cell_max.x,cell_max.x - ref_cell_max.x);
        float lmin_x = min_f2.x;
        if (lmin_x > min_f2.y)
            {
            lmin_x = min_f2.y;
            }
        if (lmin_x > max_f2.x)
            {
            lmin_x = max_f2.x;
            }
        if (lmin_x > max_f2.y)
            {
            lmin_x = max_f2.y;
            }
        float lmax_x = min_f2.x;
        if (lmax_x < min_f2.y)
            {
            lmax_x = min_f2.y;
            }
        if (lmax_x < max_f2.x)
            {
            lmax_x = max_f2.x;
            }
        if (lmax_x < max_f2.y)
            {
            lmax_x = max_f2.y;
            }
        // now do y; because there are only 4 comparison, we can do this super simply
        min_f2 = make_float2(cell_min.y - ref_cell_min.y,cell_max.y - ref_cell_min.y);
        max_f2 = make_float2(cell_min.y - ref_cell_max.y,cell_max.y - ref_cell_max.y);
        float lmin_y = min_f2.x;
        if (lmin_y > min_f2.y)
            {
            lmin_y = min_f2.y;
            }
        if (lmin_y > max_f2.x)
            {
            lmin_y = max_f2.x;
            }
        if (lmin_y > max_f2.y)
            {
            lmin_y = max_f2.y;
            }
        float lmax_y = min_f2.x;
        if (lmax_y < min_f2.y)
            {
            lmax_y = min_f2.y;
            }
        if (lmax_y < max_f2.x)
            {
            lmax_y = max_f2.x;
            }
        if (lmax_y < max_f2.y)
            {
            lmax_y = max_f2.y;
            }
        float x_offset = (lmax_x - lmin_x) / 2.0;
        float y_offset = (lmax_y - lmin_y) / 2.0;
        // determine the number of bins in the smaller histogram
        unsigned int lbins_x = (unsigned int)(2*floorf((lmax_x - lmin_x) * dx_inv));
        unsigned int lbins_y = (unsigned int)(2*floorf((lmax_y - lmin_y) * dy_inv));
        Index2D l_i = Index2D(lbins_x, lbins_y);
        // printf("number of elements in the cell list: %d\n", (int)l_i.getNumElements());
        // now we have the min/max delta coordinates for the histogram
        // load the cell into shared mem
        unsigned int num_particles = it[2*check_cell+1] - it[2*check_cell+0];
        // create the indexer
        Index2D thread_indexer = Index2D(num_particles, num_ref_particles);
        if (local_idx < num_particles)
            {
            pos[local_idx] = points[it[2*check_cell+0] + local_idx];
            }
        __syncthreads();
        // determine number of times for a thread to go through the list
        // for now, it's hardcoded
        unsigned int multiple = (num_particles*num_ref_particles/64) + 1;
        for (unsigned int times = 0; times < multiple; times++)
            {
            if ((times*64 + local_idx) > num_particles*num_ref_particles)
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
            float x = delta.x*ref_theta[p_idx.y].x - delta.y*ref_theta[p_idx.y].y + x_offset;
            float y = delta.x*ref_theta[p_idx.y].y + delta.y*ref_theta[p_idx.y].x + y_offset;
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
        float binx = floorf((lmin_x + max_x) * dx_inv);
        float biny = floorf((lmin_y + max_y) * dy_inv);
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
                uint2 local_bins = l_i((times*64 + local_idx));
                atomicAdd(&pmftArray[b_i(local_bins.x + global_binx_min, local_bins.y + global_biny_min)], histogram[(times*64 + local_idx)]);
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
