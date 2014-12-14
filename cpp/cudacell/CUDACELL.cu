#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <iomanip>
#include <ostream>
#include "HOOMDMath.h"
#include "Index1D.h"
#include "CUDACELL.cuh"

#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

using namespace std;

namespace freud { namespace cudacell {

__global__ void computeCellList(unsigned int *p_array,
                                thrust::device_ptr<unsigned int> cp_array,
                                thrust::device_ptr<unsigned int> cc_array,
                                unsigned int np,
                                unsigned int nc,
                                trajectory::CudaBox box,
                                Index3D cell_idx,
                                const float3 *points)
    {
    // determine particle being calculated
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < np)
        {
        // get the point
        float3 point = points[idx];
        // determine cell for idx
        float3 alpha = box.makeFraction(point);
        uint3 c;
        c.x = floorf(alpha.x * float(cell_idx.getW()));
        c.x %= cell_idx.getW();
        c.y = floorf(alpha.y * float(cell_idx.getH()));
        c.y %= cell_idx.getH();
        c.z = floorf(alpha.z * float(cell_idx.getD()));
        c.z %= cell_idx.getD();
        unsigned int c_idx = cell_idx((int)c.x, (int)c.y, (int)c.z);
        p_array[idx] = c_idx;
        cp_array[idx] = (unsigned int)idx;
        cc_array[idx] = c_idx;
        }
    }

__global__ void fillUINT2(unsigned int *o_array,
                          thrust::device_ptr<unsigned int> x_array,
                          thrust::device_ptr<unsigned int> y_array,
                          unsigned int arr_size)
    {
    // create the indexer for the cell list
    Index2D myIndexer(2, arr_size);
    // determine particle being calculated
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < arr_size)
        {
        o_array[myIndexer(0, idx)] = x_array[idx];
        o_array[myIndexer(1, idx)] = y_array[idx];
        }
    }

__global__ void computeCellNeighbors(thrust::device_ptr<unsigned int> neighbor_list,
                                     uint3 celldim,
                                     uint3 num_neighbors,
                                     unsigned int nc,
                                     unsigned int total_num_neighbors,
                                     Index3D cell_idx,
                                     Index2D thread_indexer,
                                     Index3D neighbor_indexer)
    {
    // determine raw idx being calculated
    int raw_idx = blockIdx.x*blockDim.x + threadIdx.x;
    // get the neighbor, cell being calc'd
    int neigh_idx1D = thread_indexer((unsigned int)raw_idx).x;
    int cell_idx1D = thread_indexer((unsigned int)raw_idx).y;
    uint3 cell_idx3D = cell_idx((unsigned int)cell_idx1D);
    uint3 neigh_idx3D = neighbor_indexer((unsigned int)neigh_idx1D);
    int start_i, start_j, start_k;
    // int end_i, end_j, end_k;
    if (cell_idx1D < nc)
        {
        if (celldim.x < 3)
            {
            start_i = (int)cell_idx3D.x;
            }
        else
            {
            start_i = (int)cell_idx3D.x - num_neighbors.x;
            }
        if (celldim.y < 3)
            {
            start_j = (int)cell_idx3D.y;
            }
        else
            {
            start_j = (int)cell_idx3D.y - num_neighbors.y;
            }
        if (celldim.z < 3)
            {
            start_k = (int)cell_idx3D.z;
            }
        else
            {
            start_k = (int)cell_idx3D.z - num_neighbors.z;
            }

        // if (celldim.x < 2)
        //     {
        //     end_i = (int)cell_idx3D.x;
        //     }
        // else
        //     {
        //     end_i = (int)cell_idx3D.x + num_neighbors.x;
        //     }
        // if (celldim.y < 2)
        //     {
        //     end_j = (int)cell_idx3D.y;
        //     }
        // else
        //     {
        //     end_j = (int)cell_idx3D.y + num_neighbors.y;
        //     }
        // if (celldim.z < 2)
        //     {
        //     end_k = (int)cell_idx3D.z;
        //     }
        // else
        //     {
        //     end_k = (int)cell_idx3D.z + num_neighbors.z;
        //     }
        // wrap back into the box
        int wrap_i = (cell_idx.getW()+start_i+neigh_idx3D.x) % cell_idx.getW();
        int wrap_j = (cell_idx.getH()+start_j+neigh_idx3D.y) % cell_idx.getH();
        int wrap_k = (cell_idx.getD()+start_k+neigh_idx3D.z) % cell_idx.getD();
        neighbor_list[raw_idx] = cell_idx(wrap_i, wrap_j, wrap_k);
        }
    }

__global__ void fillKeyArray(thrust::device_ptr<unsigned int> key_arr,
                             uint3 num_neighbors,
                             unsigned int nc,
                             Index2D thread_indexer)
    {
    // determine particle being calculated
    int raw_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int cell_idx1D = thread_indexer((unsigned int)raw_idx).y;
    if (cell_idx1D < nc)
        {
        key_arr[raw_idx] = (unsigned int)cell_idx1D;
        }
    }

void cudaComputeCellList(unsigned int *p_array,
                         unsigned int *c_array,
                         unsigned int *it_array,
                         unsigned int np,
                         unsigned int nc,
                         trajectory::CudaBox &box,
                         Index3D& cell_idx,
                         const float3 *points)
    {
    // create thrust vectors to handle the two parts of the c_array
    thrust::device_vector<unsigned int> cc_array(np);
    thrust::device_vector<unsigned int> cp_array(np);
    thrust::device_ptr<unsigned int> cc_ptr = cc_array.data();
    thrust::device_ptr<unsigned int> cp_ptr = cp_array.data();
    // define grid and block size
    int numThreadsPerBlock = 32;
    int numBlocks = (np / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

    computeCellList<<< dimGrid, dimBlock >>>(p_array,
                                             cp_ptr,
                                             cc_ptr,
                                             np,
                                             nc,
                                             box,
                                             cell_idx,
                                             points);
    cudaDeviceSynchronize();
    checkCUDAError("kernel execution");
    // now, perform a sort on the cell list
    // copy to thrust vector
    thrust::device_vector<unsigned int> c_begin(nc);
    thrust::device_vector<unsigned int> c_end(nc);
    // sort
    thrust::stable_sort_by_key(cc_array.begin(),
                               cc_array.end(),
                               cp_array.begin());
    cudaDeviceSynchronize();
    checkCUDAError("thrust sort");
    // now find the beginning and end of the cells
    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::lower_bound(cc_array.begin(),
                        cc_array.end(),
                        search_begin,
                        search_begin + nc,
                        c_begin.begin());
    cudaDeviceSynchronize();
    checkCUDAError("find lower_bound");
    thrust::upper_bound(cc_array.begin(),
                        cc_array.end(),
                        search_begin,
                        search_begin + nc,
                        c_end.begin());
    cudaDeviceSynchronize();
    checkCUDAError("find upper_bound");
    thrust::device_ptr<unsigned int> cb_ptr = c_begin.data();
    thrust::device_ptr<unsigned int> ce_ptr = c_end.data();
    // copy back
    fillUINT2<<< dimGrid, dimBlock >>>(c_array,
                                       cp_ptr,
                                       cc_ptr,
                                       np);
    cudaDeviceSynchronize();
    checkCUDAError("copy back");
    // redefine grid and block size
    // because these are smaller
    numThreadsPerBlock = 16;
    numBlocks = (nc / numThreadsPerBlock) + 1;
    dimGrid = numBlocks;
    dimBlock = numThreadsPerBlock;
    fillUINT2<<< dimGrid, dimBlock >>>(it_array,
                                       cb_ptr,
                                       ce_ptr,
                                       nc);
    cudaDeviceSynchronize();
    checkCUDAError("copy back");
    }

void cudaComputeCellNeighbors(unsigned int *cell_neighbors,
                              uint3 celldim,
                              uint3 num_neighbors,
                              unsigned int nc,
                              unsigned int total_num_neighbors,
                              Index3D& cell_idx,
                              Index2D& thread_indexer)
    {
    // create a neighbor indexer
    Index3D neighbor_indexer(2*(int)num_neighbors.x+1,
                             2*(int)num_neighbors.y+1,
                             2*(int)num_neighbors.z+1);
    // create thrust vectors to handle the calc and sorting of the cell list
    thrust::device_vector<unsigned int> t_cell_neighbors(cell_neighbors, cell_neighbors + nc*total_num_neighbors);
    thrust::device_ptr<unsigned int> t_cell_ptr = t_cell_neighbors.data();
    // define grid and block size
    int numThreadsPerBlock = 32;
    int numBlocks = (nc*total_num_neighbors / numThreadsPerBlock) + 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    computeCellNeighbors<<< dimGrid, dimBlock>>>(t_cell_ptr,
                                                 celldim,
                                                 num_neighbors,
                                                 nc,
                                                 total_num_neighbors,
                                                 cell_idx,
                                                 thread_indexer,
                                                 neighbor_indexer);
    cudaDeviceSynchronize();
    checkCUDAError("kernel execution");
    // now perform a segmented sort
    // this will require two keys
    // first generate the key array
    thrust::device_vector<unsigned int> t_key_arr(nc*total_num_neighbors);
    thrust::device_ptr<unsigned int> t_key_ptr = t_key_arr.data();
    fillKeyArray<<< dimGrid, dimBlock>>>(t_key_ptr,
                                         num_neighbors,
                                         nc,
                                         thread_indexer);
    cudaDeviceSynchronize();
    checkCUDAError("fill key array");
    thrust::stable_sort_by_key(t_cell_neighbors.begin(),
                               t_cell_neighbors.end(),
                               t_key_arr.begin());
    cudaDeviceSynchronize();
    checkCUDAError("first key sort");
    thrust::stable_sort_by_key(t_key_arr.begin(),
                               t_key_arr.end(),
                               t_cell_neighbors.begin());
    cudaDeviceSynchronize();
    checkCUDAError("second key sort");
    thrust::copy(t_cell_neighbors.begin(), t_cell_neighbors.end(), cell_neighbors);
    cudaDeviceSynchronize();
    checkCUDAError("thrust copy");
    }

void createArray(unsigned int **array, size_t memSize)
    {
    cudaMallocManaged(array, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("createArray");
    }

void createArray(float3 **array, size_t memSize)
    {
    cudaMallocManaged(array, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("createArray");
    }

void createArray(uint2 **array, size_t memSize)
    {
    cudaMallocManaged(array, memSize);
    cudaDeviceSynchronize();
    checkCUDAError("createArray");
    }

void freeArray(unsigned int *array)
    {
    cudaFree(array);
    cudaDeviceSynchronize();
    checkCUDAError("free unsigned int Array");
    }

void freeArray(float3 *array)
    {
    cudaFree(array);
    cudaDeviceSynchronize();
    checkCUDAError("free float 3 Array");
    }


void freeArray(uint2 *array)
    {
    cudaFree(array);
    cudaDeviceSynchronize();
    checkCUDAError("free uint2 Array");
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
