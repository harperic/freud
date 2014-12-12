#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <iostream>
#include <iomanip>
#include <ostream>
#include "HOOMDMath.h"
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
    // should this be an if > return?
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

__global__ void fillUINT2(uint2 *o_array,
                          thrust::device_ptr<unsigned int> x_array,
                          thrust::device_ptr<unsigned int> y_array,
                          unsigned int arr_size)
    {
    // determine particle being calculated
    // should this be an if > return?
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < arr_size)
        {
        o_array[idx].x = x_array[idx];
        o_array[idx].y = y_array[idx];
        }
    }

void cudaComputeCellList(unsigned int *p_array,
                         uint2 *c_array,
                         uint2 *it_array,
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
                                             cc_ptr,
                                             cp_ptr,
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
    thrust::device_ptr<unsigned int> cb_ptr = c_begin.data();
    thrust::device_ptr<unsigned int> ce_ptr = c_end.data();
    // sort
    thrust::stable_sort_by_key(cc_array.begin(),
                               cc_array.end(),
                               cp_array.begin());
    // now find the beginning and end of the cells
    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::lower_bound(cc_array.begin(),
                        cc_array.end(),
                        search_begin,
                        search_begin + nc,
                        c_begin.begin());
    thrust::upper_bound(cc_array.begin(),
                        cc_array.end(),
                        search_begin,
                        search_begin + nc,
                        c_end.begin());
    // copy back
    fillUINT2<<< dimGrid, dimBlock >>>(c_array,
                                       cp_ptr,
                                       cc_ptr,
                                       np);
    // define grid and block size
    numThreadsPerBlock = 16;
    numBlocks = (nc / numThreadsPerBlock) + 1;
    dimGrid = numBlocks;
    dimBlock = numThreadsPerBlock;
    fillUINT2<<< dimGrid, dimBlock >>>(it_array,
                                       cb_ptr,
                                       ce_ptr,
                                       nc);
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
    checkCUDAError("freeArray");
    }

void freeArray(float3 *array)
    {
    cudaFree(array);
    checkCUDAError("freeArray");
    }


void freeArray(uint2 *array)
    {
    cudaFree(array);
    checkCUDAError("freeArray");
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
