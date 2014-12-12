#include "CUDACELL.h"
#include "ScopedGILRelease.h"

#include "HOOMDMath.h"

#ifdef NVCC
// #define HOSTDEVICE __host__ __device__ inline
#define HOSTDEVICE __host__ __device__
#endif

/*! \internal
    \file CUDACELL.cc
    \brief Routines for computing 2D anisotropic potential of mean force
*/

namespace freud { namespace cudacell {

HOSTDEVICE CudaCell::CudaCell()
    : d_box(trajectory::CudaBox()), m_np(0), m_cell_width(0)
    {
    m_celldim.x = 1;
    m_celldim.y = 1;
    m_celldim.z = 1;
    // initialize arrays on gpu
    createArray(&d_cidx_array, sizeof(uint2));
    createArray(&d_it_array, sizeof(uint2));
    createArray(&d_pidx_array, sizeof(unsigned int));
    createArray(&d_point_array, sizeof(float3));
    }

HOSTDEVICE CudaCell::CudaCell(const trajectory::CudaBox& box, float cell_width)
    : d_box(box), m_cell_width(cell_width)
    {
    // check if the cell width is too wide for the box
    m_celldim = computeDimensions(d_box, m_cell_width);
    // Check if box is too small!
    // will only check if the box is not null
    if (box != trajectory::CudaBox())
        {
        float3 L = d_box.getNearestPlaneDistance();
        bool too_wide =  m_cell_width > L.x/2.0 || m_cell_width > L.y/2.0;
        if (!d_box.is2D())
            {
            too_wide |=  m_cell_width > L.z/2.0;
            }
        // if (too_wide)
        //     {
        //     throw runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
        //     }
        //only 1 cell deep in 2D
        if (d_box.is2D())
            {
            m_celldim.z = 1;
            }
        }
    // initialize arrays on gpu
    createArray(&d_cidx_array, sizeof(uint2));
    createArray(&d_it_array, sizeof(uint2));
    createArray(&d_pidx_array, sizeof(unsigned int));
    createArray(&d_point_array, sizeof(float3));
    m_cell_index = Index3D(m_celldim.x, m_celldim.y, m_celldim.z);
    computeCellNeighbors();
    }

HOSTDEVICE CudaCell::~CudaCell()
    {
    freeArray(d_cidx_array);
    freeArray(d_it_array);
    freeArray(d_pidx_array);
    freeArray(d_point_array);
    }

HOSTDEVICE void CudaCell::setCellWidth(float cell_width)
    {
    if (cell_width != m_cell_width)
        {
        float3 L = d_box.getNearestPlaneDistance();
        uint3 celldim  = computeDimensions(d_box, cell_width);
        //Check if box is too small!
        bool too_wide =  cell_width > L.x/2.0 || cell_width > L.y/2.0;
        if (!d_box.is2D())
            {
            too_wide |=  cell_width > L.z/2.0;
            }
        // if (too_wide)
        //     {
        //     throw runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
        //     }
        //only 1 cell deep in 2D
        if (d_box.is2D())
            {
            celldim.z = 1;
            }
        // check if the dims changed
        if (!((celldim.x == m_celldim.x) && (celldim.y == m_celldim.y) && (celldim.z == m_celldim.z)))
            {
            m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
            // if (m_cell_index.getNumElements() < 1)
            //     {
            //     throw runtime_error("At least one cell must be present");
            //     }
            m_celldim  = celldim;
            computeCellNeighbors();
            }
        m_cell_width = cell_width;
        }
    }

HOSTDEVICE void CudaCell::setRCut(float r_cut)
    {
    // m_rcut is replacing the cpu cell_width, kind of
    // they work together
    // so that what the user used to set as the cell_width is now the rcut
    // cell width is still set-able to make for efficient gpu usage
    if (r_cut != m_r_cut)
        {
        // still need to check to make sure that requested r_cut isn't too big
        float3 L = d_box.getNearestPlaneDistance();
        // this needs to be changed to number of neighbors
        uint3 num_neighbors = computeNumNeighbors(d_box, r_cut);
        //Check if box is too small!
        bool too_wide =  r_cut > L.x/2.0 || r_cut > L.y/2.0;
        if (!d_box.is2D())
            {
            too_wide |=  r_cut > L.z/2.0;
            }
        //only 1 cell deep in 2D (no neighbors)
        if (d_box.is2D())
            {
            num_neighbors.z = 0;
            }
        // check if the dims changed
        if (!((num_neighbors.x == m_num_neighbors.x) && (num_neighbors.y == m_num_neighbors.y) && (num_neighbors.z == m_num_neighbors.z)))
            {
            // I don't think I need this
            // m_cell_index = Index3D(num_neighbors.x, num_neighbors.y, num_neighbors.z);
            // I wonder if this should be changed to hold the number of cells for a neighbor...
            m_num_neighbors  = num_neighbors;
            computeCellNeighbors();
            }
        m_r_cut = r_cut;
        }
    }

HOSTDEVICE void CudaCell::updateBox(const trajectory::CudaBox& box)
    {
    // check if the cell width is too wide for the box
    float3 L = box.getNearestPlaneDistance();
    uint3 celldim  = computeDimensions(box, m_cell_width);
    //Check if box is too small!
    bool too_wide =  m_cell_width > L.x/2.0 || m_cell_width > L.y/2.0;
    if (!box.is2D())
        {
        too_wide |=  m_cell_width > L.z/2.0;
        }
    // if (too_wide)
    //     {
    //     throw runtime_error("Cannot generate a cell list where cell_width is larger than half the box.");
    //     }
    //only 1 cell deep in 2D
    if (box.is2D())
        {
        celldim.z = 1;
        }
    // check if the box is changed
    d_box = box;
    if (!((celldim.x == m_celldim.x) && (celldim.y == m_celldim.y) && (celldim.z == m_celldim.z)))
        {
        m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
        // if (m_cell_index.getNumElements() < 1)
        //     {
        //     throw runtime_error("At least one cell must be present");
        //     }
        m_celldim  = celldim;
        computeCellNeighbors();
        }
    }

HOSTDEVICE unsigned int CudaCell::roundDown(unsigned int v, unsigned int m)
    {
    // use integer floor division
    unsigned int d = v/m;
    return d*m;
    }

HOSTDEVICE unsigned int CudaCell::roundUp(unsigned int v, unsigned int m)
    {
    // use integer floor division
    // was obtained off stack exchange
    unsigned int d = v/m + (v % m != 0);
    return d*m;
    }

HOSTDEVICE const uint3 CudaCell::computeDimensions(const trajectory::CudaBox& box, float cell_width) const
    {
    uint3 dim;

    //multiple is a holdover from hpmc...doesn't really need to be kept
    unsigned int multiple = 1;
    float3 L = box.getNearestPlaneDistance();
    dim.x = roundDown((unsigned int)((L.x) / (cell_width)), multiple);
    dim.y = roundDown((unsigned int)((L.y) / (cell_width)), multiple);

    if (box.is2D())
        {
        dim.z = 1;
        }
    else
        {
        dim.z = roundDown((unsigned int)((L.z) / (cell_width)), multiple);
        }

    // In extremely small boxes, the calculated dimensions could go to zero, but need at least one cell in each dimension
    //  for particles to be in a cell and to pass the checkCondition tests.
    // Note: Freud doesn't actually support these small boxes (as of this writing), but this function will return the correct dimensions
    //  required anyways.
    if (dim.x == 0)
        dim.x = 1;
    if (dim.y == 0)
        dim.y = 1;
    if (dim.z == 0)
        dim.z = 1;
    return dim;
    }

HOSTDEVICE const uint3 CudaCell::computeNumNeighbors(const trajectory::CudaBox& box, float r_cut) const
    {
    uint3 dim;

    //multiple is a holdover from hpmc...doesn't really need to be kept
    unsigned int multiple = 1;
    // determine the number of cells required
    // always need to round down
    dim.x = dim.y = dim.z = (unsigned int)(r_cut/m_cell_width);

    if (box.is2D())
        {
        dim.z = 0;
        }

    // In extremely small boxes, the calculated dimensions could go to zero, but need at least one cell in each dimension
    //  for particles to be in a cell and to pass the checkCondition tests.
    // Note: Freud doesn't actually support these small boxes (as of this writing), but this function will return the correct dimensions
    //  required anyways.
    // if (dim.x == 0)
    //     dim.x = 1;
    // if (dim.y == 0)
    //     dim.y = 1;
    // if (dim.z == 0)
    //     dim.z = 1;
    return dim;
    }

HOSTDEVICE void CudaCell::computeCellList(trajectory::CudaBox& box,
                                          const float3 *points,
                                          unsigned int np)
    {
    updateBox(box);
    // can't call from device
    // may not make device callable, if that's possible
    // if (np == 0)
    //     {
    //     throw runtime_error("Cannot generate a cell list of 0 particles");
    //     }

    // determine the number of cells and allocate memory
    unsigned int nc = getNumCells();
    assert(nc > 0);
    if ((m_np != np) || (m_nc != nc))
        {
        freeArray(d_cidx_array);
        freeArray(d_it_array);
        freeArray(d_pidx_array);
        freeArray(d_point_array);

        createArray(&d_cidx_array, sizeof(uint2)*np);
        createArray(&d_it_array, sizeof(uint2)*nc);
        createArray(&d_pidx_array, sizeof(unsigned int)*np);
        createArray(&d_point_array, sizeof(float3)*np);
        }
    memcpy((void*)d_point_array, (void*)points, sizeof(float3)*np);
    m_np = np;
    m_nc = nc;
    // points needs put onto the device
    cudaComputeCellList(d_pidx_array, d_cidx_array, d_it_array, m_np, m_nc, d_box, m_cell_index, d_point_array);
    }

HOSTDEVICE void CudaCell::computeCellNeighbors()
    {
    // clear the list
    // will be a call to cuda
    // m_cell_neighbors.clear();
    // m_cell_neighbors.resize(getNumCells());

    // will be a call to cuda
    // for each cell
    // for (unsigned int k = 0; k < m_cell_index.getD(); k++)
    //     for (unsigned int j = 0; j < m_cell_index.getH(); j++)
    //         for (unsigned int i = 0; i < m_cell_index.getW(); i++)
    //             {
    //             // clear the list
    //             unsigned int cur_cell = m_cell_index(i,j,k);
    //             m_cell_neighbors[cur_cell].clear();

    //             // loop over the neighbor cells
    //             int starti, startj, startk;
    //             int endi, endj, endk;
    //             if (m_celldim.x < 3)
    //                 {
    //                 starti = (int)i;
    //                 }
    //             else
    //                 {
    //                 starti = (int)i - 1;
    //                 }
    //             if (m_celldim.y < 3)
    //                 {
    //                 startj = (int)j;
    //                 }
    //             else
    //                 {
    //                 startj = (int)j - 1;
    //                 }
    //             if (m_celldim.z < 3)
    //                 {
    //                 startk = (int)k;
    //                 }
    //             else
    //                 {
    //                 startk = (int)k - 1;
    //                 }

    //             if (m_celldim.x < 2)
    //                 {
    //                 endi = (int)i;
    //                 }
    //             else
    //                 {
    //                 endi = (int)i + 1;
    //                 }
    //             if (m_celldim.y < 2)
    //                 {
    //                 endj = (int)j;
    //                 }
    //             else
    //                 {
    //                 endj = (int)j + 1;
    //                 }
    //             if (m_celldim.z < 2)
    //                 {
    //                 endk = (int)k;
    //                 }
    //             else
    //                 {
    //                 endk = (int)k + 1;
    //                 }
    //             if (d_box.is2D())
    //                 startk = endk = k;

    //             for (int neighk = startk; neighk <= endk; neighk++)
    //                 for (int neighj = startj; neighj <= endj; neighj++)
    //                     for (int neighi = starti; neighi <= endi; neighi++)
    //                         {
    //                         // wrap back into the box
    //                         int wrapi = (m_cell_index.getW()+neighi) % m_cell_index.getW();
    //                         int wrapj = (m_cell_index.getH()+neighj) % m_cell_index.getH();
    //                         int wrapk = (m_cell_index.getD()+neighk) % m_cell_index.getD();

    //                         unsigned int neigh_cell = m_cell_index(wrapi, wrapj, wrapk);
    //                         // add to the list
    //                         m_cell_neighbors[cur_cell].push_back(neigh_cell);
    //                         }

    //             // sort the list
    //             sort(m_cell_neighbors[cur_cell].begin(), m_cell_neighbors[cur_cell].end());
    //             }
    }

}; }; // end namespace freud::cudacell
