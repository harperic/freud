#include "CUDACELL.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include "HOOMDMath.h"
#include "VectorMath.h"

using namespace std;
using namespace boost::python;

/*! \internal
    \file CUDACELL.cc
    \brief Routines for computing 2D anisotropic potential of mean force
*/

namespace freud { namespace cudacell {

CudaCell::CudaCell()
    : d_box(trajectory::CudaBox()), m_np(0), m_cell_width(0)
    {
    m_celldim.x = 1;
    m_celldim.y = 1;
    m_celldim.z = 1;
    // initialize arrays on gpu
    createArray(&d_cidx_array, sizeof(unsigned int)*2);
    createArray(&d_it_array, sizeof(unsigned int)*2);
    createArray(&d_pidx_array, sizeof(unsigned int));
    createArray(&d_cell_neighbors, sizeof(unsigned int));
    createArray(&d_point_array, sizeof(float3));
    }

CudaCell::CudaCell(const trajectory::CudaBox& box, float cell_width, float r_cut)
    : d_box(box), m_cell_width(cell_width), m_r_cut(r_cut)
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
    createArray(&d_cidx_array, sizeof(unsigned int)*2);
    createArray(&d_it_array, sizeof(unsigned int)*2);
    createArray(&d_pidx_array, sizeof(unsigned int));
    createArray(&d_cell_neighbors, sizeof(unsigned int));
    createArray(&d_point_array, sizeof(float3));
    m_cell_index = Index3D(m_celldim.x, m_celldim.y, m_celldim.z);
    m_num_neighbors = computeNumNeighbors(d_box, m_r_cut, m_cell_width);
    computeCellNeighbors();
    }

CudaCell::~CudaCell()
    {
    freeArray(d_cidx_array);
    freeArray(d_it_array);
    freeArray(d_pidx_array);
    freeArray(d_cell_neighbors);
    freeArray(d_point_array);
    }

void CudaCell::setCellWidth(float cell_width)
    {
    if (cell_width != m_cell_width)
        {
        float3 L = d_box.getNearestPlaneDistance();
        uint3 celldim  = computeDimensions(d_box, cell_width);
        uint3 num_neighbors = computeNumNeighbors(d_box, m_r_cut, cell_width);
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
        // set up bool operators to figure out whether the cells dims changed, the num neighbors changes, or both
        bool celldim_changed = !((celldim.x == m_celldim.x) && (celldim.y == m_celldim.y) && (celldim.z == m_celldim.z));
        bool num_neighbors_changed = !((num_neighbors.x == m_num_neighbors.x) && (num_neighbors.y == m_num_neighbors.y) && (num_neighbors.z == m_num_neighbors.z));
        // check if the dims of either changed
        if (celldim_changed || num_neighbors_changed)
            {
            if (celldim_changed)
                {
                // recalculate the index
                m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
                m_celldim  = celldim;
                }
            if (num_neighbors_changed)
                {
                // set the number of neighbors
                m_num_neighbors = num_neighbors;
                }
            computeCellNeighbors();
            }
        m_cell_width = cell_width;
        }
    }

void CudaCell::setRCut(float r_cut)
    {
    // m_r_cut is replacing the cpu cell_width, kind of
    // they work together
    // so that what the user used to set as the cell_width is now the rcut
    // cell width is still set-able to make for efficient gpu usage
    if (r_cut != m_r_cut)
        {
        // still need to check to make sure that requested r_cut isn't too big
        float3 L = d_box.getNearestPlaneDistance();
        uint3 num_neighbors = computeNumNeighbors(d_box, r_cut, m_cell_width);
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
            m_num_neighbors = num_neighbors;
            computeCellNeighbors();
            }
        m_r_cut = r_cut;
        }
    }

void CudaCell::updateBox(const trajectory::CudaBox& box)
    {
    // check if the cell width is too wide for the box
    float3 L = box.getNearestPlaneDistance();
    uint3 celldim  = computeDimensions(box, m_cell_width);
    uint3 num_neighbors = computeNumNeighbors(box, m_r_cut, m_cell_width);
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
    // set up bool operators to figure out whether the cells dims changed, the num neighbors changes, or both
    bool celldim_changed = !((celldim.x == m_celldim.x) && (celldim.y == m_celldim.y) && (celldim.z == m_celldim.z));
    bool num_neighbors_changed = !((num_neighbors.x == m_num_neighbors.x) && (num_neighbors.y == m_num_neighbors.y) && (num_neighbors.z == m_num_neighbors.z));
    // check if the dims of either changed
    if (celldim_changed || num_neighbors_changed)
        {
        if (celldim_changed)
            {
            // recalculate the index
            m_cell_index = Index3D(celldim.x, celldim.y, celldim.z);
            m_celldim  = celldim;
            }
        if (num_neighbors_changed)
            {
            // set the number of neighbors
            m_num_neighbors = num_neighbors;
            }
        computeCellNeighbors();
        }
    }

unsigned int CudaCell::roundDown(unsigned int v, unsigned int m)
    {
    // use integer floor division
    unsigned int d = v/m;
    return d*m;
    }

unsigned int CudaCell::roundUp(unsigned int v, unsigned int m)
    {
    // use integer floor division
    // was obtained off stack exchange
    unsigned int d = v/m + (v % m != 0);
    return d*m;
    }

const uint3 CudaCell::computeDimensions(const trajectory::CudaBox& box, float cell_width) const
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

const uint3 CudaCell::computeNumNeighbors(const trajectory::CudaBox& box, float r_cut, float cell_width) const
    {
    uint3 dim;

    // determine the number of cells required
    dim.x = dim.y = dim.z = (unsigned int)(r_cut/cell_width);

    if (box.is2D())
        {
        dim.z = 0;
        }

    return dim;
    }

void CudaCell::computeCellList(trajectory::CudaBox& box,
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

        createArray(&d_cidx_array, sizeof(unsigned int)*np*2);
        createArray(&d_it_array, sizeof(unsigned int)*nc*2);
        createArray(&d_pidx_array, sizeof(unsigned int)*np);
        createArray(&d_point_array, sizeof(float3)*np);
        }
    memcpy((void*)d_point_array, (void*)points, sizeof(float3)*np);
    m_np = np;
    m_nc = nc;
    // points needs put onto the device
    cudaComputeCellList(d_pidx_array, d_cidx_array, d_it_array, m_np, m_nc, d_box, m_cell_index, d_point_array);
    }

void CudaCell::computeCellNeighbors()
    {
    unsigned int nc = getNumCells();
    int total_num_neighbors = (2*m_num_neighbors.x + 1)*(2*m_num_neighbors.y + 1)*(2*m_num_neighbors.z + 1);
    Index2D expanded_thread_indexer(total_num_neighbors, nc);
    int arr_size = (int)nc*total_num_neighbors;
    freeArray(d_cell_neighbors);
    createArray(&d_cell_neighbors, sizeof(unsigned int)*arr_size);
    cudaComputeCellNeighbors(d_cell_neighbors,
                             m_celldim,
                             m_num_neighbors,
                             nc,
                             total_num_neighbors,
                             m_cell_index,
                             expanded_thread_indexer);
    }

void CudaCell::computeCellListPy(trajectory::CudaBox& box,
                                 boost::python::numeric::array points)
    {
    // validate input type and rank
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    // get the raw data pointers and compute the cell list
    float3* points_raw = (float3*) num_util::data(points);

        // compute the cell list with the GIL released
        {
        util::ScopedGILRelease gil;
        computeCellList(box, points_raw, Np);
        }
    }

void export_CudaCell()
    {
    class_<CudaCell>("CudaCell", init<trajectory::CudaBox&, float, float>())
        .def("getBox", &CudaCell::getBox, return_internal_reference<>())
        .def("getCellIndexer", &CudaCell::getCellIndexer, return_internal_reference<>())
        .def("getNumCells", &CudaCell::getNumCells)
        .def("getCell", &CudaCell::getCellPy)
        .def("getCellList", &CudaCell::getCellListPy)
        .def("getCellNeighborList", &CudaCell::getCellNeighborListPy)
        .def("getCellNeighbors", &CudaCell::getCellNeighborsPy)
        .def("computeCellList", &CudaCell::computeCellListPy)
        ;
    }

}; }; // end namespace freud::cudacell
