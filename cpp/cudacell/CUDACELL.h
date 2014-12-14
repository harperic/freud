#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"

#include "CUDACELL.cuh"
#include "num_util.h"
#include "cudabox.h"
#include "Index1D.h"

#ifndef _CUDACELL_H__
#define _CUDACELL_H__

/*! \internal
    \file CUDACELL.h
    \brief Routines for computing anisotropic potential of mean force in 2D
*/

namespace freud { namespace cudacell {

//! Computes the PCF for a given set of points
/*! A given set of reference points is given around which the PCF is computed and averaged in a sea of data points.
    Computing the PCF results in a pcf array listing the value of the PCF at each given x, y, listed in the x, y arrays.

    The values of x, y to compute the pcf at are controlled by the xmax, ymax and dx, dy parameters to the constructor.
    xmax, ymax determines the minimum/maximum x, y at which to compute the pcf and dx, dy is the step size for each bin.

    <b>2D:</b><br>
    This PCF only works for 2D boxes. As with everything else in freud, 2D points must be passed in as
    3 component vectors x,y,0. Failing to set 0 in the third component should not matter as the code forces z=0.
    However, this could still lead to undefined behavior and should be avoided anyway.
*/
class CudaCell
    {
    public:
        //! Constructor
        CudaCell(const trajectory::CudaBox& box, float cell_width, float r_cut);

        //! Null Constructor for triclinic behavior
        CudaCell();

        //! Destructor
        ~CudaCell();

        //! Update cell_width
        void setCellWidth(float cell_width);

        //! Update cell_width
        void setRCut(float r_cut);

        //! Update box used in linkCell
        void updateBox(const trajectory::CudaBox& box);

        //! Compute CudaCell dimensions
        const uint3 computeDimensions(const trajectory::CudaBox& box, float cell_width) const;

        //! Compute number of CudaCell neighbors
        const uint3 computeNumNeighbors(const trajectory::CudaBox& box, float r_cut, float cell_width) const;

        //! Get a reference to the cell list used on the gpu array
        const unsigned int* getPointList() const
            {
            return d_pidx_array;
            }

        //! Get a reference to the cell list used on the gpu array
        boost::python::numeric::array getPointListPy()
            {
            boost::shared_array<unsigned int> return_array = boost::shared_array<unsigned int>(new unsigned int[m_np]);
            memcpy((void*)return_array.get(), d_cidx_array, sizeof(unsigned int)*m_np);
            unsigned int *arr = return_array.get();
            return num_util::makeNum(arr, m_np);
            }

        //! Get a reference to the cell list used on the gpu array
        const unsigned int* getCellList() const
            {
            return d_cidx_array;
            }

        //! Get a reference to the cell list used on the gpu array
        boost::python::numeric::array getCellListPy()
            {
            boost::shared_array<unsigned int> return_array = boost::shared_array<unsigned int>(new unsigned int[m_np*2]);
            memcpy((void*)return_array.get(), d_cidx_array, sizeof(unsigned int)*m_np*2);
            unsigned int *arr = return_array.get();
            std::vector<intp> dims(2);
            dims[0] = m_np;
            dims[1] = 2;
            return num_util::makeNum(arr, dims);
            }

        //! Get a reference to the cell list used on the gpu array
        const unsigned int* getIterList() const
            {
            return d_it_array;
            }

        //! Get a reference to the cell list used on the gpu array
        boost::python::numeric::array getIterListPy()
            {
            boost::shared_array<unsigned int> return_array = boost::shared_array<unsigned int>(new unsigned int[m_nc*2]);
            memcpy((void*)return_array.get(), d_it_array, sizeof(unsigned int)*m_nc*2);
            unsigned int *arr = return_array.get();
            std::vector<intp> dims(2);
            dims[0] = m_nc;
            dims[1] = 2;
            return num_util::makeNum(arr, dims);
            }

        //! Get a reference to the cell list used on the gpu array
        const unsigned int* getCellNeighborList() const
            {
            return d_cell_neighbors;
            }

        //! Get a reference to the cell list used on the gpu array
        boost::python::numeric::array getCellNeighborListPy()
            {
            int arr_size = (int)((2*m_num_neighbors.x + 1)*(2*m_num_neighbors.y + 1)*(2*m_num_neighbors.z + 1));
            boost::shared_array<unsigned int> return_array = boost::shared_array<unsigned int>(new unsigned int[m_nc*arr_size]);
            memcpy((void*)return_array.get(), d_cell_neighbors, sizeof(unsigned int)*m_nc*arr_size);
            unsigned int *arr = return_array.get();
            std::vector<intp> dims(2);
            dims[0] = m_nc;
            dims[1] = arr_size;
            return num_util::makeNum(arr, dims);
            }

        //! Get a list of neighbors to a cell
        const unsigned int* getCellNeighbors(unsigned int cell) const
            {
            unsigned int* return_array = new unsigned int[m_total_num_neighbors];
            memcpy((void*)return_array, d_cell_neighbors + (cell*m_total_num_neighbors), m_total_num_neighbors);
            return return_array;
            }

        //! Python wrapper for getCellNeighbors
        boost::python::numeric::array getCellNeighborsPy(unsigned int cell)
            {
            const unsigned int *start = getCellNeighbors(cell);
            return num_util::makeNum(start, m_total_num_neighbors);
            }

        //! Get the simulation box
        const trajectory::CudaBox& getBox() const
            {
            return d_box;
            }

        //! Get the cell indexer
        const Index3D& getCellIndexer() const
            {
            return m_cell_index;
            }

        //! Get the neighbor indexer
        const Index2D& getNeighborIndexer() const
            {
            return m_expanded_thread_indexer;
            }

        //! Get the number of cells
        unsigned int getNumCells() const
            {
            return m_cell_index.getNumElements();
            }

        //! Get the cell width
        float getCellWidth() const
            {
            return m_cell_width;
            }

        //! Compute the cell id for a given particle idx
        // this is just a simple wrapper
        unsigned int getCell(const unsigned int& p) const
            {
            return d_pidx_array[(int)p];
            }

        //! Compute the cell id for a given position
        unsigned int getCell(const float3& p) const
            {
            uint3 c = getCellCoord(p);
            return m_cell_index(c.x, c.y, c.z);
            }

        //! Wrapper for python to getCell (1D index)
        unsigned int getCellPy(boost::python::numeric::array p)
            {
            // validate input type and rank
            num_util::check_type(p, NPY_FLOAT);
            num_util::check_rank(p, 1);

            // validate that the 2nd dimension is only 3
            num_util::check_size(p, 3);

            // get the raw data pointers and compute the cell index
            float3* p_raw = (float3*) num_util::data(p);
            return getCell(*p_raw);
            }

        //! Compute cell coordinates for a given position
        uint3 getCellCoord(const float3 p) const
            {
            float3 alpha = d_box.makeFraction(p);
            uint3 c;
            c.x = floorf(alpha.x * float(m_cell_index.getW()));
            c.x %= m_cell_index.getW();
            c.y = floorf(alpha.y * float(m_cell_index.getH()));
            c.y %= m_cell_index.getH();
            c.z = floorf(alpha.z * float(m_cell_index.getD()));
            c.z %= m_cell_index.getD();
            return c;
            }

        const uint3 getNumCellNeighbors() const
            {
            return m_num_neighbors;
            }

        const int getTotalNumNeighbors() const
            {
            return m_total_num_neighbors;
            }

        //! Compute the cell list
        void computeCellList(trajectory::CudaBox& box, const float3 *points, unsigned int Np);

        //! Python wrapper for computeCellList
        void computeCellListPy(trajectory::CudaBox& box, boost::python::numeric::array points);

    private:
        //! Rounding helper function.
        static unsigned int roundDown(unsigned int v, unsigned int m);
        static unsigned int roundUp(unsigned int v, unsigned int m);

        trajectory::CudaBox d_box;            //!< Simulation box the particles belong in
        Index3D m_cell_index;       //!< Indexer to compute cell indices
        Index2D m_expanded_thread_indexer;       //!< Indexer to the expanded cell neighbors list
        int m_total_num_neighbors; //!< Number of neighbors for each cell
        unsigned int m_np;          //!< Number of particles last placed into the cell list
        unsigned int m_nc;          //!< Number of cells last used
        float m_cell_width;         //!< Minimum necessary cell width cutoff
        float m_r_cut;         //!< desired width to investigate
        uint3 m_celldim; //!< Cell dimensions
        uint3 m_num_neighbors; //!< Number of neighbors in one direction (or two idk)

        unsigned int *d_cp_array;    //!< The list of cell indices
        unsigned int *d_cc_array;    //!< The list of cell indices
        // consider making copies of these for easier python export
        unsigned int *d_cidx_array;    //!< The list of particle and cell indices, sorted by cell (.y)
        unsigned int *d_it_array;    //!< The array that holds the first/last idx of a cell
        unsigned int *d_pidx_array;    //!< The list of particle indices
        float3 *d_point_array;    //!< The list of particle indices

        // This needs to be reimagined
        // std::vector< std::vector<unsigned int> > m_cell_neighbors;    //!< List of cell neighbors to each cell
        unsigned int *d_cell_neighbors; //!< List of cell neighbors to each cell

        //! Helper function to compute cell neighbors
        void computeCellNeighbors();

    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_CudaCell();

}; }; // end namespace freud::cudacell

#endif // _CUDACELL_H__
