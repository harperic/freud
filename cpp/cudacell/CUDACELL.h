#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....

#include "HOOMDMath.h"

#include "CUDACELL.cuh"
#include "cudabox.h"
#include "Index1D.h"

#ifndef _CUDACELL_H__
#define _CUDACELL_H__

#ifdef NVCC
// #define HOSTDEVICE __host__ __device__ inline
#define HOSTDEVICE __host__ __device__
#endif


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
        HOSTDEVICE CudaCell(const trajectory::CudaBox& box, float cell_width);

        //! Null Constructor for triclinic behavior
        HOSTDEVICE CudaCell();

        //! Destructor
        HOSTDEVICE ~CudaCell();

        //! Update cell_width
        HOSTDEVICE void setCellWidth(float cell_width);

        //! Update cell_width
        HOSTDEVICE void setRCut(float r_cut);

        //! Update box used in linkCell
        HOSTDEVICE void updateBox(const trajectory::CudaBox& box);

        //! Compute LinkCell dimensions
        HOSTDEVICE const uint3 computeDimensions(const trajectory::CudaBox& box, float cell_width) const;

        //! Get the simulation box
        HOSTDEVICE const trajectory::CudaBox& getBox() const
            {
            return d_box;
            }

        //! Get the point list
        HOSTDEVICE const unsigned int *getPIDX() const
            {
            return d_pidx_array;
            }

        //! Get the cell list
        HOSTDEVICE const uint2 *getCIDX() const
            {
            return d_cidx_array;
            }

        //! Get the cell indexer
        HOSTDEVICE const Index3D& getCellIndexer() const
            {
            return m_cell_index;
            }

        //! Get the number of cells
        HOSTDEVICE unsigned int getNumCells() const
            {
            return m_cell_index.getNumElements();
            }

        //! Get the cell width
        HOSTDEVICE float getCellWidth() const
            {
            return m_cell_width;
            }

        // recode to just use the data
        //! Compute the cell id for a given position
        HOSTDEVICE unsigned int getCell(const float3& p) const
            {
            uint3 c = getCellCoord(p);
            return m_cell_index(c.x, c.y, c.z);
            }

        //! Compute cell coordinates for a given position
        HOSTDEVICE uint3 getCellCoord(const float3 p) const
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

        //! Compute the cell list
        HOSTDEVICE void computeCellList(trajectory::CudaBox& box, const float3 *points, unsigned int Np);

    private:
        //! Rounding helper function.
        HOSTDEVICE static unsigned int roundDown(unsigned int v, unsigned int m);
        HOSTDEVICE static unsigned int CudaCell::roundUp(unsigned int v, unsigned int m)

        trajectory::CudaBox d_box;            //!< Simulation box the particles belong in
        Index3D m_cell_index;       //!< Indexer to compute cell indices
        unsigned int m_np;          //!< Number of particles last placed into the cell list
        unsigned int m_nc;          //!< Number of cells last used
        float m_cell_width;         //!< Minimum necessary cell width cutoff
        float m_r_cut;         //!< desired width to investigate
        uint3 m_celldim; //!< Cell dimensions
        uint3 m_num_neighbors; //!< Number of neighbors in one direction (or two idk)

        unsigned int *d_cp_array;    //!< The list of cell indices
        unsigned int *d_cc_array;    //!< The list of cell indices
        // consider making copies of these for easier python export
        uint2 *d_cidx_array;    //!< The list of particle and cell indices, sorted by cell (.y)
        uint2 *d_it_array;    //!< The array that holds the first/last idx of a cell
        unsigned int *d_pidx_array;    //!< The list of particle indices
        float3 *d_point_array;    //!< The list of particle indices

        // This needs to be reimagined
        // std::vector< std::vector<unsigned int> > m_cell_neighbors;    //!< List of cell neighborts to each cell

        //! Helper function to compute cell neighbors
        HOSTDEVICE void computeCellNeighbors();

    };

}; }; // end namespace freud::cudacell

#endif // _CUDACELL_H__
