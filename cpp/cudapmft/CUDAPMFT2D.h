#include <ostream>

// work around nasty issue where python #defines isalpha, toupper, etc....
#undef __APPLE__
#include <Python.h>
#define __APPLE__

#include <boost/python.hpp>
#include <boost/shared_array.hpp>

#include "HOOMDMath.h"
#include "VectorMath.h"

#include "CUDAPMFT2D.cuh"
#include "LinkCell.h"
#include "CUDACELL.h"
#include "num_util.h"
#include "trajectory.h"
#include "cudabox.h"
#include "Index1D.h"

#ifndef _CUDAPMFT2D_H__
#define _CUDAPMFT2D_H__

/*! \internal
    \file CUDAPMFT2D.h
    \brief Routines for computing anisotropic potential of mean force in 2D
*/

namespace freud { namespace cudapmft {

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
class CUDAPMFT2D
    {
    public:
        //! Constructor
        CUDAPMFT2D(float max_x, float max_y, float dx, float dy);

        //! Destructor
        ~CUDAPMFT2D();

        //! Get the simulation box
        const trajectory::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the PCF array to all zeros
        void resetPCF();

        //! Python wrapper for reset method
        void resetPCFPy()
            {
            resetPCF();
            }

        /*! Compute the PCF for the passed in set of points. The function will be added to previous values
            of the pcf
        */
        void compute(float3 *ref_points,
                     float *ref_orientations,
                     unsigned int Nref,
                     float3 *points,
                     float *orientations,
                     unsigned int Np);

        //! Python wrapper for compute
        void computePy(trajectory::Box& box,
                       boost::python::numeric::array ref_points,
                       boost::python::numeric::array ref_orientations,
                       boost::python::numeric::array points,
                       boost::python::numeric::array orientations);

        //! Get a reference to the PCF array
        boost::shared_array<unsigned int> getPCF()
            {
            memcpy((void*)m_pcf_array.get(), d_pcf_array, m_memSize);
            return m_pcf_array;
            }

        //! Get a reference to the x array
        boost::shared_array<float> getX()
            {
            return m_x_array;
            }

        //! Get a reference to the y array
        boost::shared_array<float> getY()
            {
            return m_y_array;
            }

        //! Python wrapper for getPCF() (returns a copy)
        boost::python::numeric::array getPCFPy()
            {
            memcpy((void*)m_pcf_array.get(), d_pcf_array, m_memSize);
            unsigned int *arr = m_pcf_array.get();
            std::vector<intp> dims(2);
            dims[0] = m_nbins_y;
            dims[1] = m_nbins_x;
            return num_util::makeNum(arr, dims);
            }

        //! Python wrapper for getX() (returns a copy)
        boost::python::numeric::array getXPy()
            {
            float *arr = m_x_array.get();
            return num_util::makeNum(arr, m_nbins_x);
            }

        //! Python wrapper for getY() (returns a copy)
        boost::python::numeric::array getYPy()
            {
            float *arr = m_y_array.get();
            return num_util::makeNum(arr, m_nbins_y);
            }

    private:
        trajectory::Box m_box;            //!< Simulation box the particles belong in
        trajectory::CudaBox d_box;            //!< Simulation box the particles belong in
        float m_max_x;                     //!< Maximum x at which to compute pcf
        float m_max_y;                     //!< Maximum y at which to compute pcf
        float m_dx;                       //!< Step size for x in the computation
        float m_dy;                       //!< Step size for y in the computation
        cudacell::CudaCell* m_cc;          //!< CudaCell to bin particles for the computation
        unsigned int m_nbins_x;             //!< Number of x bins to compute pcf over
        unsigned int m_nbins_y;             //!< Number of y bins to compute pcf over

        unsigned int *d_pcf_array;         //!< array of pcf computed
        boost::shared_array<unsigned int> m_pcf_array;         //!< array of pcf computed
        boost::shared_array<float> m_x_array;           //!< array of x values that the pcf is computed at
        float* d_x_array;           //!< array of x values that the pcf is computed at
        boost::shared_array<float> m_y_array;           //!< array of y values that the pcf is computed at
        float* d_y_array;           //!< array of y values that the pcf is computed at
        float3* d_ref_points;           //!< array of points
        float* d_ref_orientations;           //!< array of points
        float3* d_points;           //!< array of points
        float* d_orientations;           //!< array of points
        // tbb::enumerable_thread_specific<unsigned int *> m_local_pcf_array;
        unsigned int m_arrSize;
        size_t m_memSize;
        unsigned int m_nref;
        unsigned int m_np;
    };

/*! \internal
    \brief Exports all classes in this file to python
*/
void export_CUDAPMFT2D();

}; }; // end namespace freud::cudapmft

#endif // _CUDAPMFT2D_H__
