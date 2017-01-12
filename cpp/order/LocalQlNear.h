// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include <memory>
#include <complex>

#include "HOOMDMath.h"
#define swap freud_swap
#include "VectorMath.h"
#undef swap

#include "NearestNeighbors.h"
#include "box.h"
#include "../../extern/fsph/src/spherical_harmonics.hpp"

#ifndef _LOCAL_QL_NEAR_H__
#define _LOCAL_QL_NEAR_H__

/*! \file LocalQlNear.h
    \brief Compute a Ql per particle using N nearest neighbors instead of r_cut
*/

namespace freud { namespace order {

//! Compute the local Steinhardt rotationally invariant Ql order parameter for a set of points
/*!
 * Implements the local rotationally invariant Ql order parameter described by Steinhardt.
 * For a particle i, we calculate the average Q_l by summing the spherical harmonics between particle i and its neighbors j in a local region:
 * \f$ \overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b} Y_{lm}(\theta(\vec{r}_{ij}),\phi(\vec{r}_{ij})) \f$
 *
 * This is then combined in a rotationally invariant fashion to remove local orientational order as follows:
 * \f$ Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }  \f$
 *
 * For more details see PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
*/
//! Added first/second shell combined average Ql order parameter for a set of points
/*!
 * Variation of the Steinhardt Ql order parameter
 * For a particle i, we calculate the average Q_l by summing the spherical harmonics between particle i and its neighbors j and the neighbors k of neighbor j in a local region:
 *
 * For more details see Wolfgan Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
*/
class LocalQlNear
    {
    public:
        //! LocalQlNear Class Constructor
        /**Constructor for LocalQl  analysis class.
        @param box A freud box object containing the dimensions of the box associated with the particles that will be fed into compute.
        @param rmax Cutoff radius for running the local order parameter. Values near first minima of the rdf are recommended.
        @param l Spherical harmonic quantum number l.  Must be a positive even number.
        @param kn Number of nearest neighbors.  Must be a positive integer.
        **/
        //! Constructor
        LocalQlNear(const box::Box& box, float rmax, unsigned int l, unsigned int kn=12);

        //! Destructor
        ~LocalQlNear();

        //! Get the simulation box
        const box::Box& getBox() const
            {
            return m_box;
            }

        //! Reset the simulation box size
        void setBox(const box::Box newbox)
            {
            m_box = newbox;  //Set
            delete m_nn;
            m_nn = new locality::NearestNeighbors(m_rmax, m_k);
            }


        //! Compute the local rotationally invariant Ql order parameter
        // void compute(const float3 *points,
        //              unsigned int Np);
        void compute(const vec3<float> *points,
                     unsigned int Np);

        // //! Python wrapper for computing the order parameter from a Nx3 numpy array of float32.
        // void computePy(boost::python::numeric::array points);

        //! Compute the local rotationally invariant (with 2nd shell) Ql order parameter
        // void computeAve(const float3 *points,
        //                 unsigned int Np);
        void computeAve(const vec3<float> *points,
                        unsigned int Np);

        // //! Python wrapper for computing the order parameter (with 2nd shell) from a Nx3 numpy array of float32.
        // void computeAvePy(boost::python::numeric::array points);

        //! Compute the Ql order parameter globally (averaging over the system Qlm)
        // void computeNorm(const float3 *points,
        //                  unsigned int Np);
        void computeNorm(const vec3<float> *points,
                         unsigned int Np);

        // //! Python wrapper for computing the global Ql order parameter from Nx3 numpy array of float32
        // void computeNormPy(boost::python::numeric::array points);

        //! Compute the Ql order parameter globally (averaging over the system AveQlm)
        // void computeAveNorm(const float3 *points,
        //                  unsigned int Np);
        void computeAveNorm(const vec3<float> *points,
                         unsigned int Np);

        // //! Python wrapper for computing the global Ql order parameter from Nx3 numpy array of float32
        // void computeAveNormPy(boost::python::numeric::array points);


        //! Get a reference to the last computed Ql for each particle.  Returns NaN instead of Ql for particles with no neighbors.
        std::shared_ptr< float > getQl()
            {
            return m_Qli;
            }

        // //! Python wrapper for getQl() (returns a copy of array).  Returns NaN instead of Ql for particles with no neighbors.
        // boost::python::numeric::array getQlPy()
        //     {
        //     float *arr = m_Qli.get();
        //     return num_util::makeNum(arr, m_Np);
        //     }

        //! Get a reference to the last computed AveQl for each particle.  Returns NaN instead of AveQl for particles with no neighbors.
        std::shared_ptr< float > getAveQl()
            {
            return m_AveQli;
            }

        // //! Python wrapper for getAveQl() (returns a copy of array).  Returns NaN instead of AveQl for particles with no neighbors.
        // boost::python::numeric::array getAveQlPy()
        //     {
        //     float *arr = m_AveQli.get();
        //     return num_util::makeNum(arr, m_Np);
        //     }

        //! Get a reference to the last computed QlNorm for each particle.  Returns NaN instead of QlNorm for particles with no neighbors.
        std::shared_ptr< float > getQlNorm()
            {
            return m_QliNorm;
            }

        // //! Python wrapper for getQlNorm() (returns a copy of array). Returns NaN instead of QlNorm for particles with no neighbors.
        // boost::python::numeric::array getQlNormPy()
        //     {
        //     float *arr = m_QliNorm.get();
        //     return num_util::makeNum(arr, m_Np);
        //     }

        //! Get a reference to the last computed QlNorm for each particle.  Returns NaN instead of QlNorm for particles with no neighbors.
        std::shared_ptr< float > getQlAveNorm()
            {
            return m_QliAveNorm;
            }

        unsigned int getNP()
            {
            return m_Np;
            }

        // //! Python wrapper for getQlNorm() (returns a copy of array). Returns NaN instead of QlNorm for particles with no neighbors.
        // boost::python::numeric::array getQlAveNormPy()
        //     {
        //     float *arr = m_QliAveNorm.get();
        //     return num_util::makeNum(arr, m_Np);
        //     }

        //!Spherical harmonics calculation for Ylm filling a vector<complex<float>> with values for m = -l..l.
        void Ylm(const float theta, const float phi, std::vector<std::complex<float> > &Y);

    private:
        box::Box m_box;            //!< Simulation box the particles belong in
        float m_rmin;                     //!< Minimum r at which to determine neighbors
        float m_rmax;                     //!< Maximum r at which to determine neighbors
        float m_k;             //!< Number of neighbors
        locality::NearestNeighbors *m_nn;          //!< NearestNeighbors to bin particles for the computation
        unsigned int m_l;                 //!< Spherical harmonic l value.
        unsigned int m_Np;                //!< Last number of points computed
        std::shared_ptr< std::complex<float> > m_Qlmi;        //!  Qlm for each particle i
        std::shared_ptr< float > m_Qli;         //!< Ql locally invariant order parameter for each particle i;
        std::shared_ptr< std::complex<float> > m_AveQlmi;     //! AveQlm for each particle i
        std::shared_ptr< float > m_AveQli;     //!< AveQl locally invariant order parameter for each particle i;
        std::shared_ptr< std::complex<float> > m_Qlm;  //! NormQlm for the system
        std::shared_ptr< float > m_QliNorm;   //!< QlNorm order parameter for each particle i
        std::shared_ptr< std::complex<float> > m_AveQlm; //! AveNormQlm for the system
        std::shared_ptr< float > m_QliAveNorm;     //! < QlAveNorm order paramter for each particle i
    };

}; }; // end namespace freud::order

#endif // #define _LOCAL_QL_NEAR_H__
