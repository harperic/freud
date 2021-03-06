// Copyright (c) 2010-2016 The Regents of the University of Michigan
// This file is part of the Freud project, released under the BSD 3-Clause License.

#include "HexOrderParameter.h"
#include "ScopedGILRelease.h"

#include <stdexcept>
#include <complex>

using namespace std;
using namespace tbb;

/*! \file HexOrderParameter.h
    \brief Compute the hexatic order parameter for each particle
*/

namespace freud { namespace order {

HexOrderParameter::HexOrderParameter(float rmax, float k, unsigned int n)
    : m_box(box::Box()), m_rmax(rmax), m_k(k), m_Np(0)
    {
    m_nn = new locality::NearestNeighbors(m_rmax, n==0? (unsigned int) k: n);
    }

HexOrderParameter::~HexOrderParameter()
    {
    delete m_nn;
    }

void HexOrderParameter::compute(box::Box& box, const vec3<float> *points, unsigned int Np)
    {
    // compute the cell list
    m_box = box;
    m_nn->compute(m_box,points,Np,points,Np);
    m_nn->setRMax(m_rmax);

    // reallocate the output array if it is not the right size
    if (Np != m_Np)
        {
        m_psi_array = std::shared_ptr<complex<float> >(new complex<float> [Np], std::default_delete<complex<float>[]>());
        }

    // compute the order parameter
    parallel_for(blocked_range<size_t>(0,Np),
        [=] (const blocked_range<size_t>& r)
        {

        for(size_t i=r.begin(); i!=r.end(); ++i)
            {
            m_psi_array.get()[i] = 0;
            vec3<float> ref = points[i];

            //loop over neighbors
            locality::NearestNeighbors::iteratorneighbor it = m_nn->iterneighbor(i);
            for (unsigned int j = it.begin(); !it.atEnd(); j = it.next())
                {

                //compute r between the two particles
                vec3<float> delta = m_box.wrap(points[j] - ref);

                float rsq = dot(delta, delta);
                if (rsq > 1e-6)
                    {
                    //compute psi for neighboring particle(only constructed for 2d)
                    float psi_ij = atan2f(delta.y, delta.x);
                    m_psi_array.get()[i] += exp(complex<float>(0,m_k*psi_ij));
                    }
                }
            m_psi_array.get()[i] /= complex<float>(m_k);
            }
        });
    // save the last computed number of particles
    m_Np = Np;
    }

}; }; // end namespace freud::order
