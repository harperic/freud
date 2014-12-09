#include "CUDAPMFT2D.h"
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
    \file CUDAPMFT2D.cc
    \brief Routines for computing 2D anisotropic potential of mean force
*/

namespace freud { namespace cudapmft {

CUDAPMFT2D::CUDAPMFT2D(float max_x, float max_y, float dx, float dy)
    : m_box(trajectory::Box()), d_box(trajectory::CudaBox()), m_max_x(max_x), m_max_y(max_y), m_dx(dx), m_dy(dy)
    {
    if (dx < 0.0f)
        throw invalid_argument("dx must be positive");
    if (dy < 0.0f)
        throw invalid_argument("dy must be positive");
    if (max_x < 0.0f)
        throw invalid_argument("max_x must be positive");
    if (max_y < 0.0f)
        throw invalid_argument("max_y must be positive");
    if (dx > max_x)
        throw invalid_argument("max_x must be greater than dx");
    if (dy > max_y)
        throw invalid_argument("max_y must be greater than dy");

    m_nbins_x = int(2 * floorf(m_max_x / m_dx));
    assert(m_nbins_x > 0);
    m_nbins_y = int(2 * floorf(m_max_y / m_dy));
    assert(m_nbins_y > 0);

    // precompute the bin center positions for x
    createCudaArray(&d_x_array, sizeof(float)*m_nbins_x);
    m_x_array = boost::shared_array<float>(new float[m_nbins_x]);
    for (unsigned int i = 0; i < m_nbins_x; i++)
        {
        float x = float(i) * m_dx;
        float nextx = float(i+1) * m_dx;
        m_x_array[i] = -m_max_x + ((x + nextx) / 2.0);
        d_x_array[i] = m_x_array[i];
        }

    // precompute the bin center positions for y
    createCudaArray(&d_y_array, sizeof(float)*m_nbins_y);
    m_y_array = boost::shared_array<float>(new float[m_nbins_y]);
    for (unsigned int i = 0; i < m_nbins_y; i++)
        {
        float y = float(i) * m_dy;
        float nexty = float(i+1) * m_dy;
        m_y_array[i] = -m_max_y + ((y + nexty) / 2.0);
        d_y_array[i] = m_y_array[i];
        }

    // create and populate the pcf_array
    createPMFTArray(&d_pcf_array, m_arrSize, m_memSize, m_nbins_x, m_nbins_y);
    memset((void*)d_pcf_array, 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);
    m_pcf_array = boost::shared_array<unsigned int>(new unsigned int[m_nbins_x * m_nbins_y]);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);

    // m_lc = new locality::LinkCell(m_box, sqrtf(m_max_x*m_max_x + m_max_y*m_max_y));
    }

CUDAPMFT2D::~CUDAPMFT2D()
    {
    freeCudaArray(&d_x_array);
    freeCudaArray(&d_y_array);
    freePMFTArray(&d_pcf_array);
    // delete m_lc;
    }

//! \internal
/*! \brief Function to reset the pcf array if needed e.g. calculating between new particle types
*/

void CUDAPMFT2D::resetPCF()
    {
    memset((void*)d_pcf_array, 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);
    memset((void*)m_pcf_array.get(), 0, sizeof(unsigned int)*m_nbins_x*m_nbins_y);
    }

//! \internal
/*! \brief Helper functionto direct the calculation to the correct helper class
*/

void CUDAPMFT2D::compute(vec3<float> *ref_points,
                      float *ref_orientations,
                      unsigned int Nref,
                      vec3<float> *points,
                      float *orientations,
                      unsigned int Np)
    {
    // m_lc->computeCellList(m_box, points, Np);

    // run the cuda kernel
    // don't need to explicitly accumulate since d is already populated
    CallMyFirstKernel(d_pcf_array, m_arrSize, d_box);
    }

//! \internal
/*! \brief Exposed function to python to calculate the PMF
*/

void CUDAPMFT2D::computePy(trajectory::Box& box,
                        boost::python::numeric::array ref_points,
                        boost::python::numeric::array ref_orientations,
                        boost::python::numeric::array points,
                        boost::python::numeric::array orientations)
    {
    // validate input type and rank
    m_box = box;
    d_box = trajectory::CudaBox(m_box.getLx(),
                                m_box.getLy(),
                                m_box.getLz(),
                                m_box.getTiltFactorXY(),
                                m_box.getTiltFactorXZ(),
                                m_box.getTiltFactorYZ(),
                                m_box.is2D());
    num_util::check_type(ref_points, NPY_FLOAT);
    num_util::check_rank(ref_points, 2);
    num_util::check_type(ref_orientations, NPY_FLOAT);
    num_util::check_rank(ref_orientations, 1);
    num_util::check_type(points, NPY_FLOAT);
    num_util::check_rank(points, 2);
    num_util::check_type(orientations, NPY_FLOAT);
    num_util::check_rank(orientations, 1);

    // validate that the 2nd dimension is only 3
    num_util::check_dim(points, 1, 3);
    unsigned int Np = num_util::shape(points)[0];

    num_util::check_dim(ref_points, 1, 3);
    unsigned int Nref = num_util::shape(ref_points)[0];

    // check the size of angles to be correct
    num_util::check_dim(ref_orientations, 0, Nref);
    num_util::check_dim(orientations, 0, Np);

    // get the raw data pointers and compute the cell list
    vec3<float>* ref_points_raw = (vec3<float>*) num_util::data(ref_points);
    float* ref_orientations_raw = (float*) num_util::data(ref_orientations);
    vec3<float>* points_raw = (vec3<float>*) num_util::data(points);
    float* orientations_raw = (float*) num_util::data(orientations);

        // compute with the GIL released
        {
        util::ScopedGILRelease gil;
        compute(ref_points_raw,
                ref_orientations_raw,
                Nref,
                points_raw,
                orientations_raw,
                Np);
        }
    }

void export_CUDAPMFT2D()
    {
    class_<CUDAPMFT2D>("CUDAPMFT2D", init<float, float, float, float>())
        .def("getBox", &CUDAPMFT2D::getBox, return_internal_reference<>())
        .def("compute", &CUDAPMFT2D::computePy)
        .def("getPCF", &CUDAPMFT2D::getPCFPy)
        .def("getX", &CUDAPMFT2D::getXPy)
        .def("getY", &CUDAPMFT2D::getYPy)
        .def("resetPCF", &CUDAPMFT2D::resetPCFPy)
        ;
    }

}; }; // end namespace freud::pmft
