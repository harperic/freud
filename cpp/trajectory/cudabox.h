#include "HOOMDMath.h"


#ifndef _CUDABOX_H__
#define _CUDABOX_H__

#ifdef NVCC
// #define HOSTDEVICE __host__ __device__ inline
#define HOSTDEVICE __host__ __device__
#endif

/*! \file trajectory.h
    \brief Helper routines for trajectory
*/

namespace freud { namespace trajectory {

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
/*! Box stores a standard hoomd simulation box that goes from -L/2 to L/2 in each dimension, allowing Lx, Ly, Lz, and triclinic tilt factors xy, xz, and yz to be specified independently.
 *

    A number of utility functions are provided to work with coordinates in boxes. These are provided as inlined methods
    in the header file so they can be called in inner loops without sacrificing performance.
     - wrap()
     - unwrap()

    A Box can represent either a two or three dimensional box. By default, a Box is 3D, but can be set as 2D with the
    method set2D(), or via an optional boolean argument to the constructor. is2D() queries if a Box is 2D or not.
    2D boxes have a "volume" of Lx * Ly, and Lz is set to 0. To keep programming simple, all inputs and outputs are
    still 3-component vectors even for 2D boxes. The third component ignored (assumed set to 0).
*/
class CudaBox
    {
    public:
        //! Construct a box of length 0.
        HOSTDEVICE CudaBox() //Lest you think of removing this, it's needed by the DCDLoader. No touching.
            {
            m_2d = false; //Assign before calling setL!
            setL(0,0,0);
            m_periodic = make_uchar3(1,1,1);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct a cubic box
        HOSTDEVICE CudaBox(float L, bool _2d=false)
            {
            m_2d = _2d; //Assign before calling setL!
            setL(L,L,L);
            m_periodic = make_uchar3(1,1,1);
            m_xy = m_xz = m_yz = 0;
            }
        //! Construct an orthorhombic box
        HOSTDEVICE CudaBox(float Lx, float Ly, float Lz, bool _2d=false)
            {
            m_2d = _2d;  //Assign before calling setL!
            setL(Lx,Ly,Lz);
            m_periodic = make_uchar3(1,1,1);
            m_xy = m_xz = m_yz = 0;
            }

        //! Construct a triclinic box
        HOSTDEVICE CudaBox(float Lx, float Ly, float Lz, float xy, float xz, float yz, bool _2d=false)
            {
            m_2d = _2d;  //Assign before calling setL!
            setL(Lx,Ly,Lz);
            m_periodic = make_uchar3(1,1,1);
            m_xy = xy; m_xz = xz; m_yz = yz;
            }

        HOSTDEVICE inline bool operator ==(const CudaBox&b) const
            {
            return ( (this->getLx() == b.getLx()) &&
                     (this->getLy() == b.getLy()) &&
                     (this->getLz() == b.getLz()) &&
                     (this->getTiltFactorXY() == b.getTiltFactorXY()) &&
                     (this->getTiltFactorXZ() == b.getTiltFactorXZ()) &&
                     (this->getTiltFactorYZ() == b.getTiltFactorYZ()) );
            }

        HOSTDEVICE inline bool operator !=(const CudaBox&b) const
            {
            return ( (this->getLx() != b.getLx()) ||
                     (this->getLy() != b.getLy()) ||
                     (this->getLz() != b.getLz()) ||
                     (this->getTiltFactorXY() != b.getTiltFactorXY()) ||
                     (this->getTiltFactorXZ() != b.getTiltFactorXZ()) ||
                     (this->getTiltFactorYZ() != b.getTiltFactorYZ()) );
            }

        //! Set L, box lengths, inverses.  Box is also centered at zero.
        HOSTDEVICE void setL(const float3 L)
            {
            setL(L.x,L.y,L.z);
            }

        //! Set L, box lengths, inverses.  Box is also centered at zero.
        HOSTDEVICE void setL(const float Lx, const float Ly, const float Lz)
            {
            m_L = make_float3(Lx,Ly,Lz);
            m_hi = make_float3(m_L.x/float(2.0), m_L.y/float(2.0), m_L.z/float(2.0));
            m_lo = make_float3(-m_L.x/float(2.0), -m_L.y/float(2.0), -m_L.z/float(2.0));;

            if(m_2d)
                {
                m_Linv = make_float3(1/m_L.x, 1/m_L.y, 0);
                m_L.z = float(0);
                }
            else
                {
                m_Linv = make_float3(1/m_L.x, 1/m_L.y, 1/m_L.z);
                }
            }

        //! Set whether box is 2D
        HOSTDEVICE void set2D(bool _2d)
            {
            m_2d = _2d;
            m_L.z = 0;
            m_Linv.z =0;
            }

        //! Returns whether box is two dimensional
        HOSTDEVICE bool is2D() const
            {
            return m_2d;
            }


        //! Get the value of Lx
        HOSTDEVICE float getLx() const
            {
            return m_L.x;
            }
        //! Get the value of Ly
        HOSTDEVICE float getLy() const
            {
            return m_L.y;
            }
        //! Get the value of Lz
        HOSTDEVICE float getLz() const
            {
            return m_L.z;
            }
        //! Get current L
        HOSTDEVICE float3 getL() const
            {
            return m_L;
            }
        //! Get current stored inverse of L
        HOSTDEVICE float3 getLinv() const
            {
            return m_Linv;
            }

        //! Get tilt factor xy
        HOSTDEVICE float getTiltFactorXY() const
            {
            return m_xy;
            }
        //! Get tilt factor xz
        HOSTDEVICE float getTiltFactorXZ() const
            {
            return m_xz;
            }
        //! Get tilt factor yz
        HOSTDEVICE float getTiltFactorYZ() const
            {
            return m_yz;
            }

        //! Get the volume of the box (area in 2D)
        HOSTDEVICE float getVolume() const
            {
            //TODO:  Unit test these
            if (m_2d)
                return m_L.x*m_L.y;
            else
                return m_L.x*m_L.y*m_L.z;
            }

        //! Compute the position of the particle in box relative coordinates
        /*! \param p point
            \returns alpha

            alpha.x is 0 when \a x is on the far left side of the box and 1.0 when it is on the far right. If x is
            outside of the box in either direction, it will go larger than 1 or less than 0 keeping the same scaling.
        */
        HOSTDEVICE float3 makeFraction(const float3& v, const float3& ghost_width=make_float3(0.0,0.0,0.0)) const
            {
            float3 delta;
            delta.x = v.x - m_lo.x;
            delta.y = v.y - m_lo.y;
            delta.z = v.z - m_lo.z;

            delta.x -= (m_xz-m_yz*m_xy)*v.z+m_xy*v.y;
            delta.y -= m_yz * v.z;
            float3 result;
            result.x = (delta.x + ghost_width.x)/(m_L.x + float(2.0)*ghost_width.x);
            result.y = (delta.y + ghost_width.y)/(m_L.y + float(2.0)*ghost_width.y);
            result.z = (delta.z + ghost_width.z)/(m_L.z + float(2.0)*ghost_width.z);
            return result;
            }

        //! Convert fractional coordinates into real coordinates
        /*! \param f Fractional coordinates between 0 and 1 within parallelpipedal box
            \return A vector inside the box corresponding to f
        */
        HOSTDEVICE float3 makeCoordinates(const float3 &f) const
            {
            float3 v = make_float3(m_lo.x + f.x*m_L.x, m_lo.y + f.y*m_L.y, m_lo.z + f.z*m_L.z);
            v.x += m_xy*v.y+m_xz*v.z;
            v.y += m_yz*v.z;
            return v;
            }

        //! Get the periodic image a vector belongs to
        /*! \param v The vector to check
            \returns the integer coordinates of the periodic image
         */
        HOSTDEVICE int3 getImage(const float3 &v) const
            {
            float3 f = makeFraction(v);
            f.x -= 0.5;
            f.y -= 0.5;
            f.z -= 0.5;
            int3 img;
            img.x = (int)((f.x >= float(0.0)) ? f.x + float(0.5) : f.x - float(0.5));
            img.y = (int)((f.y >= float(0.0)) ? f.y + float(0.5) : f.y - float(0.5));
            img.z = (int)((f.z >= float(0.0)) ? f.z + float(0.5) : f.z - float(0.5));
            return img;
            }

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \param flags Vector of flags to force wrapping along certain directions
            \post \a img and \a v are updated appropriately
            \note \a v must not extend more than 1 image beyond the box
        */
        HOSTDEVICE void wrap(float3& w, int3& img, char3 flags = make_char3(0,0,0)) const
            {
            float3 L = getL();

            if (m_periodic.x)
                {
                float tilt_x = (m_xz - m_xy*m_yz) * w.z + m_xy * w.y;
                if (((w.x >= m_hi.x + tilt_x) && !flags.x) || flags.x == 1)
                    {
                    w.x -= L.x;
                    img.x++;
                    }
                else if (((w.x < m_lo.x + tilt_x) && !flags.x) || flags.x == -1)
                    {
                    w.x += L.x;
                    img.x--;
                    }
                }

            if (m_periodic.y)
                {
                float tilt_y = m_yz * w.z;
                if (((w.y >= m_hi.y + tilt_y) && !flags.y)  || flags.y == 1)
                    {
                    w.y -= L.y;
                    w.x -= L.y * m_xy;
                    img.y++;
                    }
                else if (((w.y < m_lo.y + tilt_y) && !flags.y) || flags.y == -1)
                    {
                    w.y += L.y;
                    w.x += L.y * m_xy;
                    img.y--;
                    }
                }

            if (m_periodic.z)
                {
                if (((w.z >= m_hi.z) && !flags.z) || flags.z == 1)
                    {
                    w.z -= L.z;
                    w.y -= L.z * m_yz;
                    w.x -= L.z * m_xz;
                    img.z++;
                    }
                else if (((w.z < m_lo.z) && !flags.z) || flags.z == -1)
                    {
                    w.z += L.z;
                    w.y += L.z * m_yz;
                    w.x += L.z * m_xz;
                    img.z--;
                    }
                }
           }

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \param flags Vector of flags to force wrapping along certain directions
            \returns w;

            \note \a w must not extend more than 1 image beyond the box
        */
        //Is this even sane? I assume since we previously had image free version
        // that I can just use our new getImage to pass through and make as few as possible
        // changes to the codebase here.
        // Followup: I don't remember why I put this comment here, better refer later to
        // original trajectory.h

        HOSTDEVICE float3 wrap(const float3& w, const char3 flags = make_char3(0,0,0)) const
            {
            float3 wcopy = w;
            int3 img = getImage(w);
            wrap(wcopy, img, flags);
            return wcopy;
            }

        //! Unwrap a given position to its "real" location
        /*! \param p coordinates to unwrap
            \param image image flags for this point
            \returns The unwrapped coordinates
        */
        HOSTDEVICE float3 unwrap(const float3& p, const int3& image) const
            {
            float3 newp = p;

            newp.x += getLatticeVector(0).x * float(image.x);
            newp.y += getLatticeVector(0).y * float(image.x);
            newp.z += getLatticeVector(0).z * float(image.x);

            newp.x += getLatticeVector(1).x * float(image.y);
            newp.y += getLatticeVector(1).y * float(image.y);
            newp.z += getLatticeVector(1).z * float(image.y);

            if(!m_2d)
                {
                newp.x += getLatticeVector(2).x * float(image.z);
                newp.y += getLatticeVector(2).y * float(image.z);
                newp.z += getLatticeVector(2).z * float(image.z);
                }
            return newp;
            }
        //! Get the shortest distance between opposite boundary planes of the box
        /*! The distance between two planes of the lattice is 2 Pi/|b_i|, where
         *   b_1 is the reciprocal lattice vector of the Bravais lattice normal to
         *   the lattice vectors a_2 and a_3 etc.
         *
         * \return A vec3<float> containing the distance between the a_2-a_3, a_3-a_1 and
         *         a_1-a_2 planes for the triclinic lattice
         */
        HOSTDEVICE float3 getNearestPlaneDistance() const
            {
            float3 dist;
            dist.x = m_L.x/sqrt(1.0f + m_xy*m_xy + (m_xy*m_yz - m_xz)*(m_xy*m_yz - m_xz));
            dist.y = m_L.y/sqrt(1.0f + m_yz*m_yz);
            dist.z = m_L.z;

            return dist;
            }

        /*! Get the lattice vector with index i
            \param i Index (0<=i<d) of the lattice vector, where d is dimension (2 or 3)
            \returns the lattice vector with index i
         */
        HOSTDEVICE float3 getLatticeVector(unsigned int i) const
            {
            if (i == 0)
                {
                return make_float3(m_L.x,0.0,0.0);
                }
            else if (i == 1)
                {
                return make_float3(m_L.y*m_xy, m_L.y, 0.0);
                }
            else if (i == 2 && !m_2d)
                {
                return make_float3(m_L.z*m_xz, m_L.z*m_yz, m_L.z);
                }
            // else
            //     {
            //     throw std::out_of_range("box lattice vector index requested does not exist");
            //     }
            return make_float3(0.0,0.0,0.0);
            }

        HOSTDEVICE uchar3 getPeriodic() const
            {
            return m_periodic;
            }

        //! Set the periodic flags
        /*! \param periodic Flags to set
            \post Period flags are set to \a periodic
            \note It is invalid to set 1 for a periodic dimension where lo != -hi. This error is not checked for.
        */
        HOSTDEVICE void setPeriodic(uchar3 periodic)
            {
            m_periodic = periodic;
            }

    private:
        float3 m_lo;      //!< Minimum coords in the box
        float3 m_hi;      //!< Maximum coords in the box
        float3 m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
        float3 m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
        float m_xy;       //!< xy tilt factor
        float m_xz;       //!< xz tilt factor
        float m_yz;       //!< yz tilt factor
        uchar3 m_periodic;//!< 0/1 in each direction to tell if the box is periodic in that direction
        bool m_2d;        //!< Specify whether box is 2D.
    };

}; };

#endif // _CUDABOX_H__


