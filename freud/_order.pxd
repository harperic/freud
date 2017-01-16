# Copyright (c) 2010-2017 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._VectorMath cimport quat
from freud.util._Boost cimport shared_array
from libcpp.memory cimport shared_ptr
from libcpp.complex cimport complex
from libcpp.vector cimport vector
from libcpp.map cimport map
cimport freud._box as box

cdef extern from "BondOrder.h" namespace "freud::order":
    cdef cppclass BondOrder:
        BondOrder(float, float, unsigned int, unsigned int, unsigned int)
        const box.Box &getBox() const
        void resetBondOrder()
        void accumulate(box.Box &,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        vec3[float]*,
                        quat[float]*,
                        unsigned int,
                        unsigned int) nogil
        void reduceBondOrder()
        shared_ptr[float] getBondOrder()
        shared_ptr[float] getTheta()
        shared_ptr[float] getPhi()
        unsigned int getNBinsTheta()
        unsigned int getNBinsPhi()

cdef extern from "CubaticOrderParameter.h" namespace "freud::order":
    cdef cppclass CubaticOrderParameter:
        CubaticOrderParameter(float, float, float, float*, unsigned int, unsigned int)
        void resetCubaticOrderParameter()
        void compute(quat[float]*,
                     unsigned int,
                     unsigned int) nogil except +
        void reduceCubaticOrderParameter()
        unsigned int getNumParticles()
        float getCubaticOrderParameter()
        shared_ptr[float] getParticleCubaticOrderParameter()
        shared_ptr[float] getParticleTensor()
        shared_ptr[float] getGlobalTensor()
        shared_ptr[float] getCubaticTensor()
        shared_ptr[float] getGenR4Tensor()
        float getTInitial()
        float getTFinal()
        float getScale()
        quat[float] getCubaticOrientation()

cdef extern from "HexOrderParameter.h" namespace "freud::order":
    cdef cppclass HexOrderParameter:
        HexOrderParameter(float, float, unsigned int)
        const box.Box &getBox() const
        void compute(box.Box &,
                     const vec3[float]*,
                     unsigned int) nogil except +
        # unsure how to pass back the std::complex, but this seems to compile...
        shared_array[float complex] getPsi()
        unsigned int getNP()
        float getK()

cdef extern from "LocalDescriptors.h" namespace "freud::order":
    ctypedef enum LocalDescriptorOrientation:
        LocalNeighborhood
        Global
        ParticleLocal

    cdef cppclass LocalDescriptors:
        LocalDescriptors(unsigned int,
                         unsigned int,
                         float,
                         bool)
        unsigned int getNNeigh() const
        unsigned int getLMax() const
        unsigned int getSphWidth() const
        float getRMax() const
        unsigned int getNP()
        void computeNList(const box.Box&, const vec3[float]*, unsigned int,
                          const vec3[float]*, unsigned int) nogil except +
        void compute(const box.Box&, unsigned int, const vec3[float]*,
                     unsigned int, const vec3[float]*, unsigned int,
                     const quat[float]*, LocalDescriptorOrientation) nogil except +
        shared_array[float complex] getSph()

cdef extern from "TransOrderParameter.h" namespace "freud::order":
    cdef cppclass TransOrderParameter:
        TransOrderParameter(float, float, unsigned int)
        const box.Box &getBox() const,
        void compute(box.Box &,
                     const vec3[float]*,
                     unsigned int) nogil except +
        shared_array[float complex] getDr()
        unsigned int getNP()

cdef extern from "LocalQl.h" namespace "freud::order":
    cdef cppclass LocalQl:
        LocalQl(const box.Box&, float, unsigned int, float)
        const box.Box& getBox() const
        void setBox(const box.Box)
        void compute(const vec3[float]*,
                     unsigned int) nogil except +
        void computeAve(const vec3[float]*,
                        unsigned int) nogil except +
        void computeNorm(const vec3[float]*,
                         unsigned int) nogil except +
        void computeAveNorm(const vec3[float]*,
                            unsigned int) nogil except +
        shared_array[float] getQl()
        shared_array[float] getAveQl()
        shared_array[float] getQlNorm()
        shared_array[float] getQlAveNorm()
        unsigned int getNP()


cdef extern from "LocalQlNear.h" namespace "freud::order":
    cdef cppclass LocalQlNear:
        LocalQlNear(const box.Box&, float, unsigned int, unsigned int)
        const box.Box& getBox() const
        void setBox(const box.Box)
        void compute(const vec3[float]*,
                     unsigned int) nogil except +
        void computeAve(const vec3[float]*,
                        unsigned int) nogil except +
        void computeNorm(const vec3[float]*,
                         unsigned int) nogil except +
        void computeAveNorm(const vec3[float]*,
                            unsigned int) nogil except +
        shared_array[float] getQl()
        shared_array[float] getAveQl()
        shared_array[float] getQlNorm()
        shared_array[float] getQlAveNorm()
        unsigned int getNP()

cdef extern from "LocalWl.h" namespace "freud::order":
    cdef cppclass LocalWl:
        LocalWl(const box.Box&, float, unsigned int)
        const box.Box& getBox() const
        void setBox(const box.Box)
        void compute(const vec3[float]*,
                     unsigned int) nogil except +
        void computeAve(const vec3[float]*,
                        unsigned int) nogil except +
        void computeNorm(const vec3[float]*,
                         unsigned int) nogil except +
        void computeAveNorm(const vec3[float]*,
                            unsigned int) nogil except +
        shared_array[float] getQl()
        shared_array[float complex] getWl()
        shared_array[float complex] getAveWl()
        shared_array[float complex] getWlNorm()
        shared_array[float complex] getAveNormWl()
        void enableNormalization()
        void disableNormalization()
        unsigned int getNP()

cdef extern from "LocalWlNear.h" namespace "freud::order":
    cdef cppclass LocalWlNear:
        LocalWlNear(const box.Box&, float, unsigned int, unsigned int)
        const box.Box& getBox() const
        void setBox(const box.Box)
        void compute(const vec3[float]*,
                     unsigned int) nogil except +
        void computeAve(const vec3[float]*,
                        unsigned int) nogil except +
        void computeNorm(const vec3[float]*,
                         unsigned int) nogil except +
        void computeAveNorm(const vec3[float]*,
                            unsigned int) nogil except +
        shared_array[float] getQl()
        shared_array[float complex] getWl()
        shared_array[float complex] getWlNorm()
        shared_array[float complex] getAveWl()
        shared_array[float complex] getWlAveNorm()
        void enableNormalization()
        void disableNormalization()
        unsigned int getNP()

cdef extern from "SolLiq.h" namespace "freud::order":
    cdef cppclass SolLiq:
        SolLiq(const box.Box&, float, float, unsigned int, unsigned int)
        const box.Box& getBox() const
        void setBox(const box.Box)
        void setClusteringRadius(float)
        void compute(const vec3[float]*,
                     unsigned int) nogil except +
        void computeSolLiqVariant(const vec3[float]*,
                                  unsigned int) nogil except +
        void computeSolLiqNoNorm(const vec3[float]*,
                                 unsigned int) nogil except +
        unsigned int getLargestClusterSize()
        vector[unsigned int] getClusterSizes()
        shared_array[float complex] getQlmi()
        shared_array[unsigned int] getClusters()
        shared_array[unsigned int] getNumberOfConnections()
        vector[float complex] getQldot_ij()
        unsigned int getNP()
        unsigned int getNumClusters()

cdef extern from "MatchEnv.h" namespace "freud::order":
    cdef cppclass MatchEnv:
        MatchEnv(const box.Box&, float, unsigned int) nogil except +
        void setBox(const box.Box)
        void cluster(const vec3[float]*,
                     unsigned int,
                     float,
                     bool,
                     bool,
                     bool) nogil except +
        void matchMotif(const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int,
                        float,
                        bool) nogil except +
        vector[float] minRMSDMotif(const vec3[float]*,
                        unsigned int,
                        const vec3[float]*,
                        unsigned int,
                        bool) nogil except +
        map[unsigned int, unsigned int] isSimilar(const vec3[float]*,
                                        vec3[float]*,
                                        unsigned int,
                                        float,
                                        bool) nogil except +
        map[unsigned int, unsigned int] minimizeRMSD(const vec3[float]*,
                                        vec3[float]*,
                                        unsigned int,
                                        float&,
                                        bool) nogil except +
        shared_array[unsigned int] getClusters()
        shared_array[vec3[float]] getEnvironment(unsigned int)
        shared_array[vec3[float]] getTotEnvironment()
        unsigned int getNP()
        unsigned int getNumClusters()
        unsigned int getNumNeighbors()
        unsigned int getMaxNumNeighbors()

cdef extern from "SolLiqNear.h" namespace "freud::order":
    cdef cppclass SolLiqNear:
        SolLiqNear(const box.Box&, float, float, unsigned int, unsigned int, unsigned int)
        const box.Box& getBox() const
        void setBox(const box.Box)
        void setClusteringRadius(float)
        void compute(const vec3[float]*,
                     unsigned int) nogil except +
        void computeSolLiqVariant(const vec3[float]*,
                                  unsigned int) nogil except +
        void computeSolLiqNoNorm(const vec3[float]*,
                                 unsigned int) nogil except +
        unsigned int getLargestClusterSize()
        vector[unsigned int] getClusterSizes()
        shared_array[float complex] getQlmi()
        shared_array[unsigned int] getClusters()
        shared_array[unsigned int] getNumberOfConnections()
        vector[float complex] getQldot_ij()
        unsigned int getNumClusters()
        unsigned int getNP()

cdef extern from "Pairing2D.h" namespace "freud::order":
    cdef cppclass Pairing2D:
        Pairing2D(const float, const unsigned int, float)
        const box.Box &getBox() const
        void resetBondOrder()
        void compute(box.Box &,
                     vec3[float]*,
                     float*,
                     float*,
                     unsigned int,
                     unsigned int) nogil except +
        shared_array[unsigned int] getMatch()
        shared_array[unsigned int] getPair()
        unsigned int getNumParticles()
