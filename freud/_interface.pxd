# Copyright (c) 2010-2017 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from freud.util._VectorMath cimport vec3
from freud.util._Boost cimport shared_array
cimport freud._box as box

cdef extern from "InterfaceMeasure.h" namespace "freud::interface":
    cdef cppclass InterfaceMeasure:
        InterfaceMeasure(const box.Box&, float)
        unsigned int compute(const vec3[float]*, unsigned int, const vec3[float]*, unsigned int)
