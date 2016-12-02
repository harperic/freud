.. _kspace:

==============
KSpace Module
==============

Modules for calculating quantities in reciprocal space, including fourier transforms of shapes and diffraction pattern
generation.

Meshgrid
========

.. autoclass:: freud.kspace.meshgrid2(*arrs)
    :members:


Structure Factor
================

Methods for calculating the structure factor of different systems

.. autoclass:: freud.kspace.SFactor3DPoints(box, g)
    :members:

.. autoclass:: freud.kspace.AnalyzeSFactor3D(S)
    :members:

.. autoclass:: freud.kspace.SingleCell3D(k, ndiv, dK, boxMatrix)
    :members:

.. autoclass:: freud.kspace.FTfactory()
    :members:

.. autoclass:: freud.kspace.FTbase()
    :members:

.. autoclass:: freud.kspace.FTdelta()
    :members:

.. autoclass:: freud.kspace.FTsphere()
    :members:

.. autoclass:: freud.kspace.FTpolyhedron()
    :members:

.. autoclass:: freud.kspace.FTconvexPolyhedron()
    :members:

Diffraction Patterns
====================

Methods for calculating diffraction patterns of various systems

.. autoclass:: freud.kspace.DeltaSpot()
    :members:

.. autoclass:: freud.kspace.GaussianSpot()
    :members:

Utilities
=========

Classes and methods used by other kspace modules

.. autoclass:: freud.kspace.Constraint()
    :members:

.. autoclass:: freud.kspace.AlignedBoxConstraint()
    :members:

.. autoclass:: freud.kspace.constrainedLatticePoints()
    :members:

.. autoclass:: freud.kspace.reciprocalLattice3D()
    :members:
