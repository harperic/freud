How to cite Freud
==================

If you use Freud in your work, please acknowledge by using the below formulation and citation(s):

*Data analysis for this publication utilized methods provided by the Freud Analysis Suite[1].*

  [1] Eric S. Harper, Matthew Spellings, Joshua Anderson, and Sharon C. Glotzer. *harperic/freud: v0.6.1*. Nov. 2016. DOI:10.5281/zenodo.166564. URL: https://doi.org/10.5281/zenodo.166564.

.. tip::

    You can auto-generate the first formatted reference and the corresponding BibTeX file via :py:func:`freud.cite.reference` and :py:func:`freud.cite.bibtex`.


References and Citations
========================

Parts of Freud use algorithms, libraries, etc. from the following:

.. [Cit0] Bokeh Development Team (2014). Bokeh: Python library for interactive visualization
          URL http://www.bokeh.pydata.org.

.. [Cit1] Haji-Akbari, A. ; Glotzer, S. C. Strong Orientational Coordinates and Orientational
          Order Parameters for Symmetric Objects. Journal of Physics A: Mathematical and Theoretical 2015, 48, 485201.

.. [Cit2] van Anders, G. ; Ahmed, N. K. ; Klotsa, D. ; Engel, M. ; Glotzer, S. C. Unified Theoretical Framework for
          Shape Entropy in Colloids", arXiv:1309.1187.

.. [Cit3] van Anders, G. ; Ahmed, N. K. ; Smith, R. ; Engel, M. ; Glotzer, S. C. Entropically Patchy Particles,
          arXiv:1304.7545.

.. [Cit4] Wolfgan Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)

.. [Cit5] Eigen: a c++ linear algebra library.
          URL http://eigen.tuxfamily.org

Cite Module
===========

Generate citations for Freud

.. autoclass:: freud.cite.bibtex(*args, **kwargs)
    :members:

.. autoclass:: freud.cite.reference(*args, **kwargs)
    :members:
