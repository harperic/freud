# Copyright (c) 2010-2017 The Regents of the University of Michigan
# This file is part of the Freud project, released under the BSD 3-Clause License.

"""Functions to support citing this software."""
import sys


# ARXIV_BIBTEX = """@online{Freud,
#     author      = {Eric S. Harper and Matthew P. Spellings and Joshua A. Anderson and Sharon C. Glotzer},
#     title       = {Freud: A Software Suite for High-Throughput Simulation Analysis},
#     year        = {2017},
#     eprinttype  = {arxiv},
#     eprintclass = {XXXX},
#     eprint      = {XXXX}
# }
# """

# I should just use some kind of api...
ZENODO_BIBTEX = """@misc{FreudSoftware,
  author       = {Eric S. Harper and Matthew Spellings and Joshua Anderson and Sharon C. Glotzer},
  title        = {harperic/freud: Zenodo DOI release},
  month        = nov,
  year         = 2016,
  doi          = {10.5281/zenodo.166564},
  url          = {https://doi.org/10.5281/zenodo.166564}
}
"""


# ARXIV_REFERENCE = "Eric S. Harper, Matthew P. Spellings, Joshua A. Anderson, and Sharon C. Glotzer. Freud: A Software Suite for High-Throughput Simulation Analysis. 2017. arXiv:XXXX [XXXX]"
ZENODO_REFERENCE = "Eric S. Harper, Matthew Spellings, Joshua Anderson, and Sharon C. Glotzer. *harperic/freud: v0.6.1*. Nov. 2016. DOI:10.5281/zenodo.166564. URL: https://doi.org/10.5281/zenodo.166564."


def bibtex(file=None):
    """Generate bibtex entries for Freud.

    The bibtex entries will be printed to screen unless a
    filename or a file-like object are provided, in which
    case they will be written to the corresponding file.

    .. note::

        A full reference should also include the
        version of this software. Please refer to the
        documentation on how to cite a specific version.

    :param file: A str or file-like object.
        Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout
    elif isinstance(file, str):
        file = open(file, 'w')
    file.write(ZENODO_BIBTEX)


def reference(file=None):
    """Generate formatted reference entries for signac.

    The references will be printed to screen unless a
    filename or a file-like object are provided, in which
    case they will be written to the corresponding file.

    .. note::

        A full reference should also include the
        version of this software. Please refer to the
        documentation on how to cite a specific version.

    :param file: A str or file-like object.
        Defaults to sys.stdout.
    """
    if file is None:
        file = sys.stdout
    elif isinstance(file, str):
        file = open(file, 'w')
    file.write(ZENODO_REFERENCE + '\n')
