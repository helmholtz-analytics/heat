"""
add data utility functions to the ht.utils.data namespace
"""

from . import matrixgallery

if matrixgallery.HAVE_MPI:
    from .datatools import *
    from . import mnist
    from .partial_dataset import *
    from . import spherical
