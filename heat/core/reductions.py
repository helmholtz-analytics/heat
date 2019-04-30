import torch
import numpy as np

from .communication import MPI
from . import types
from . import tensor
from . import exponential
from .operations import __local_operation as local_op
from .operations import __reduce_op as reduce_op


__all__ = [
    'mean',
    'std',
    'sum',
    'var'
]



