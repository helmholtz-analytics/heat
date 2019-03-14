import itertools
import torch

from .communication import MPI
from . import stride_tricks
from . import types
from . import tensor
from .operations import __local_operation as local_op
from .operations import __reduce_op as reduce_op

# Definition of HeAT Constants

INF = float('inf')
NAN = float('nan')
NINF = - float('inf')
PI = 3.141592653589793
E = 2.718281828459045

#Aliases
inf = Inf = Infty = Infinity = INF
nan = NaN = NAN
pi = PI
e = Euler = E


__all__ = [

]
