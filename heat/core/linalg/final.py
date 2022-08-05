"""
SVD of input DNDarray.
"""
import itertools
from operator import imod
from turtle import left
import torch
import math
import warnings
import heat as ht

from mpi4py import MPI

from utils import (
    apply_house,
    apply_house_left,
    apply_house_right,
    gen_house_mat,
    gen_house_vec,
)
from bcg import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from typing import List, Callable, Union, Optional, Tuple, TypeVar

from torch._C import Value
from inspect import stack
from pathlib import Path

from heat.core.devices import cpu

from heat.core import communication
from heat.core import arithmetics
from heat.core import complex_math
from heat.core import constants
from heat.core import exponential
from heat.core import dndarray
from heat.core import factories
from heat.core import manipulations
from heat.core.manipulations import *
from heat.core import rounding
from heat.core import sanitation
from heat.core import statistics
from heat.core import stride_tricks
from heat.core import types
from svd import *
from qr import *
