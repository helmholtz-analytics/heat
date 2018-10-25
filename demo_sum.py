import heat as ht
import torch
import numpy as np
from copy import copy as _copy
from heat.core.tensor import tensor
from heat.core.communicator import mpi, MPICommunicator, NoneCommunicator
from heat.core import types



a = ht.ones(10, split=0).astype(ht.int)
b = a.sum(axis=0)
print('a: ', a.split, b.split)



