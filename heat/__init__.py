from .core import *
from .core.linalg import *
from .core import __version__

from . import core
from . import classification
from . import cluster
from . import graph
from . import naive_bayes
from . import nn
from . import optim
from . import regression
from . import spatial
from . import utils

import sys

if sys.version_info.minor < 7:
    import torch.nn as tnn

    torch_all = tnn.modules.__dict__
    for n in torch_all.keys():
        setattr(nn, n, torch_all[n])

    import torch.optim as tnn

    torch_all = tnn.__dict__
    for n in torch_all.keys():
        setattr(optim, n, torch_all[n])
