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
    import torch

    class Wrap(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, item):
            try:
                getattr(self.wrapped, item)
            except AttributeError:
                try:
                    getattr(torch.nn, item)
                except AttributeError:
                    raise AttributeError("Module not in heat.nn or torch.nn")

    nn = Wrap(nn)
