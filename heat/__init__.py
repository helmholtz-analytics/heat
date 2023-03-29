"""
add modules/namespaces to the heat namespace
"""

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
from . import sparse
from . import spatial
from . import utils
