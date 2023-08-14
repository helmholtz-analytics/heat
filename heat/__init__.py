"""
add modules/namespaces to the heat namespace
"""

from . import (
    classification,
    cluster,
    core,
    graph,
    naive_bayes,
    nn,
    optim,
    regression,
    sparse,
    spatial,
    utils,
)
from .core import *
from .core import __version__
from .core.linalg import *
