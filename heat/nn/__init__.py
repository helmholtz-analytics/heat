import torch
import unittest

from .data_parallel import *
from . import functional

import sys

if sys.version_info.minor >= 7:
    functional.__getattr__ = functional.func_getattr

    # todo: skip this for functions in unittest
    def __getattr__(name):
        torch_all = torch.nn.modules.__all__
        if name in torch_all:
            return torch.nn.__getattribute__(name)
        else:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                raise AttributeError(f"module {name} not implemented in Torch or Heat")


else:
    import torch.nn.functional as tnn

    torch_all = tnn.__dict__
    for n in torch_all.keys():
        setattr(functional, n, torch_all[n])
