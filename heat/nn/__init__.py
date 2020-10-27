import torch
import unittest

from .data_parallel import *
from . import functional

functional.__getattr__ = functional.func_getattr


def __getattr__(name):
    torch_all = torch.nn.modules.__all__
    if name in torch_all:
        return torch.nn.__getattribute__(name)
    else:
        try:
            unittest.__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"module {name} not implemented in Torch or Heat")
