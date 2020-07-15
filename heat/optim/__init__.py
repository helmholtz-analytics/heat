from .lr_scheduler import *
from . import dp_optimizer
import torch
import unittest


def __getattr__(name):
    if name not in dp_optimizer.__all__:
        try:
            return torch.optim.__getattribute__(name)
        except AttributeError:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                if name is not None:
                    raise AttributeError(f"module {name} not implemented in torch.optim")
    else:
        object.__getattribute__(name)
