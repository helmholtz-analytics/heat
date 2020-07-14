from .lr_scheduler import *
import torch
import unittest


def __getattr__(name):
    try:
        return torch.optim.__getattribute__(name)
    except AttributeError:
        try:
            unittest.__getattribute__(name)
        except AttributeError:
            if name is not None:
                raise AttributeError(f"module {name} not implemented in torch.optim")
