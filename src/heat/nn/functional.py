"""
File containing the heat.nn.functional submodule
"""

import sys
import torch


def func_getattr(name):
    """
    When a function is called for the heat.nn.functional module it will attempt to run the
    heat.nn.functional module with that name, then, if there is no such heat nn module,
    it will attempt to get the torch.nn.functional module of that name.
    """
    # call through to the torch.nn.functional module
    try:
        return torch.nn.functional.__getattribute__(name)
    except AttributeError:
        if name is not None:
            raise AttributeError("module not implemented in Torch Functional")
