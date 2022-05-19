"""
This is the heat.nn submodule.

It contains data parallel specific nn modules. It also includes all of the modules in the torch.nn namespace
"""

import sys
import torch
import unittest

from . import functional


if sys.version_info.minor >= 7:
    from .data_parallel import *

    functional.__getattr__ = functional.func_getattr

    def __getattr__(name):
        """
        When a function is called for the heat.nn module it will attempt to run the heat nn module with that
        name, then, if there is no such heat nn module, it will attempt to get the torch nn module of that name.
        """
        torch_all = torch.nn.modules.__all__
        if name in torch_all:
            return torch.nn.__getattribute__(name)
        else:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                raise AttributeError(f"module {name} not implemented in Torch or Heat")

else:
    from . import data_parallel
    from . import tests

    class Wrapper(object):
        """
        Wrapper to handle the dynamic calling of torch.nn modules in the heat namespace
        """

        def __init__(self, wrapped):  # noqa: D107
            self.wrapped = wrapped
            self.torch_all = torch.nn.modules.__all__
            self.data_parallel_all = data_parallel.__all__

        def __getattr__(self, name):
            """
            When a function is called for the heat.nn module it will attempt to run the heat nn module with that
            name, then, if there is no such heat nn module, it will attempt to get the torch nn module of that name.
            """
            if name in self.torch_all:
                return torch.nn.__getattribute__(name)
            elif name == "functional":
                return functional
            elif name in self.data_parallel_all:
                return data_parallel.__getattribute__(name)
            elif name == "tests":
                return tests
            else:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    raise AttributeError(f"module '{name}' not implemented in Torch or Heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
