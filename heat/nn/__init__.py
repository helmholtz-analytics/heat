import sys
import torch
import unittest

from . import functional

functional.__getattr__ = functional.func_getattr

if sys.version_info.minor >= 7:
    from .data_parallel import *

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
    from . import data_parallel

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            torch_all = torch.nn.modules.__all__
            data_parallel_all = data_parallel.__all__
            if name in torch_all:
                return torch.nn.__getattribute__(name)
            elif name == "functional":
                return functional
            elif name in data_parallel_all:
                return data_parallel.__getattribute__(name)
            else:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    raise AttributeError(f"module {name} not implemented in Torch or Heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
