import sys
import unittest

from .data_parallel import *
import torch

if sys.version_info.minor >= 7:
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


else:
    # class Wrap(object):
    #     def __init__(self, wrapped):
    #         self.wrapped = wrapped
    #
    #     def __getattr__(self, item):
    #         try:
    #
    # wrap functional
    from . import functional
    import sys

    if sys.version_info.minor < 7:

        class Wrap(object):
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, item):
                try:
                    getattr(self.wrapped, item)
                except AttributeError:
                    try:
                        getattr(torch.nn.functional, item)
                    except AttributeError:
                        raise AttributeError("Module not in heat.nn or torch.nn")

        functional = Wrap(functional)
    pass
