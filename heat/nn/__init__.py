import sys
#<<<<<<< HEAD
#import unittest
#
#from .data_parallel import *
#import torch
#
#if sys.version_info.minor >= 7:
#    from . import functional
#
#    functional.__getattr__ = functional.func_getattr
#
#    def __getattr__(name):
#        torch_all = torch.nn.modules.__all__
#        if name in torch_all:
#            return torch.nn.__getattribute__(name)
#        else:
#            try:
#                unittest.__getattribute__(name)
#            except AttributeError:
#                raise AttributeError(f"module {name} not implemented in Torch or Heat")
#
#
#else:
#    # class Wrap(object):
#    #     def __init__(self, wrapped):
#    #         self.wrapped = wrapped
#    #
#    #     def __getattr__(self, item):
#    #         try:
#    #
#    # wrap functional
#    from . import functional
#    import sys
#
#    if sys.version_info.minor < 7:
#
#        class Wrap(object):
#            def __init__(self, wrapped):
#                self.wrapped = wrapped
#
#            def __getattr__(self, item):
#                try:
#                    getattr(self.wrapped, item)
#                except AttributeError:
#                    try:
#                        getattr(torch.nn.functional, item)
#                    except AttributeError:
#                        raise AttributeError("Module not in heat.nn or torch.nn")
#
#        functional = Wrap(functional)
#    pass
#=======
import torch
import unittest

from . import functional


if sys.version_info.minor >= 7:
    from .data_parallel import *

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
    from . import data_parallel
    from . import tests

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.torch_all = torch.nn.modules.__all__
            self.data_parallel_all = data_parallel.__all__

        def __getattr__(self, name):
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
#>>>>>>> c7ea00081aadee7d43efd4e8062fab5e7f27e99f
