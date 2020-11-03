from .lr_scheduler import *

import sys
import torch
import unittest


if sys.version_info.minor >= 7:
    from .dp_optimizer import *

    def __getattr__(name):
        # this will call the Heat optimizers if available,
        # otherwise, it falls back to call a torch optimizer
        if name in dp_optimizer.__all__:
            return dp_optimizer.__getattribute__(name)

        try:
            return torch.optim.__getattribute__(name)
        except AttributeError:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                if name is not None:
                    raise AttributeError(f"module {name} not implemented in torch.optim")


else:
    from . import dp_optimizer

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            # this will call the Heat optimizers if available,
            # otherwise, it falls back to call a torch optimizer
            if name in dp_optimizer.__all__:
                return dp_optimizer.__getattribute__(name)

            try:
                return torch.optim.__getattribute__(name)
            except AttributeError:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    if name is not None:
                        raise AttributeError(f"module {name} not implemented in torch.optim")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
