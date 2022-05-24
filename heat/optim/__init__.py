"""
This is the heat.optimizer submodule.

It contains data parallel specific optimizers and learning rate schedulers. It also includes all of the
optimizers and learning rate schedulers in the torch namespace
"""

from .lr_scheduler import *
from . import utils

import sys
import torch
import unittest


if sys.version_info.minor >= 7:
    from .dp_optimizer import *

    def __getattr__(name):
        """
        When a function is called for the heat.optim module it will attempt to run the heat optimizer with that
        name, then, if there is no such heat optimizer, it will attempt to get the torch optimizer of that name.
        """
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
    from . import tests

    class _Wrapper36(object):
        """
        Wrapper to handle the dynamic calling of torch.optim modules in the heat namespace
        """

        def __init__(self, wrapped):  # noqa: D107
            self.wrapped = wrapped

        def __getattr__(self, name):
            """
            When a function is called for the heat.optim module it will attempt to run the heat optimizer with that
            name, then, if there is no such heat optimizer, it will attempt to get the torch optimizer of that name.
            """
            # this will call the Heat optimizers if available,
            # otherwise, it falls back to call a torch optimizer
            if name in dp_optimizer.__all__:
                return dp_optimizer.__getattribute__(name)
            elif name == "tests":
                return tests

            try:
                return torch.optim.__getattribute__(name)
            except AttributeError:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    if name is not None:
                        raise AttributeError(f"module '{name}' not implemented in torch or heat")

    sys.modules[__name__] = _Wrapper36(sys.modules[__name__])
