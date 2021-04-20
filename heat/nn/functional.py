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


if sys.version_info.minor == 6:

    class _Wrapper36(object):
        """
        Wrapper to handle the dynamic calling of torch.nn.functional modules in the heat namespace
        """

        def __init__(self, wrapped):  # noqa: D107
            self.wrapped = wrapped

        def __getattr__(self, name):
            """
            When a function is called for the heat.nn.functional module it will attempt to run the
            heat.nn.functional module with that name, then, if there is no such heat nn module,
            it will attempt to get the torch.nn.functional module of that name.
            """
            try:
                return torch.nn.functional.__getattribute__(name)
            except AttributeError:
                if name is not None:
                    raise AttributeError("module not implemented in torch.nn.functional")

    sys.modules[__name__] = _Wrapper36(sys.modules[__name__])
