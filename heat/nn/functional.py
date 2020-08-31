import torch


def func_getattr(name):
    # call through to the torch.nn.functional module
    try:
        return torch.nn.functional.__getattribute__(name)
    except AttributeError:
        if name is not None:
            raise AttributeError("module not implemented in Torch Functional")
