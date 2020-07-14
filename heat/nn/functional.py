import torch


def func_getattr(name):
    try:
        return torch.nn.functional.__getattribute__(name)
    except AttributeError:
        if name is not None:
            raise NotImplementedError("module not implemented in Torch Functional")
