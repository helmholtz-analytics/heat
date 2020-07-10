import torch


def func_getattr(name):
    try:
        return torch.nn.functional.__getattribute__(name)
    except AttributeError:
        print("functional getattr", name)
        if name is not None:
            raise NotImplementedError("module not implemented in Torch Functional")
