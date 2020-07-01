import torch

__all__ = []


def __getattr__(name):
    if name in __all__:
        return name
    else:
        return torch.nn.__getattribute__(name)
