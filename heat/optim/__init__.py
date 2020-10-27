from . import dp_optimizer
import torch
import unittest
import sys

if sys.version_info.minor >= 7:
    from .lr_scheduler import *

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
    # todo lr scheduler
    from . import lr_scheduler
    import torch.optim as tnn

    torch_all = tnn.__dict__
    for n in torch_all.keys():
        setattr(lr_scheduler, n, torch_all[n])
