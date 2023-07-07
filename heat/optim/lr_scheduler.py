"""
Learning rate schedulers in the heat namespace
"""

import sys
import torch.optim.lr_scheduler as lrs


def __getattr__(name):
    """
    Call the torch learning rate scheduler of a specified name
    """
    try:
        return lrs.__getattribute__(name)
    except AttributeError:
        raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")
