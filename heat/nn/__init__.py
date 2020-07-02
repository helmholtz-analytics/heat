from .data_parallel import *

import torch

torch_all = torch.nn.modules.__all__


def __getattr__(name):
    if name not in torch_all:
        return name
    else:
        return torch.nn.__getattribute__(name)
