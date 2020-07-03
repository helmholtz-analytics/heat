import torch
import torchvision.transforms

torch_all = torchvision.transforms.transforms.__all__


def __getattr__(name):
    if name in torch_all:
        return torch.nn.__getattribute__(name)
    else:
        raise NotImplementedError("module not implemented in Torch or Heat")
