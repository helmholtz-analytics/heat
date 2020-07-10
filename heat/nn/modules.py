import torch

torch_all = torch.nn.modules.__all__


def __getattr__(name):
    if name in torch_all:
        return torch.nn.__getattribute__(name)
    else:
        print(name)
        raise NotImplementedError("module not implemented in Torch or Heat")
