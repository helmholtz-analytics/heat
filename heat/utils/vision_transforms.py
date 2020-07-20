# import base64
# import torch
import torchvision.transforms

# __all__ = ["ImagenetStringToTensorReshape"]


def __getattr__(name):
    if name in torchvision.transforms.transforms.__all__:
        return torchvision.transforms.__getattribute__(name)
    else:
        raise AttributeError(f"module {name} not implemented in Torch or Heat")
