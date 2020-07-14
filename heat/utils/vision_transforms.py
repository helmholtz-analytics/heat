import torchvision.transforms

torch_all = torchvision.transforms.transforms.__all__


def __getattr__(name):
    if name in torch_all:
        return torchvision.transforms.__getattribute__(name)
    else:
        raise AttributeError(f"module {name} not implemented in Torch or Heat")
