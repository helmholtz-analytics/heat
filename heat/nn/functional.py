import sys
import torch

if sys.version_info.minor >= 7:

    def func_getattr(name):
        # call through to the torch.nn.functional module
        try:
            return torch.nn.functional.__getattribute__(name)
        except AttributeError:
            if name is not None:
                raise AttributeError("module not implemented in Torch Functional")


else:

    def fn(name):
        def name():
            return torch.nn.__getattr__(name)

    torch_names = torch.nn.functional.__dict__.keys()
    torch_names = [t for t in torch_names if t[0] != "_"]
    for name in torch_names:
        fn(name)
