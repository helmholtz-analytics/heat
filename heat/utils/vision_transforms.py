import torchvision.transforms
import sys
import unittest


if sys.version_info.minor >= 7:

    def __getattr__(name):
        if name in torchvision.transforms.transforms.__all__:
            return torchvision.transforms.__getattribute__(name)
        else:
            raise AttributeError(f"module {name} not implemented in Torch or Heat")


else:

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            torch_all = torchvision.transforms.transforms.__all__
            if name in torch_all:
                return torchvision.transforms.__getattribute__(name)
            else:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    raise AttributeError(f"module {name} not implemented in Torch or Heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
