"""
File with the available transforms for images
"""

import torchvision.transforms
import sys
import unittest


if sys.version_info.minor >= 7:

    def __getattr__(name):
        """
        When a function is called for the heat.vision_transforms module it will attempt to run the
        heat function/class with that name, then, if there is no such object, it will attempt to
        get the torchvision.transforms object of that name.
        """
        if name in torchvision.transforms.transforms.__all__:
            return torchvision.transforms.__getattribute__(name)
        else:
            raise AttributeError(f"module {name} not implemented in Torch or Heat")


else:

    class Wrapper(object):
        """
        Call through to torchvision.transforms
        """

        def __init__(self, wrapped):  # noqa: D107
            self.wrapped = wrapped

        def __getattr__(self, name):
            """
            When a function is called for the heat.vision_transforms module it will attempt to run the
            heat function/class with that name, then, if there is no such object, it will attempt to
            get the torchvision.transforms object of that name.
            """
            torch_all = torchvision.transforms.transforms.__all__
            if name in torch_all:
                return torchvision.transforms.__getattribute__(name)
            else:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    raise AttributeError(f"module {name} not implemented in Torch or Heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
