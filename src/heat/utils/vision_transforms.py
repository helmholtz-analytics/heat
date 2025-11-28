"""
File with the available transforms for images
"""

import torchvision.transforms
import sys
import unittest


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
