import torch.optim.lr_scheduler as lr_scheduler
import sys


if sys.version_info.minor >= 7:

    def __getattr__(name):
        try:
            return lr_scheduler.__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")


else:

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            try:
                return lr_scheduler.__getattribute__(name)
            except AttributeError:
                raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
