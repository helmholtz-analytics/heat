"""
Learning rate schedulers in the heat namespace
"""

import sys
import torch.optim.lr_scheduler as lrs

if sys.version_info.minor >= 7:

    def __getattr__(name):
        """
        Call the torch learning rate scheduler of a specified name
        """
        try:
            return lrs.__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")


else:

    class _Wrapper36(object):
        """
        Class to add the torch learning rate schedulers to the heat.optim namespace
        """

        def __init__(self, wrapped):  # noqa: D107
            self.wrapped = wrapped

        def __getattr__(self, name):
            """
            Call the torch learning rate scheduler of a specified name
            """
            try:
                return lrs.__getattribute__(name)
            except AttributeError:
                raise AttributeError(f"name {name} is not implemented in torch.optim.lr_scheduler")

    sys.modules[__name__] = _Wrapper36(sys.modules[__name__])
