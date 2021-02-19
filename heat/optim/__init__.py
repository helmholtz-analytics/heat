#<<<<<<< HEAD
#import sys
#=======
from .lr_scheduler import *

import sys
import torch
import unittest
#>>>>>>> c7ea00081aadee7d43efd4e8062fab5e7f27e99f

if sys.version_info.minor >= 7:
    from .lr_scheduler import *
    from . import dp_optimizer
    import torch
    import unittest

#<<<<<<< HEAD
#=======
#if sys.version_info.minor >= 7:
#    from .dp_optimizer import *
#
#>>>>>>> c7ea00081aadee7d43efd4e8062fab5e7f27e99f
    def __getattr__(name):
        # this will call the Heat optimizers if available,
        # otherwise, it falls back to call a torch optimizer
        if name in dp_optimizer.__all__:
            return dp_optimizer.__getattribute__(name)

        try:
            return torch.optim.__getattribute__(name)
        except AttributeError:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                if name is not None:
                    raise AttributeError(f"module {name} not implemented in torch.optim")


else:
#<<<<<<< HEAD
#    # from . import lr_scheduler
#    from .dp_optimizer import *
#    import torch.optim as topt
#
#    Adadelta = topt.Adadelta
#    Adagrad = topt.Adagrad
#    Adam = topt.Adam
#    AdamW = topt.AdamW
#    Adamax = topt.Adamax
#    ASGD = topt.ASGD
#    LBFGS = topt.LBFGS
#    RMSprop = topt.RMSprop
#    Rprop = topt.Rprop
#    SGD = topt.SGD
#    SparseAdam = topt.SparseAdam
#    """
#    from . import lr_scheduler as lr_scheduler
#    from .optimizer import Optimizer as Optimizer
#
#    """
#=======
    from . import dp_optimizer
    from . import tests

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            # this will call the Heat optimizers if available,
            # otherwise, it falls back to call a torch optimizer
            if name in dp_optimizer.__all__:
                return dp_optimizer.__getattribute__(name)
            elif name == "tests":
                return tests

            try:
                return torch.optim.__getattribute__(name)
            except AttributeError:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    if name is not None:
                        raise AttributeError(f"module '{name}' not implemented in torch or heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])
#>>>>>>> c7ea00081aadee7d43efd4e8062fab5e7f27e99f
