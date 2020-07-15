import torch

__all__ = ["DataParallelOptimizer"]


class DataParallelOptimizer:
    """

    blocking : bool (optional)
        Flag for blocking synchronization. If not given, synchronization is blocking by default.
    """

    def __init__(self, torch_optimizer, blocking=True):
        self.torch_optimizer = torch_optimizer
        self.blocking = blocking

    def step(self):
        if self.blocking:
            self.torch_optimizer.step()
        else:
            # todo: this is the nonblocking case, @lehr_fa does anything need to happen here?
            pass
