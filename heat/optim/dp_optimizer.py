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

        # flag indicating if optimizer should take a step during next iteration (only relevant for non-blocking)
        self.update_next = False

    def step(self):
        """
        Force torch optimizer to update model parameters. For blocking, optimizer immediately updates parameters. For
        non-blocking, optimizer will update parameters during next forward.
        """

        if self.blocking:
            self.torch_optimizer.step()
        else:
            self.update_next = True
