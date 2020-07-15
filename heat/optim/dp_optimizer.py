import torch

__all__ = ["DataParallelOptimizer"]


class DataParallelOptimizer:
    """

    blocking : bool (optional)
        Flag for blocking synchronization. If not given, synchronization is blocking by default.
    """

    def __init__(self, torch_optimizer):
        self.torch_optimizer = torch_optimizer

        # flag indicating if communication during parameter updates is blocking. Set by the DataParallel entity this is
        # assigned to
        self.blocking_parameter_updates = None

        # flag indicating if optimizer should take a step during next iteration (only relevant for non-blocking)
        self.update_next = False

    def step(self):
        """
        Force torch optimizer to update model parameters. For blocking, optimizer immediately updates parameters. For
        non-blocking, optimizer will update parameters during next forward.
        """

        # abort, if this is not assigned to a DataParallel entity and therefore has no blocking flag set
        if self.blocking_parameter_updates is None:
            raise TypeError(
                "Attribute 'blocking_parameter_updates' must be set. Assign this to a valid entity of "
                "ht.nn.DataParallel to do so."
            )

        if self.blocking_parameter_updates:
            self.torch_optimizer.step()
        else:
            self.update_next = True
