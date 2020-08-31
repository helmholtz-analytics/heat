import torch

__all__ = ["DataParallelOptimizer"]


class DataParallelOptimizer:
    """
    Uses a Torch.optim.Optimizer for data parallelism. It should be used in combination with DataParallel (DP) class.
    To optimize a DP module, DP optimizer has to be passed to DP module during its initialization.
    See :func:`..nn.DataParallel` for a basic example of usage.

    Attributes
    ----------
    torch_optimizer : torch.optim.Optimizer
        the wrapped Torch optimizer
    blocking : bool
        use blocking communications or not. will typically be overwritten by heat.nn.DataParallel
    """

    def __init__(self, torch_optimizer: torch.optim.Optimizer, blocking: bool = False):
        self.torch_optimizer = torch_optimizer
        if not isinstance(blocking, bool):
            raise TypeError(f"blocking parameter must be a boolean, currently {type(blocking)}")
        # flag indicating if communication during parameter updates is blocking.
        self.blocking_parameter_updates = blocking
        # flag indicating if optimizer should take a step during next iteration (only relevant for non-blocking)
        self.update_next = False
        # reference of optimizer's params
        self.params_ref = torch_optimizer.param_groups[0]["params"]

    def step(self) -> None:
        """
        Force torch optimizer to update model parameters. For blocking, optimizer immediately updates parameters. For
        non-blocking, optimizer will update parameters during next forward.
        """
        if self.blocking_parameter_updates:
            self.torch_optimizer.step()
        else:
            self.update_next = True

    def zero_grad(self) -> None:
        """
        Reset gradients of optimizer's params.
        """
        # reset view onto params in order to reset all gradients
        self.torch_optimizer.param_groups[0]["params"] = self.params_ref[:]
        self.torch_optimizer.zero_grad()
