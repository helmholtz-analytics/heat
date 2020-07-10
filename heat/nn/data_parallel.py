import bisect
import functools
import heat as ht
import operator
import torch
import torch.nn as tnn

from collections import OrderedDict
from typing import Callable, List

from heat.core.communication import MPI

__all__ = ["DataParallel"]


class DataParallel(tnn.Module):
    """
    Implements data parallelism across multiple processes. This means that the same model will be run locally
    on each process. Creation of the model parallels to PyTorch, the only changes are using HeAT layers (ht.nn.layer)
    in the initialization of the network. I.E. (example code contains)
    .. code-block:: python

        class TestingModel(torch.nn.Module):
            def __init__(self):
                super(TestingModel, self).__init__()
                self.net1 = ht.nn.Linear(10, 10)
                self.relu = ht.nn.ReLU()
                self.net2 = ht.nn.Linear(10, 5)

            def forward(self, x):
                return self.net2(self.relu(self.net1(x)))

        ht_model = ht.nn.DataParallel(TestingModel(), comm)

    and a requirement of giving a HeAT communicator (``comm``). For the given model both the ``__init__()`` and
    ``forward()`` functions must be defined in the class defining the network.

    It is highly recommended that a HeAT DataLoader is used, see :func:`..utils.data.datatools.DataLoader`. The
    default communications scheme for this is blocking. The blocking scheme will average the model parameters during
    the backwards step, synchronizing them before the next model iteration.

    To use the non-blocking communications for parameter updates, provide a torch optimizer as the optimizer flag.

    Attributes
    ----------
    module : torch.nn.Module
        the local module
    comm : heat.MPICommunicator
        HeAT communicator to use
    optimizer : torch.optim.Optimizer (optional)
        Optimizer used for parameter updates. If given, synchronization is non-blocking, else blocking.
    """

    def __init__(self, module: torch.nn.Module, comm: ht.MPICommunication, optimizer=None):
        super(DataParallel, self).__init__()
        self.module = module
        self.comm = comm
        self.optimizer = optimizer

        self.wait_handles = OrderedDict()
        self.fwd_hook_handles = list()
        # slices of parameters belonging to one and the same layer
        self.param_slices = dict()
        # pytorch internal parameter indexing
        self.param_indices = dict()
        # reference of optimizer's params
        self.params_ref = None

        # check if non-blocking
        if optimizer is not None:
            # check if optimizer matches module
            if list(module.parameters()) != optimizer.param_groups[0]["params"]:
                raise ValueError("given module and optimizer don't share same parameters.")
            else:
                # take reference of optimizer's params
                self.params_ref = optimizer.param_groups[0]["params"]
                optimizer.param_groups[0]["params"] = []

        # get parameter indexing and slices
        start_idx = 0
        layer_name_prev = None
        for idx, (name, param) in enumerate(module.named_parameters()):
            self.param_indices[name] = idx
            layer_name = name.split(sep=".", maxsplit=1)[0]
            if layer_name_prev is None:
                layer_name_prev = layer_name
            if layer_name_prev != layer_name:
                self.param_slices[layer_name_prev] = slice(start_idx, idx)
                layer_name_prev = layer_name
                start_idx = idx

            # register backward hooks for all model parameter tensors
            if optimizer is not None:
                param.register_hook(self.nonblocking_hook(layer_name, name))
            else:
                param.register_hook(self.blocking_hook)
        self.param_slices[layer_name_prev] = slice(start_idx, len(self.param_indices))

    def forward(self, *inputs: tuple, **kwargs: dict) -> torch.Tensor:
        data = inputs[0]

        if isinstance(data, ht.DNDarray):
            lcl_data = data._DNDarray__array
        elif isinstance(data, torch.Tensor):
            lcl_data = data
        else:
            lcl_data = torch.tensor(data)

        # check if non-blocking
        if self.optimizer is not None and self.module.training:
            # reset gradients before forward pass
            self.optimizer.zero_grad()
            # register forward hooks for all layers
            for name, submodule in self.module.named_modules():
                if name == "":
                    continue
                if name in self.wait_handles:
                    hook_handle = submodule.register_forward_pre_hook(self.forward_hook(name))
                    self.fwd_hook_handles.append(hook_handle)
        # perform forward pass
        ret = self.module(lcl_data, *inputs[1:], **kwargs)
        # clear dictionary after all wait handles are used up (dynamic computation graph)
        self.wait_handles.clear()
        # remove forward hooks (dynamic computation graph)
        for hook_handle in self.fwd_hook_handles:
            hook_handle.remove()
        self.fwd_hook_handles.clear()

        return ret

    def async_update(self, param_slice: slice = None, layer_names: List[str] = None):
        """
        Update parameters asynchronously via wait handles.

        Parameters
        ----------
        param_slice : slice (optional)
            Slice object for creating a view onto optimizer's params list. If None, the whole params list is taken.
        layer_names : list(str) (optional)
            List of layer names, whose parameters are to be updated. Has to match param_slice. If None, all layers are
            taken.
        """
        # perform update on the whole model
        if param_slice is None:
            param_slice = slice(len(self.params_ref))
        if layer_names is None:
            layer_names = list(self.wait_handles.keys())

        # update params that are visible for the optimizer
        self.optimizer.param_groups[0]["params"] = self.params_ref[param_slice]

        # iterate over layers
        for layer_name in layer_names:
            # iterate over layer's parameters/associated wait handles
            for _, param_name, wait_handle in self.wait_handles[layer_name]:
                # get internal index of selected parameter
                param_idx = self.param_indices[param_name]
                # synchronize, get parameter's global gradient
                wait_handle.wait()
                # check if shapes are matching
                if self.params_ref[param_idx].grad.data.shape != wait_handle.tensor.shape:
                    raise ValueError("Shapes must be equal.")
                # set parameter's global gradient
                self.params_ref[param_idx].grad.data = wait_handle.tensor
        # perform actual parameter update
        self.optimizer.step()

    def blocking_hook(self, grad_loc: torch.Tensor) -> torch.Tensor:
        """
        Add a blocking hook to the PyTorch DAG for all of the backwards calls.

        Parameters
        ----------
        grad_loc : torch.Tensor
            the local gradient

        References
        ----------
        - (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
        """
        # grad_loc_cpy = grad_loc
        # average local gradients
        grad_loc *= 1 / float(self.comm.size)
        # perform MPI Allreduce to compute global gradient
        self.comm.Allreduce(ht.MPI.IN_PLACE, grad_loc, ht.MPI.SUM)
        return grad_loc

    def blocking_grad_update(self, learning_rate: float):
        """
        Do a blocking gradient update for a SGD optimizer. This is an example only to be used to make sure that
        the model updates are being done correctly.

        Parameters
        ----------
        learning_rate : float
            the learning rate of the model
        """
        # need to send the self.parameters() to the other processes after the backwards loss step
        for f in self.parameters():
            c = torch.true_divide(f.grad.data, self.comm.size)
            self.comm.Allreduce(self.comm.MPI.IN_PLACE, c, MPI.SUM)
            f.grad.data = c
            f.data.sub_(f.grad.data * learning_rate)

    def nonblocking_hook(self, layer_name: str, param_name: str) -> Callable:
        """
        Add a nonblocking hook to send and receive the averaged parameters after the backwards step

        Parameters
        ----------
        layer_name : str
            name of the layer
        param_name : str
            name of the parameter
        """
        # hook function for blocking gradient data exchange
        def _hook(grad_loc: torch.Tensor) -> torch.Tensor:
            # counterbalance local gradient averaging
            grad_loc *= 1 / float(self.comm.size)
            # perform MPI IAllreduce to compute global gradient, returns wait handle
            wait_handle = self.comm.Iallreduce(ht.MPI.IN_PLACE, grad_loc, ht.MPI.SUM)
            # if wait handle dict does not contain the layer, add it -> automatically tracks reversed layer order
            if layer_name not in self.wait_handles:
                self.wait_handles[layer_name] = list()
            # get size of flattened tensor
            size1D = functools.reduce(operator.mul, grad_loc.shape, 1)
            # assign wait handle to its layer, layer-internal sorting by size
            bisect.insort(self.wait_handles[layer_name], (size1D, param_name, wait_handle))
            return grad_loc

        return _hook

    def forward_hook(self, layer_name: str) -> Callable:
        """
        Add a forward hook to update parameters during the forward step. This will return a hook with can be added
        using the ``submodule.register_forward_pre_hook`` command.

        Parameters
        ----------
        layer_name : str
            name of the layer
        """
        # hook function for non-blocking parameter update
        def _hook(_, input_):
            # update parameters of given layer
            param_slice = self.param_slices[layer_name]
            self.async_update(param_slice, [layer_name])

            return input_

        return _hook
