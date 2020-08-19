import bisect
import warnings

import torch
import torch.nn as tnn

from collections import OrderedDict
from typing import Callable, List
from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core import dndarray

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

        t_model = TestingModel()
        t_optimizer = torch.optim.SGD(t_model.parameters(), lr=0.01)
        ht_optimizer = ht.optim.DataParallelOptimizer(t_optimizer)
        ht_model = ht.nn.DataParallel(t_model, comm, ht_optimizer)

    and a requirement of giving a HeAT communicator (``comm``) as well as at least one DataParallelOptimizer
    (``dp_optimizers``). It's possible to pass more than one optimizer, but communication during parameter updates is
    limited to blocking then. The same limitation takes effect when passing an optimizer that does not deal exactly with
    the set of model's parameters . For the given model both the ``__init__()`` and ``forward()`` functions must be
    defined in the class defining the network.

    It is highly recommended that a HeAT DataLoader is used, see :func:`..utils.data.datatools.DataLoader`. The
    default communications scheme for this is blocking. The blocking scheme will average the model parameters during
    the backwards step, synchronizing them before the next model iteration.

    To use the non-blocking communications for parameter updates, negate the optional flag
    ``blocking_parameter_updates`.

    Attributes
    ----------
    module : torch.nn.Module
        the local module
    comm : heat.MPICommunicator
        HeAT communicator to use
    dp_optimizers : tuple
        sequence of DataParallelOptimizers to be used. Usage of more than one optimizer causes MPI communication
        to be blocking during parameter updates, regardless of ``blocking_parameter_updates`` flag
    blocking_parameter_updates : bool (optional)
        flag indicating if communication during parameter updates is blocking
    """

    def __init__(
        self,
        module: torch.nn.Module,
        comm: MPICommunication,
        *dp_optimizers,
        blocking_parameter_updates: bool = True,
    ):
        super(DataParallel, self).__init__()
        self.module = module
        self.comm = comm
        self.blocking_parameter_updates = blocking_parameter_updates

        self._dp_optimizers = list()
        self._layer_wait_handles = OrderedDict()
        self._fwd_hook_handles = list()
        # set of layers' names with active wait handles (only relevant for non-blocking)
        self._active_layers = set()
        # slices of parameters belonging to one and the same layer
        self._param_slices = dict()
        # pytorch internal parameter indexing
        self._param_indices = dict()

        # raise error if no DP optimizer is given
        if len(dp_optimizers) == 0:
            raise ValueError("You have to pass at least one DataParallelOptimizer.")

        # current implementation of non-blocking communication during parameter updates has some limitations that cause
        # fallback onto blocking in case of overstepping them
        if not self.blocking_parameter_updates:
            # usage of multiple optimizers isn't supported
            if len(dp_optimizers) > 1:
                self.blocking_parameter_updates = True
                warnings.warn(
                    "Usage of more than one DataParallelOptimizer causes fallback on blocking MPI "
                    "communication during parameter updates.",
                    stacklevel=2,
                )
            # usage of optimizer with parameters being unequal to model's parameters isn't supported
            elif (
                list(module.parameters())
                != dp_optimizers[0].torch_optimizer.param_groups[0]["params"]
            ):
                self.blocking_parameter_updates = True
                warnings.warn(
                    "Usage of optimizer whose parameters differ from module's parameters causes fallback on blocking "
                    "MPI communication during parameter updates.",
                    stacklevel=2,
                )

        # assign given optimizers to this model
        for dp_optimizer in dp_optimizers:
            self._dp_optimizers.append(dp_optimizer)
            dp_optimizer.blocking_parameter_updates = self.blocking_parameter_updates

        # unify parameters across nodes by unifying the random seed and resetting parameters
        # seed = torch.tensor([torch.random.seed() >> 33], device=torch.device('cpu'))
        # comm.Bcast(seed)
        # torch.random.manual_seed(seed.item())
        # self.module.apply(self._reset_parameters)

        # get parameter indexing and slices
        start_idx = 0
        layer_name_prev = None
        for idx, (name, param) in enumerate(module.named_parameters()):
            self._param_indices[name] = idx
            layer_name = name.rsplit(sep=".", maxsplit=1)[0]
            if layer_name_prev is None:
                layer_name_prev = layer_name
            if layer_name_prev != layer_name:
                self._param_slices[layer_name_prev] = slice(start_idx, idx)
                layer_name_prev = layer_name
                start_idx = idx

            # register backward hooks for all model parameter tensors
            if self.blocking_parameter_updates:
                param.register_hook(self._blocking_hook)
            else:
                param.register_hook(self._nonblocking_hook(layer_name, name))
        self._param_slices[layer_name_prev] = slice(start_idx, len(self._param_indices))

    def __setattr__(self, name, value):
        # auto-detect end of epoch's training phase and finalize wait handles (only relevant for non-blocking)
        if name == "training" and not value and not self.blocking_parameter_updates:
            self._iparam_update()
        super(DataParallel, self).__setattr__(name, value)

    def forward(self, *inputs: tuple, **kwargs: dict) -> torch.Tensor:
        data = inputs[0]

        if isinstance(data, dndarray.DNDarray):
            lcl_data = data._DNDarray__array
        elif isinstance(data, torch.Tensor):
            lcl_data = data
        else:
            lcl_data = torch.tensor(data)

        # check if non-blocking and training
        if not self.blocking_parameter_updates and self.module.training:
            # register forward hooks for all layers with wait handles
            for name, submodule in self.module.named_modules():
                if name == "":
                    continue
                if name in self._layer_wait_handles:
                    hook_handle = submodule.register_forward_pre_hook(self._forward_hook(name))
                    self._fwd_hook_handles.append(hook_handle)

        # perform forward pass
        ret = self.module(lcl_data, *inputs[1:], **kwargs)

        # finalize potentially remaining wait handles and update corresponding params (if
        # computation graph has changed between previous backward and this forward)
        if not self.blocking_parameter_updates and self.module.training:
            # set has to be copied in order to be manipulated during iteration
            active_layers_cpy = self._active_layers.copy()
            for layer_name in active_layers_cpy:
                self._forward_hook(layer_name)(None, None)

        # reset optimizer flag
        for dp_optimizer in self._dp_optimizers:
            dp_optimizer.update_next = False
        # clear dictionary after all wait handles are used up (dynamic computation graph)
        self._layer_wait_handles.clear()
        # remove forward hooks (dynamic computation graph)
        for hook_handle in self._fwd_hook_handles:
            hook_handle.remove()
        self._fwd_hook_handles.clear()

        return ret

    def _iparam_update(self, param_slice: slice = None, layer_names: List[str] = None) -> None:
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

        # for non-blocking, only one dp optimizer is allowed
        dp_optimizer = self._dp_optimizers[0]

        # perform update on the whole model
        if param_slice is None:
            param_slice = slice(len(dp_optimizer.params_ref))
        if layer_names is None:
            layer_names = list(self._layer_wait_handles.keys())

        # update params that are visible for the optimizer
        dp_optimizer.torch_optimizer.param_groups[0]["params"] = dp_optimizer.params_ref[
            param_slice
        ]
        # iterate over layers
        for layer_name in layer_names:
            # only perform update, if all given layers hold unfinalized wait handles (important for layer reuse)
            if layer_name not in self._active_layers:
                return
            # iterate over layer's parameters/associated wait handles
            for (_, param_name, wait_handle) in self._layer_wait_handles[layer_name]:
                # get internal index of selected parameter
                param_idx = self._param_indices[param_name]
                # synchronize, get parameter's global gradient
                wait_handle.wait()
                # check if shapes are matching
                if dp_optimizer.params_ref[param_idx].grad.data.shape != wait_handle.tensor.shape:
                    raise ValueError("Shapes must be equal.")
                # set parameter's global gradient
                dp_optimizer.params_ref[param_idx].grad.data = wait_handle.tensor
                # remove layer from set of active layers, if present
                self._active_layers.discard(layer_name)
        # if desired, perform actual parameter update
        if dp_optimizer.update_next:
            dp_optimizer.torch_optimizer.step()

    def _blocking_hook(self, grad_loc: torch.Tensor) -> torch.Tensor:
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
        # average local gradients
        grad_loc *= 1 / float(self.comm.size)
        # perform MPI Allreduce to compute global gradient
        self.comm.Allreduce(MPI.IN_PLACE, grad_loc, MPI.SUM)

        return grad_loc

    def _nonblocking_hook(self, layer_name: str, param_name: str) -> Callable:
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
            wait_handle = self.comm.Iallreduce(MPI.IN_PLACE, grad_loc, MPI.SUM)
            # if layer wait handle dict does not contain the layer, add it -> automatically tracks reversed layer order
            if layer_name not in self._layer_wait_handles:
                self._layer_wait_handles[layer_name] = list()
            # add layer to set of active layers
            self._active_layers.add(layer_name)
            # get size of flattened tensor
            size_1d = torch.numel(grad_loc)
            # assign wait handle to its layer, layer-internal sorting by size
            bisect.insort(self._layer_wait_handles[layer_name], (size_1d, param_name, wait_handle))
            return grad_loc

        return _hook

    def _forward_hook(self, layer_name: str) -> Callable:
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
            param_slice = self._param_slices[layer_name]
            self._iparam_update(param_slice, [layer_name])
            return input_

        return _hook

    @staticmethod
    def _reset_parameters(module: tnn.Module) -> None:
        """
        Reset parameters of given torch submodule. Only works for basic module types containing ``reset_parameters``
        function.

        Parameters
        ----------
        module: torch.nn.Module
            submodule whose parameters are to be reset
        """

        if callable(getattr(module, "reset_parameters", None)):
            module.reset_parameters()
