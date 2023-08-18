"""
This file is for the general data parallel neural network classes.
"""
import warnings
import torch
import torch.distributed
import torch.nn as tnn

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Union, Tuple

from .. import optim
from ..core.communication import MPI
from ..core.communication import MPI_WORLD
from ..core.communication import MPICommunication


__all__ = ["DataParallel", "DataParallelMultiGPU"]


class DataParallel(tnn.Module):
    """
    Implements data parallelism across multiple processes. This means that the same model will be run locally
    on each process. Creation of the model is similar to PyTorch, the only changes are using HeAT layers (ht.nn.layer)
    in the initialization of the network/optimizer. If there is not a HeAT layer, it will fall back to the PyTorch layer
    of the same name. The same is true for the optimizer. It's possible to use more than one optimizer, but
    communication during parameter updates is limited to blocking. The same limitation takes effect when passing an
    optimizer that does not deal exactly with the set of model's parameters. For the given model both the
    ``__init__()`` and ``forward()`` functions must be defined in the class defining the network.

    An example of this is shown in `examples/mnist.py <https://github.com/helmholtz-analytics/heat/blob/504-docstring-formatting/examples/nn/mnist.py>`_.

    It is highly recommended that a HeAT DataLoader is used, see :func:`ht.utils.data.DataLoader <heat.utils.data.datatools.DataLoader>`.
    The default communications scheme for this is blocking. The blocking scheme will average the model parameters during
    the backwards step, synchronizing them before the next model iteration.

    Usage of more than one optimizer forces MPI communication to be parameter updates to use blocking communications.

    Attributes
    ----------
    module : torch.nn.Module
        The local module
    comm : MPICommunication
        Communicator to use
    optimizer : heat.DataParallelOptimizer, List, Tuple
        Individual or sequence of DataParallelOptimizers to be used
    blocking_parameter_updates : bool, optional
        Flag indicating the usage of blocking communications for parameter updates
        Default: non-blocking updates (``False``)
    """

    def __init__(
        self,
        module: torch.nn.Module,
        comm: MPICommunication,
        optimizer: Union[optim.DataParallelOptimizer, List, Tuple],
        blocking_parameter_updates: bool = False,
    ):  # noqa: D107
        if isinstance(optimizer, optim.DASO):
            raise TypeError(
                "For use with DASO please use DataParallelMultiGPU instead of DataParallel"
            )
        super(DataParallel, self).__init__()
        self.module = module
        self.comm = comm
        self.blocking_parameter_updates = blocking_parameter_updates

        self._dp_optimizers = []
        self._layer_wait_handles = OrderedDict()
        self._fwd_hook_handles = []
        # set of layers' names with active wait handles (only relevant for non-blocking)
        self._active_layers = set()
        # slices of parameters belonging to one and the same layer
        self._param_slices = {}
        # pytorch internal parameter indexing
        self._param_indices = {}

        # raise error if no DP optimizer is given
        if not isinstance(optimizer, (list, tuple)):
            optimizer = [optimizer]
        for i in optimizer:
            if not isinstance(i, optim.DataParallelOptimizer):
                raise TypeError("optimizers must be optim.DataParallelOptimizer")

        # current implementation of non-blocking communication during parameter updates has some limitations that cause
        # fallback onto blocking in case of overstepping them
        if not self.blocking_parameter_updates and (
            len(optimizer) > 1
            or list(module.parameters()) != optimizer[0].torch_optimizer.param_groups[0]["params"]
        ):
            self.blocking_parameter_updates = True
            warnings.warn(
                "Usage of more than one DataParallelOptimizer causes fallback on blocking MPI "
                "communication during parameter updates.",
                stacklevel=2,
            )

        # assign given optimizers to this model
        for dp_optimizer in optimizer:
            self._dp_optimizers.append(dp_optimizer)
            dp_optimizer.blocking_parameter_updates = self.blocking_parameter_updates

        # unify parameters across nodes by unifying the random seed and resetting parameters
        torch.random.manual_seed(2147483646)  # max int32 value - 1
        self.module.apply(self._reset_parameters)

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

    def __setattr__(self, name: str, value: Union[torch.nn.Module, torch.Tensor, Any]) -> None:
        """
        Overwrite the current torch.nn.Module.__setattr__ so that it auto-detects the end of epoch's
        training phase and finalize wait handles (only relevant for non-blocking)
        """
        if name == "training" and not value and not self.blocking_parameter_updates:
            self._iparam_update()
        super(DataParallel, self).__setattr__(name, value)

    def forward(self, *inputs: tuple, **kwargs: dict) -> torch.Tensor:
        """
        Do the forward step for the network, receive the parameters from the last
        """
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
        ret = self.module(*inputs, **kwargs)
        # finalize potentially remaining wait handles and update corresponding params (if
        # computation graph has changed between previous backward and this forward)
        if not self.blocking_parameter_updates and self.module.training:
            # set has to be copied in order to be manipulated during iteration
            active_layers_cpy = self._active_layers.copy()
            for layer_name in active_layers_cpy:
                self._forward_hook(layer_name)(None, None)
        # reset optimizer flag
        for ldp_optimizer in self._dp_optimizers:
            ldp_optimizer.update_next = False
        # clear dictionary after all wait handles are used up (dynamic computation graph)
        self._layer_wait_handles.clear()
        # remove forward hooks (dynamic computation graph)
        for hook_handle in self._fwd_hook_handles:
            hook_handle.remove()
        self._fwd_hook_handles.clear()

        return ret

    def _iparam_update(self, param_slice: slice = None, layer_names: List[str] = None) -> None:
        r"""
        Update parameters asynchronously via wait handles.

        Parameters
        ----------
        param_slice : slice, optional
            Slice object for creating a view onto optimizer's params list.\n
            By default, the whole params list is used, (``None``)
        layer_names : list(str), optional
            List of layer names which parameters will be updated, must match param_slice.\n
            By default, all layers are updated (``None``)
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
        for layer_name in reversed(layer_names):
            # only perform update, if all given layers hold unfinalized wait handles (important for layer reuse)
            if layer_name not in self._active_layers:
                return
            # iterate over layer's parameters/associated wait handles
            for param_name, wait_handle, dtp, tens in self._layer_wait_handles[layer_name]:
                # get internal index of selected parameter
                param_idx = self._param_indices[param_name]
                # synchronize, get parameter's global gradient
                wait_handle.wait()
                # check if shapes are matching
                if (
                    dp_optimizer.params_ref[param_idx].grad.data.shape != tens.shape
                ):  # wait_handle.tensor.shape:
                    raise ValueError("Shapes must be equal.")
                # accumulate parameter's global gradient
                dp_optimizer.params_ref[param_idx].grad.data += tens.to(dtp)  # wait_handle.tensor
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
            The local gradient

        References
        ----------
        [1] (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
        """
        grad_loc_bf = grad_loc.to(torch.float)  # bfloat16)
        # average local gradients
        grad_loc_bf *= 1 / float(self.comm.size)
        # perform MPI Allreduce to compute global gradient
        self.comm.Allreduce(MPI.IN_PLACE, grad_loc_bf, MPI.SUM)  # mpi_sum_bf16)
        return grad_loc_bf.to(grad_loc.dtype)

    def _nonblocking_hook(self, layer_name: str, param_name: str) -> Callable:
        """
        Add a nonblocking hook to send and receive the averaged parameters after the backwards step

        Parameters
        ----------
        layer_name : str
            Name of the layer
        param_name : str
            Name of the parameter
        """
        # hook function for blocking gradient data exchange

        def _hook(grad_loc: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                wrk = grad_loc.to(torch.float)  # bfloat16)
            # counterbalance local gradient averaging
            wrk *= 1 / float(self.comm.size)
            # perform MPI IAllreduce to compute global gradient, returns wait handle
            wait_handle = self.comm.Iallreduce(MPI.IN_PLACE, wrk, MPI.SUM)  # mpi_sum_bf16)
            # if layer wait handle dict does not contain the layer, add it -> automatically tracks reversed layer order
            if layer_name not in self._layer_wait_handles:
                self._layer_wait_handles[layer_name] = []
            # add layer to set of active layers
            self._active_layers.add(layer_name)
            # assign wait handle to its layer, layer-internal sorting by size
            # bisect.insort(
            #     self._layer_wait_handles[layer_name], (wrk.numel(), param_name, wait_handle)
            # )
            # TODO: is sorting faster? or is there any difference?
            self._layer_wait_handles[layer_name].append(
                (param_name, wait_handle, grad_loc.dtype, wrk)
            )
            # don't return grad_loc, otherwise gradient is doubled
            return torch.zeros(*wrk.size(), device=grad_loc.device)

        return _hook

    def _forward_hook(self, layer_name: str) -> Callable:
        """
        Add a forward hook to update parameters during the forward step. This will return a hook with can be added
        using the ``submodule.register_forward_pre_hook`` command.

        Parameters
        ----------
        layer_name : str
            Name of the layer
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
            Submodule whose parameters are to be reset
        """
        if callable(getattr(module, "reset_parameters", None)):
            module.reset_parameters()


class DataParallelMultiGPU(tnn.Module):
    """
    This creates data parallel networks local to each node using PyTorch's distributed class. This does NOT
    do any global synchronizations. To make optimal use of this structure, use :func:`ht.optim.DASO <heat.optim.dp_optimizer.DASO>`.

    Notes
    -----
    The PyTorch distributed process group must already exist before this class is initialized.

    Parameters
    ----------
    module: torch.nn.Module
        an implemented PyTorch model
    optimizer: optim.DASO
        A DASO optimizer. Other optimizers are not yet implemented. The DASO optimizer should be
        defined prior to calling this class.
    comm: MPICommunication, optional
        A global communicator.
        Default: :func:`MPICommunication <heat.core.comm.MPICommunication>`
    """

    def __init__(
        self, module: torch.nn.Module, optimizer: optim.DASO, comm: MPICommunication = MPI_WORLD
    ):  # noqa: D107
        super(DataParallelMultiGPU, self).__init__()
        rank = comm.rank
        if torch.cuda.device_count() > 1:
            self.loc_gpus = torch.cuda.device_count()
            local_rank = rank % self.loc_gpus
            device = f"cuda:{str(local_rank)}"
            torch.cuda.set_device(device=device)
            module = tnn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
        else:
            warnings.warn(
                "DataParallelMultiGPU should be used with multiple GPUs per node", UserWarning
            )
        self.module = module
        self.comm = comm

        # unify parameters across nodes by unifying the random seed and resetting parameters
        self.module.apply(self._reset_parameters)

        optimizer.set_model(self.module)

    def forward(self, *inputs: Tuple, **kwargs: Dict) -> torch.Tensor:
        """
        Calls the forward method for the torch model
        """
        return self.module(*inputs, **kwargs)

    @staticmethod
    def _reset_parameters(module: tnn.Module) -> None:
        """
        Reset parameters of given torch submodule. Only works for basic module types containing ``reset_parameters``
        function.

        Parameters
        ----------
        module: torch.nn.Module
            Submodule whose parameters are to be reset
        """
        if callable(getattr(module, "reset_parameters", None)):
            module.reset_parameters()
