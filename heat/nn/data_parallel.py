import bisect
import os
import warnings
import torch
import torch.nn as tnn
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as tDDP

from collections import OrderedDict
from typing import Callable, List, Union, Tuple
from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD
from .. import optim

import time
from typing import Iterator


__all__ = ["DataParallel", "DataParallelMultiGPU"]


def __sum_f16_cb(buffer_a, buffer_b, _):
    tens_a = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_a, "native"))
    tens_b = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_b, "native"))
    tens_b += tens_a


# create new OP
mpi_sum_f16 = MPI.Op.Create(__sum_f16_cb, commute=True)


def addCounter(counter1, counter2, datatype):
    for item in counter2:
        if item in counter1:
            counter1[item] += counter2[item]
        else:
            counter1[item] = counter2[item]
    return counter1


counterSumOp = MPI.Op.Create(addCounter, commute=True)


class DataParallel(tnn.Module):
    """
    Implements data parallelism across multiple processes. This means that the same model will be run locally
    on each process. Creation of the model parallels to PyTorch, the only changes are using HeAT layers (ht.nn.layer)
    in the initialization of the network. I.E. (example code contains). If there is not a HeAT layer,
    it will fall back to the PyTorch layer of the same name.
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

    and a requirement of giving a HeAT communicator (``comm``, :class:`..core.communication.MPICommunication`)
    and at least one DataParallelOptimizer (``dp_optimizers``, :class:`..optim.dp_optimizer.DataParallelOptimizer`).
    It's possible to pass more than one optimizer, but communication during parameter updates is limited to blocking
    then. The same limitation takes effect when passing an optimizer that does not deal exactly with the set of model's
    parameters. For the given model both the ``__init__()`` and ``forward()`` functions must be defined in the class
    defining the network.

    It is highly recommended that a HeAT DataLoader is used, see :func:`..utils.data.datatools.DataLoader`. The
    default communications scheme for this is blocking. The blocking scheme will average the model parameters during
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
        if not isinstance(optimizer, (list, tuple)):
            optimizer = [optimizer]
        for i in optimizer:
            if not isinstance(i, optim.DataParallelOptimizer):
                raise TypeError("optimizers must be optim.DataParallelOptimizer")

        # current implementation of non-blocking communication during parameter updates has some limitations that cause
        # fallback onto blocking in case of overstepping them
        if not self.blocking_parameter_updates:
            # usage of multiple optimizers isn't supported nor is the
            # usage of optimizer with parameters being unequal to model's parameters
            if (
                len(optimizer) > 1
                or list(module.parameters())
                != optimizer[0].torch_optimizer.param_groups[0]["params"]
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

    def __setattr__(self, name, value):
        # auto-detect end of epoch's training phase and finalize wait handles (only relevant for non-blocking)
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
        """
        Update parameters asynchronously via wait handles.

        Parameters
        ----------
        param_slice : slice, optional
            Slice object for creating a view onto optimizer's params list.
            By default, the whole params list is used, (``None``)
        layer_names : list(str), optional
            List of layer names which parameters will be updated, must match param_slice.
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
            for (param_name, wait_handle, dtp, tens) in self._layer_wait_handles[layer_name]:
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
                # print(tens.dtype, dtp)
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
        wrk = grad_loc.to(torch.bfloat16)
        # average local gradients
        wrk *= 1 / float(self.comm.size)
        # perform MPI Allreduce to compute global gradient
        self.comm.Allreduce(MPI.IN_PLACE, wrk, mpi_sum_f16)
        return wrk.to(grad_loc.dtype)

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
                wrk = grad_loc.to(torch.bfloat16)
            # counterbalance local gradient averaging
            wrk *= 1 / float(self.comm.size)
            # perform MPI IAllreduce to compute global gradient, returns wait handle
            wait_handle = self.comm.Iallreduce(MPI.IN_PLACE, wrk, mpi_sum_f16)
            # if layer wait handle dict does not contain the layer, add it -> automatically tracks reversed layer order
            if layer_name not in self._layer_wait_handles:
                self._layer_wait_handles[layer_name] = list()
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
    working for data parallel stuff

    loss_floor is where the user would hope for the loss to get to
    """

    def __init__(
        self,
        module: torch.nn.Module,
        comm: MPICommunication,
        optimizer: optim.DataParallelOptimizer,
        overlap: bool = True,
        distributed_twice: bool = True,
        skip_batches: Union[List, Tuple, int] = None,
        loss_floor: Union[float, int] = 1.0,
    ):
        super(DataParallelMultiGPU, self).__init__()
        rank = comm.rank
        loc_gpus = torch.cuda.device_count()
        self.loc_gpus = loc_gpus
        local_rank = rank % loc_gpus
        self.auto_skip, self.local_skip, self.local_skip = False, False, 1
        if loc_gpus > 1 and distributed_twice:
            base_loc_ranks = list(range(0, comm.size, loc_gpus))
            reduced_comms = []
            reduced_ranks = []
            for i in range(loc_gpus):
                lp_ranks = [j + i for j in base_loc_ranks]
                color = 111 + i if rank in lp_ranks else 222 + i
                key = 0 + i if rank in lp_ranks else 444 + i
                reduced_comms.append(MPICommunication(MPI_WORLD.Split(color, key)))
                reduced_ranks.append(tuple(lp_ranks))
            self.reduced_comms, self.reduced_ranks = reduced_comms, reduced_ranks
            self.base_loc_ranks = base_loc_ranks
            module = tDDP(module, device_ids=[local_rank])  # , process_group=lg)
            # module.share_memory()
            device = "cuda:" + str(local_rank)
            torch.cuda.set_device(device=device)
            self.old_require_backward_grad_sync = None
            if skip_batches is None:
                self.auto_skip = True
                skip_batches = 16
                self.local_skip = 1
            # self.no_sync = module.no_sync
        self.module = module
        self.comm = comm
        self.overlap = overlap
        self.waiting = False

        self._layer_wait_handles = OrderedDict()
        self._fwd_hook_handles = list()
        # set of layers' names with active wait handles (only relevant for non-blocking)
        self._active_layers = set()
        # slices of parameters belonging to one and the same layer
        self._param_slices = dict()
        # pytorch internal parameter indexing
        self._param_indices = dict()

        # raise error if no DP optimizer is given
        if not isinstance(optimizer, optim.DataParallelOptimizer):
            raise TypeError("optimizers must be optim.DataParallelOptimizer")

        # assign given optimizers to this model
        self.optimizer = optimizer
        optimizer.step = self.step
        # unify parameters across nodes by unifying the random seed and resetting parameters
        torch.random.manual_seed(2147483646)  # max int32 value - 1
        self.module.apply(self._reset_parameters)

        self.current_batch, self.last_batch = 0, None
        if isinstance(skip_batches, int):
            # todo: raise error is the given param is a list of 2 values
            skip_batches = [[skip_batches, 0]]
            # skip batches should be a list of lists or list of tuples.
            # the first index is how many to skip, the second int would be the batch number to change at
        self.skip_batches = skip_batches
        self.og_skip_batches = skip_batches
        self.global_skips = skip_batches[0][0]
        self._prev_params = []
        self.epoch = 0
        self._send_mod, self._send_mod_m1 = 0, None

        self._prev_losses_mean, self._prev_losses_std = [], []
        self._loss_rcv, self._loss_wait = None, []
        self.start_loss = None
        self.loss_switch_target = None
        self.loss_floor = loss_floor + 0.5

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _stop_local_sync(self):
        # stop local synchronizations for tDDP
        if not isinstance(self.module, tDDP) or not self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        # self.old_require_backward_grad_sync = self.module.require_backward_grad_sync
        self.module.require_backward_grad_sync = False

    def _start_local_sync(self):
        # *restart* local synchronizations for tDDP
        if not isinstance(self.module, tDDP) or self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        # self.module.require_backward_grad_sync = self.old_require_backward_grad_sync
        self.module.require_backward_grad_sync = True

    def reset_skips(self):
        # need to reset the skips after the learning rate is adjusted
        # for step based learning
        print("resetting skips")
        self.skip_batches = self.og_skip_batches
        self.global_skips = self.og_skip_batches[0][0]
        self.local_skip = self.global_skips // 2

    def epoch_loss_logic(self, loss):
        # this should be called during the epoch
        # send the current loss value
        # gather the losses from previous batches
        # get the mean and std of the losses
        # save into a list / tensor with the previous results
        with torch.no_grad():
            # send the current lossses first
            loss_send = torch.zeros(self.comm.size)
            # todo: check that its ``loss.data`` and not something else
            loss_send[self.comm.rank] = loss.data
            self._loss_wait.append(
                [self.comm.Iallreduce(MPI.IN_PLACE, loss_send, MPI.SUM), loss_send]
            )
            if self.epoch == 0:
                # wait until the next epoch to receive the data
                return

            rcv = self._loss_wait.pop(0)
            rcv[0].wait()
            loss_send = rcv[1]
            mu = torch.mean(loss_send)
            if self.start_loss is None:
                # logic to signal the drop in LR
                # todo: customizable logic?
                self.start_loss = mu
                self.loss_switch_target = mu // 2
            # std = torch.std(loss_send)
            self._prev_losses_mean.append(mu)
            # self._prev_losses_std.append(std)
            # todo: how many to collect before changing the update delay!
            if self.current_batch < 4:
                # first need to fill the list before doing anything else
                return
            means = torch.tensor(self._prev_losses_mean)
            diff = means[-1] - means[0]

            lr_adjust = False
            if diff < 0.1 or means[-1] < self.loss_switch_target:
                lr_adjust = True

            if means[-1] <= self.loss_floor and not lr_adjust:
                # adjust batch skips
                print("adjusting batch skips in loss logic")
                self.global_skips //= 2
                self.local_skip //= 2
                if self.local_skip == 0:
                    self.local_skip = 1
                if self.global_skips == 0:
                    self.global_skips = 1
            elif lr_adjust:
                # reset the skip numbers
                self.reset_skips()
            # if the diff is < 0.1 adjust learning rate, at the end of the learning rate path,
            #       then change the difference in the batch skips

            return lr_adjust
        # todo: batch decay

        # todo: reset the skip after the learning rate is adjusted

    def step(self):
        # TODO: raise error is last batch is not set
        # todo: exchange losses in nonblocking way to determine if stabilized
        # todo: non-data parallel update for torch
        #       -> toggle no_sync property or the tDDP model somehow
        # collect the parameters from the current batch -> save + (non?)blocking send
        # test for receive from last batch,
        #   if yes: receive, update parameters with rcved stuff
        # copy and send the parameter dictionary
        # t = time.perf_counter()
        # copy whole dict and sent it
        # t3 = time.perf_counter()
        self.optimizer.torch_optimizer.step()

        # adjust skip numbers
        if self.current_batch == self.last_batch or self.current_batch % self.global_skips == 0:
            # if last batch -> run last batch sync and return -> full global sync
            # if self.batch_num % global_skip == 0 -> full global sync
            #print(self.current_batch, "global sync")
            self._full_global_sync()
            # self._start_local_sync()
            #if self.comm.rank == 0:
            #    print("global send", self.current_batch - 1)
            return
        # elif self.current_batch % self.local_skip == 0:
            #    self.current_batch % self.global_skips == 2
            #self.current_batch % self.global_skips == self.global_skips - 1 or
            #self.current_batch == self.last_batch - 1
            #):
            # if self.batch_num % local_skip == local_skip - 1 -> turn on local sync
            # self._start_local_sync()
            #self._stop_local_sync()
        #    if self.comm.rank == 0:
        #        print("starting to skip", self.current_batch)
        elif self.current_batch % self.global_skips == 4:
            # elif self.current_batch % self.local_skip == 0 or self.waiting:
            # if self.batch_num % local_skip == 0 -> try to receive data, once received, turn off local sync
            # if trying to receive -> try to receive still, once received, turn off local sync           
            self._global_sync_test_rcv()
            #self._stop_local_sync()
            #if self.comm.rank == 0:
            #    print("rcv + stoping local skip", self.current_batch)
            # self._stop_local_sync()
        elif (
            self.current_batch == self.last_batch - 2 or
            self.current_batch % self.global_skips == self.global_skips - 1 or 
            self.current_batch % self.local_skip == self.local_skip - 1
        ):
            #if self.comm.rank == 0:
            #    print("starting to locally skip", self.current_batch)
            pass
            #self._start_local_sync()
        elif self.current_batch % self.local_skip == 0:
            pass
            #self._stop_local_sync()

        self.current_batch += 1
        #if self.current_batch > self.last_batch - 10:
        #    self._start_local_sync()
        if self.current_batch == 0:
            # this only occurs at the end of an epoch (batch numbers reset w/in full global sync)
            pass

        # if self.comm.rank == 0:
        #     print("optimizer step", time.perf_counter() - t3)
        #
        # if self.current_batch != self.last_batch - 1:
        #     if self._send_mod is not None and self.current_batch < self.last_batch - self.global_skips:
        #         # loop for stopping and starting the local updates
        #         # todo: map out when to hold to syncs,
        #         #       should sync on the batch before
        #         #       and turn it off after the sync
        #         pass
        #     if self.current_batch % self.global_skips != 0:  # -> this is updating when batch % global skip == 0
        #         self.current_batch += 1
        #         return
        # todo: receive the data from before and average it in
        #       -> wait 3 batches or something like that?
        #       -> need to do this AFTER a step that has a
        # if self.comm.rank == 0:
        #    print(self.current_batch, self.global_skips)
        # mod_hold = self.current_batch % self.loc_gpus
        # self._full_global_sync()
        # if self.comm.rank == 0:
        #     print("step time", time.perf_counter() - t)

    def _batch_skip_adjustments(self):
        if self.current_batch != 0:
            # this should only do things at the beginning/end of an epoch
            return

        # if self.auto_skip:
        # todo:
        #     # todo: need to reduce the number of skips as the loss decreases
        #     pass

        if (
            len(self.skip_batches) > 0
            and self.epoch == self.skip_batches[0][1]
            and not self.auto_skip
        ):
            # adjust this at the beginning of the epoch
            # del self.skip_batches[0]
            if len(self.skip_batches) > 0:
                self.global_skips = self.skip_batches[0][0]
            del self.skip_batches[0]
            # todo: set torch skips
            self.local_skip = self.global_skips // 2

    @torch.no_grad()
    def _full_global_sync(self):
        current_comm = self.reduced_comms[self._send_mod]
        current_ranks = self.reduced_ranks[self._send_mod]

        # mod_hold_m2 = None
        # if self.current_batch > 1:
        #     mod_hold_m2 = (self.current_batch - 2) % self.loc_gpus
        #     prev_ranks = self.reduced_ranks[mod_hold_m2]
        # else:
        #     prev_ranks = []

        # param_list = [[n, p] for n, p in self.parameters()]

        # mod_hold_m1 = None
        # if self.current_batch > 0:
        #     mod_hold_m1 = (self.current_batch - 1) % self.loc_gpus
        #     prev_ranks = self.reduced_ranks[mod_hold_m1]
        # else:
        #     prev_ranks = []
        # mod_hold_m1 = None
        # if self.current_batch > 0:
        # mod_hold_m1 = (self.current_batch - 1) % self.loc_gpus
        pass
        # todo: remove below is decorator works as intended
        # with torch.no_grad():
        if self.comm.rank in current_ranks:
            self._global_send_update(current_comm)

        self._update_parameters()  # -> splits off irrelevant ranks
        # needs to happen on all ranks:
        self._local_torch_param_update(self._send_mod_m1)

        self.current_batch += 1

        if self.current_batch == self.last_batch:
            # todo: abstract last batch?
            # receive the sent data to sync params across all ranks
            # print("last batch stuff")
            if self.comm.rank in current_ranks:
                if len(self._prev_params) > 1:
                    # todo: this shouldnt occur, this is here to double check if it happens
                    raise ValueError(
                        f"length of previos params was greater than 1! {len(self._prev_params)}"
                    )
                self._prev_params[0][0].wait()
                prev_params = self._prev_params.pop(0)
                shapes = prev_params[2]
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        rcv_params = prev_params[1]
                        param = (
                            rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                        )
                self._prev_params = []
            self._local_torch_param_update(self._send_mod)

            self._send_mod_m1 = None
            self._send_mod = 0
            self.epoch += 1
            self.current_batch = 0
        else:
            # self._start_local_sync()
            self._send_mod_m1 = self._send_mod
            self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0

    # before this function local updates should be on!
    def _global_sync_test_rcv(self):
        # test if the data is there -> if it isn't: return, else: rcv, update, local update, turn off local sync
        with torch.no_grad():
            if self._send_mod_m1 is None:
                return
            prev_ranks = self.reduced_ranks[self._send_mod_m1]
            # print(prev_ranks)
            if self.comm.rank in prev_ranks and len(self._prev_params) > 0 and self._prev_params[0][0] is not None:
            #if (
            #    self.comm.rank in prev_ranks
            #    and self._prev_params[0][0] is not None
            #    or self._prev_params[0][0].test()
            #):
                # receive previous ones
                self._update_parameters()
            self._local_torch_param_update(self._send_mod_m1)
            self.waiting = False

    def _global_send_update(self, current_comm):
        # pack and send the data required for a global synchronization
        # t4 = time.perf_counter()
        params = []
        shapes = {}
        st = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                # flatten and prep the data for sending
                shapes[name] = [param.shape, slice(st, st + param.numel()), param.dtype]
                params.append(param.flatten())  # .to(torch.bfloat16).flatten())
                st += param.numel()
        # factor = current_comm.size / (current_comm.size + 1)
        params = torch.cat(params) / (current_comm.size + 1.0)  # .to(torch.bfloat16)
        new_wait = current_comm.Iallreduce(MPI.IN_PLACE, params, MPI.SUM)  # mpi_sum_f16) #
        self._prev_params.append([new_wait, params, shapes])
        # if self.comm.rank == 0:
        #     print('send time', time.perf_counter() - t4)
        return new_wait

    def _local_torch_param_update(self, mod_hold_pr):
        # synchronize the local torch parameters
        if mod_hold_pr is None:
            return
        if torch.distributed.is_initialized():
            snds = {}
            # todo: test averaging in the update data instead of just setting it
            # nodes = self.comm.size // torch.cuda.device_count()
            # factor = nodes / float(self.comm.size) if torch.distributed.get_rank() == mod_hold_pr else \
            #          1 / float(self.comm.size)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    snds[name] = torch.distributed.broadcast(param, mod_hold_pr, async_op=True)
                    # param /= float(torch.cuda.device_count())
                    # param *= factor
                    # snds[name] = torch.distributed.all_reduce(param, op=torch.distributed.ReduceOp.SUM, async_op=True)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    snds[name].wait()
            del snds

    def _update_parameters(self):
        # wait for the global sync data and update on the selected rank, requires local torch param update after
        if self._send_mod_m1 is None:
            return
        prev_ranks = self.reduced_ranks[self._send_mod_m1]
        if self.comm.rank not in prev_ranks: # or self._prev_params[0][0] is None:
            # receive previous ones
            return
        # print(self._prev_params)
        if len(self._prev_params) == 0 or self._prev_params[0][0] is None:
            return
        # ttt = time.perf_counter()
        #print(self._prev_params)
        self._prev_params[0][0].wait()
        # if self.comm.rank == 0:
        #     print("wait time", time.perf_counter() - ttt)
        # need to add the weighted average to param
        prev_params = self._prev_params.pop(0)
        shapes = prev_params[2]
        for name, param in self.named_parameters():
            if param.requires_grad:
                rcv_params = prev_params[1]
                update = rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                param /= len(prev_ranks) + 1.0
                param += update

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
