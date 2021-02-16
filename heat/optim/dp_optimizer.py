import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as tDDP
from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD
from .utils import DetectMetricPlateau

from typing import Union, List, Tuple, Dict

import inspect
import math
import queue


__all__ = ["DataParallelOptimizer", "DASO"]


def queue_thread(q: queue.Queue):
    while True:
        items = q.get()
        if isinstance(items, tuple):
            func = items[0]
            args = items[1:]
            func(*args)
        else:
            items()
        q.task_done()


def __sum_f16_cb(buffer_a, buffer_b, _):
    tens_a = torch.HalfTensor().set_(torch.HalfStorage.from_buffer(buffer_a, "native"))
    tens_b = torch.HalfTensor().set_(torch.HalfStorage.from_buffer(buffer_b, "native"))
    tens_b += tens_a
    nelem = torch.prod(torch.tensor(tens_b.shape)).item()
    new_buff = MPI.memory.fromaddress(tens_b.data_ptr(), nbytes=tens_b.element_size() * nelem)
    buffer_b[:] = new_buff


def __sum_bfloat_cb(buffer_a, buffer_b, _):
    tens_a = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_a, "native"))
    tens_b = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_b, "native"))
    tens_b += tens_a
    nelem = int(tens_b.numel())
    new_buff = MPI.memory.fromaddress(tens_b.data_ptr(), nbytes=nelem * tens_b.element_size())
    buffer_b[:] = new_buff


# create new OP
mpi_sum_f16 = MPI.Op.Create(__sum_f16_cb, commute=True)
mpi_sum_bfloat = MPI.Op.Create(__sum_bfloat_cb, commute=True)


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


class DASO:
    def __init__(
        self,
        local_optimizer: torch.optim.Optimizer,
        total_epochs: int,
        comm: MPICommunication = MPI_WORLD,
        warmup_epochs: int = 4,
        cooldown_epochs: int = 4,
        scheduler: torch.optim.lr_scheduler = None,
        stablitiy_level: float = 0.05,
        max_global_skips: int = 8,
        sending_chuck_size: int = 10_000_000,
        verbose: bool = False,
        downcast_type: torch.dtype = torch.bfloat16,
    ):
        """
        Optimizer wrapper to use the Distributed Asynchronous and Selective Optimization (DASO) method.

        This optimizer using a local torch optimizer combined with the :class:`..nn.data_parappel.DataParallelMultiGPU`
        to create local DPNNs on each node consisting of the GPUs on each node. Then those networks communicate
        globally with MPI groups, each of which has a single GPU on each node.

        There are four phases to training:
            1. initialization:
                - define the `local_optimizer` -> a torch optimizer of your choice (tested with SGD)
                - (optional) choose a learning rate scheduler -- this is only for those learning rates which will also step the optimizer
                -

        Parameters
        ----------
        local_optimizer
        total_epochs
        comm
        warmup_epochs
        cooldown_epochs
        scheduler
        stablitiy_level
        max_global_skips
        sending_chuck_size
        verbose
        downcast_type
        """
        # check dtypes
        frame = inspect.currentframe()
        init_args = inspect.getargvalues(frame)[3]
        self._checktypes(init_args)

        if downcast_type == torch.bfloat16:
            self.cast_fn = mpi_sum_bfloat
        elif downcast_type == torch.half:
            self.cast_fn = mpi_sum_f16
        else:
            raise ValueError(
                f"downcast_type must be one of torch.bfloat16 or torch.half, currently {downcast_type}"
            )

        self.comm = comm
        self.verbose = verbose
        self.local_optimizer = local_optimizer
        self.params_ref = local_optimizer.param_groups[0]["params"]
        # reference of optimizer's params
        self.scheduler = scheduler

        rank = comm.rank
        loc_gpus = torch.cuda.device_count()
        self.loc_gpus = loc_gpus
        local_rank = rank % loc_gpus
        self.local_skip = 1
        if loc_gpus > 1:
            base_loc_ranks = list(range(0, comm.size, loc_gpus))
            reduced_comms, reduced_ranks = [], []
            for i in range(loc_gpus):
                lp_ranks = [j + i for j in base_loc_ranks]
                new_group = MPI_WORLD.group.Incl(lp_ranks)
                new_comm = MPI_WORLD.Create_group(new_group)
                reduced_comms.append(MPICommunication(new_comm, group=True))
                reduced_ranks.append(tuple(lp_ranks))
            self.reduced_comms, self.reduced_ranks = reduced_comms, reduced_ranks
            self.base_loc_ranks = base_loc_ranks
            if loc_gpus != torch.cuda.device_count():
                self.device = "cuda:0"
            else:
                self.device = "cuda:" + str(local_rank)
            torch.cuda.set_device(device=self.device)

        self.current_batch, self.last_batch = 0, None

        self._prev_params = []
        self.epoch = 0
        self._send_mod, self._send_mod_m1 = 0, None

        self.global_skip = 0
        self.local_skip = 0
        self.batches_to_wait = 0
        self.epochs_to_wait = 3
        self.max_gs = max_global_skips

        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.total_epochs = total_epochs

        # used in the sending of the params
        self._param_send_buffer_shape = None
        self.param_dict, self.shapes = None, None
        self._param_send_shp = None
        self.split = None

        self.stability = DetectMetricPlateau(
            mode="min", patience=2, threshold=stablitiy_level, threshold_mode="rel", eps=1e-8
        )

        self._gs8_waits = 3
        self._gs8_waited = 0

        self.split_val = sending_chuck_size

        # TODO: add these to the class params

        self.split_inds = None
        self.amp = False

    @staticmethod
    def _checktypes(args):
        # this holds all of the checks and raises for the parameters
        if not isinstance(args["local_optimizer"], torch.optim.Optimizer):
            raise TypeError(
                f"Local optimizer must be a torch optimizer object, currently {args['local_optimizer']}"
            )
        if not isinstance(args["total_epochs"], int):
            raise TypeError(f"total_epochs must be an int, currently {args['total_epochs']}")
        if not isinstance(args["comm"], MPICommunication):
            raise TypeError(f"comm must be a ht.MPICommunication object, currently {args['comm']}")
        if not isinstance(args["warmup_epochs"], int):
            raise TypeError(f"warmup_epochs must be an int, currently {args['warmup_epochs']}")
        if not isinstance(args["cooldown_epochs"], int):
            raise TypeError(f"cooldown_epochs must be an int, currently {args['cooldown_epochs']}")
        if not issubclass(args["scheduler"], torch.optim.lr_scheduler._LRScheduler):
            raise TypeError(
                f"scheduler must be a torch learning rate scheduler, currently {args['scheduler']}"
            )
        if not isinstance(args["stablitiy_level"], float):
            raise TypeError(f"stablitiy_level must be a float, currently {args['stablitiy_level']}")
        if not isinstance(args["max_global_skips"], int):
            raise TypeError(
                f"max_global_skips must be an int, currently {args['max_global_skips']}"
            )
        if not isinstance(args["sending_chuck_size"], int):
            raise TypeError(
                f"sending_chuck_size must be an int, currently {args['sending_chuck_size']}"
            )
        if not isinstance(args["verbose"], bool):
            raise TypeError(f"verbose must be a bool, currently {args['verbose']}")
        if not isinstance(args["downcast_type"], int):
            raise TypeError(
                f"downcast_type must be a torch.dtype, currently {args['downcast_type']}"
            )
        if args["warmup_epochs"] < 0:
            raise ValueError(f"warmup_epochs must be > 0")
        if args["cooldown_epochs"] < 0:
            raise ValueError(f"cooldown_epochs must be > 0")
        if args["max_global_skips"] < 0:
            raise ValueError(f"stablitiy_level must be > 0")
        if args["sending_chuck_size"] < 0:
            raise ValueError(f"sending_chuck_size must be > 0")

    def print0(self, *args, **kwargs):
        """
        Print a message on rank 0, works as a standard print statement
        """
        # print a message on rank 0
        if MPI_WORLD.rank == 0 and self.verbose:
            print(*args, **kwargs)

    def set_model(self, model: torch.nn.Module):
        """
        Set the local model for the optimizer.
        This should be called by the :class:`..nn.data_parappel.DataParallelMultiGPU`.

        Parameters
        ----------
        model: torch.nn.Module
            the local torch model.
        """
        self.module = model

    def _stop_local_sync(self):
        # stop local synchronizations for tDDP
        if not isinstance(self.module, tDDP) or not self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        self.module.require_backward_grad_sync = False

    def _start_local_sync(self):
        # *start* local synchronizations for tDDP
        if not isinstance(self.module, tDDP) or self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        self.module.require_backward_grad_sync = True

    @torch.no_grad()
    def epoch_loss_logic(self, loss, loss_globally_averaged=False):
        """
        Function controlling the number of batches between global synchronizations and the
        batches to wait before receiving the sent parameters. The warm-up and cool-down phases
        are also controlled here.

        This function should be called after the :class:`DASO.step` function.

        The number of batches between local synchronizations can also be modified here with
        minor code adjustments.

        Parameters
        ----------
        loss: torch.Tensor or float
            loss value of the current epoch
        loss_globally_averaged: bool, optional
            boolean if the loss is already globally averaged
        """

        if not loss_globally_averaged:
            loss_send = torch.zeros(self.comm.size)
            # loss.data -> this will get the raw number from the lass value and nothing else
            loss_send[self.comm.rank] = loss.data if isinstance(loss, torch.Tensor) else loss
            self.comm.Allreduce(MPI.IN_PLACE, loss_send, MPI.SUM)
            avg_loss = torch.mean(loss_send)
        else:
            avg_loss = torch.tensor(loss)

        if self.epoch < self.warmup_epochs:
            self.global_skip = 0
            self.local_skip = 0
            self.batches_to_wait = 0

            self.print0("\t\t", self.global_skip, self.local_skip, self.batches_to_wait)
            return

        elif self.warmup_epochs == self.epoch:
            self.global_skip = 4
            self.local_skip = 1
            self.batches_to_wait = 1

            self.print0("\t\t", self.global_skip, self.local_skip, self.batches_to_wait)

        if self.epoch >= self.total_epochs - self.cooldown_epochs:
            self.global_skip = 0
            self.local_skip = 0
            self.batches_to_wait = 0

            self.print0(
                "\tGlobal Skips:",
                self.global_skip,
                "Local Skips:",
                self.local_skip,
                "Batches to wait:",
                self.batches_to_wait,
            )
            return

        if self.global_skip == self.max_gs and self.max_gs > 4:
            self._gs8_waited += 1

        self.print0(
            "current best:",
            self.stability.best * (1.0 - self.stability.threshold),
            "avg loss:",
            avg_loss,
            "bad epochs:",
            self.stability.num_bad_epochs,
        )

        stable = self.stability.test_if_improving(avg_loss)

        if stable and self.global_skip > 1:
            # drop gs by factor of 2
            self.global_skip //= 2
            self.local_skip //= 2
            self.batches_to_wait -= 1  # old was //= 2

            self.print0("dropping skips")

            if self.global_skip > 0:
                if self.batches_to_wait == 0:
                    self.batches_to_wait = 1
                if self.local_skip == 0:
                    self.local_skip = 1
            self._gs8_waited = 0
        elif self.global_skip == 1 and stable:
            self.global_skip = self.max_gs
            self.local_skip = self.max_gs // 4
            self.batches_to_wait = self.max_gs // 4  # + 1  # 2

            self._gs8_waited = 0

        self.print0(
            "\tGlobal Skips:",
            self.global_skip,
            "Local Skips:",
            self.local_skip,
            "Batches to wait:",
            self.batches_to_wait,
            "\nAvg loss:",
            avg_loss,
            "Number of bad Epochs",
            self.stability.num_bad_epochs,
        )

    def add_scaler(self, scaler: torch.cuda.amp.GradScaler):
        """
        Create a reference to torch's :class:`torch.cuda.amp.GradScaler` used in torch's automatic mixed
        precision. For more information on this see https://pytorch.org/docs/stable/notes/amp_examples.html

        Parameters
        ----------
        scaler: torch.cuda.amp.GradScaler
            the gradient scaler to be used
        """
        self.scaler = scaler
        self.amp = True

    def step(self):
        """
        Perform a single optimization step.
        This will perform the `step` operations of the local optimizer,
        local learning rate scheduler (if defined), and the gradient scaler used in automatic mixed
        precision (if defined).

        Also in the step is the logic used for when to send and receive the global/local synchronizations.
        Global Syncs occur on batches for which the modulus of the batch number and the `global_skip` number is 0.
        If `batches_to_wait` > 0, the next batches have only local syncs. After that number of batches,
        the data during the global sync phase is received.

        Local synchronization can also be turned off if desired by increasing `local_skips` above 1.

        Notes
        -----
        self.last_batch must be set!!
        """
        if self.last_batch is None:
            raise ValueError(
                "self.last_batch must be set as the number of batches (len(dataloader))"
            )

        if self.amp:
            self.scaler.step(self.local_optimizer)
            # todo: add something to tell if the grads have infs or nans
            # Updates the scale for next iteration.
            self.scaler.update()
        elif self.scheduler is None:
            self.local_optimizer.step()
        else:
            self.scheduler.step()
        batch = self.current_batch
        # knowing next_batch is important to make sure that the local sync is on
        #       or if the next is the last batch
        next_batch = batch + 1
        gs = self.global_skip
        ls = self.local_skip
        # determine if to do the syncs
        gmod = batch % gs if gs > 0 else 0
        lmod = batch % ls if ls > 0 else 0

        batches_to_wait = self.batches_to_wait
        # ensure that the batch that will receive will be before the end of the training loop
        btw = (
            batches_to_wait
            if batches_to_wait + batch <= self.last_batch
            else self.last_batch - batch
        )
        # do full synce on global skips and on the last batch
        if batch == self.last_batch or gmod == 0:
            return self._full_global_sync(btw)

        if next_batch % gs == 0:
            self._start_local_sync()
            self.current_batch += 1
            return

        if gmod < btw:
            # do nothing on these batches (maintain the local sync)
            self.current_batch += 1
            if next_batch == self.last_batch:
                self._start_local_sync()
            return
        elif gmod == btw:
            # local updates should be on before this is called!
            self._update_parameters()
            self._local_torch_param_update(self._send_mod_m1)
            if ls > 1:
                self._stop_local_sync()

        if ls == 1 and next_batch != self.last_batch:
            self.current_batch += 1
            self._start_local_sync()
            return

        if lmod == 0:
            self._stop_local_sync()
        elif next_batch % ls == 0:
            self._start_local_sync()

        if next_batch == self.last_batch:
            self._start_local_sync()

        self.current_batch += 1

    @torch.no_grad()
    def _full_global_sync(self, batches_to_wait):
        """
        Performs a full global synchronization. If `batches_to_wait > 0` this will wait for that many
        batches before received in the parameters.

        Full syncs are only performed on a single MPI group
        """
        current_comm = self.reduced_comms[self._send_mod]
        current_ranks = self.reduced_ranks[self._send_mod]

        if self.comm.rank in current_ranks:
            self._global_send_update(current_comm, batches_to_wait)

        if self.batches_to_wait != 0:
            # update parameters from the last sending (if there)
            self._update_parameters()  # -> splits off irrelevant ranks
            # needs to happen on all ranks:
            self._local_torch_param_update(self._send_mod_m1)

        if self.current_batch != self.last_batch and self.batches_to_wait != 0:
            self.current_batch += 1
            self._send_mod_m1 = self._send_mod
            self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0
            return

        # if self.current_batch == self.last_batch or self.batches_to_wait == 0:
        # receive the sent data to sync params across all ranks
        if self.comm.rank in current_ranks:
            self._update_last_batch(current_ranks)
        else:
            if len(self._prev_params) > 0:
                raise ValueError(
                    f"DEBUG: off ranks have data when they shouldn't, {len(self._prev_params)}"
                    f" batch number {self.current_batch}"
                )
        self._local_torch_param_update(self._send_mod)

        self._send_mod_m1 = None

        if self.current_batch == self.last_batch:
            self._send_mod = 0
            self.epoch += 1
            self.current_batch = 0
        else:
            self.current_batch += 1
            self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def __pack_data(jtparams: torch.Tensor, iter_dict: Dict[str, torch.Tensor], cast: bool):
        """ jitted loop to pack the data into a flattened buffer to be sent"""
        st = 0
        for name, par in iter_dict.items():
            if par.requires_grad:
                # flatten and prep the data for sending
                p = torch.flatten(par)
                if cast:
                    p = p.to(torch.bfloat16)
                jtparams[st : st + par.numel()] = p
                st += par.numel()
        return jtparams

    @torch.no_grad()
    def _global_send_update(self, current_comm, batches_to_wait):
        """
        pack and send the data required for a global synchronization on the `current_comm` group

        `batches_to_wait` is sent with the parameters to keep track of this between sending and receiving
        """
        op = MPI.SUM
        cast = False
        if self.global_skip < 1:
            op = mpi_sum_bfloat
            cast = True

        param_dict, shapes = self._create_param_dict_n_shapes()
        sndparams = torch.zeros(
            self._param_send_buffer_shape,
            device=self.device,
            dtype=torch.bfloat16 if cast else None,
        )
        denom = float(current_comm.size + (batches_to_wait * 2.0))

        sndparams = self.__pack_data(sndparams, param_dict, cast, denom)

        if sndparams.isnan().sum():
            # check if there are NaNs, if so, stuff is bad
            raise ValueError(f"{sndparams.isnan().sum()} NaNs in `params` shit be fucked.")

        if not self.split and sndparams.numel() <= self.split_val:
            new_wait = current_comm.Iallreduce(MPI.IN_PLACE, sndparams, op)
            self._prev_params.append([new_wait, sndparams, shapes, batches_to_wait])
            return

        # if self.split or sndparams.numel() > self.split_val:
        self.split = True
        num_splits = math.ceil(len(sndparams) / self.split_val)
        splits = [self.split_val] * (num_splits - 1)
        rem = len(sndparams) - (self.split_val * (num_splits - 1))
        # first one will be smaller then the rest (the remainder is first)
        splits = [rem] + splits
        self.split_inds = splits
        params_list = [None] * num_splits
        prev = 0
        waits = [None] * num_splits
        for s in range(num_splits):
            # need to slice the params at the split points
            params_list[s] = sndparams[prev : splits[s] + prev]
            prev += splits[s]
            waits[s] = current_comm.Iallreduce(MPI.IN_PLACE, params_list[s], op)
        self._prev_params.append([waits, params_list, shapes, batches_to_wait])

    def _create_param_dict_n_shapes(self):
        """
        create the shape and param dictionary used for sending parameters around the MPI world.
        this will also define the buffer size if it was not previously defined.
        """
        if self.shapes is not None:
            return self.param_dict, self.shapes
        param_dict = {}
        shapes = {}
        st = 0
        for name, param in self.module.named_parameters():
            param_dict[name] = param
            numel = param.numel()
            shapes[name] = [param.shape, slice(st, st + numel), param.dtype]
            st += numel

        if self._param_send_buffer_shape is None:
            # use the total number of elements to define the sending buffer shape (single int)
            self._param_send_buffer_shape = st

        self.param_dict = param_dict
        self.shapes = shapes
        return param_dict, shapes

    @torch.no_grad()
    def _local_torch_param_update(self, mod_hold_pr):
        # send the globally updated gradients from `mod_hold_pr` to the other local processes
        if not torch.distributed.is_initialized() or mod_hold_pr is None:
            return
        snds = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                snds[name] = torch.distributed.broadcast(
                    param, mod_hold_pr, async_op=True
                )  # default is SUM
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                snds[name].wait()

    @torch.no_grad()
    def _update_last_batch(self, current_ranks):
        """
        abstracted receive for the last batch (and if `global_skips` == 0)
        """
        if len(self._prev_params) > 1:
            raise ValueError(f"length of previous params > 1! {len(self._prev_params)}")
        prev_params = self._prev_params.pop(0)
        shapes = prev_params[2]
        if not self.split:
            # print("before wait")
            prev_params[0].Wait()
            rcv_params = prev_params[1] / float(len(current_ranks))
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    param[:] = (
                        rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                    )
        else:
            ind1 = 0
            prev_params[0][0].Wait()
            del prev_params[0][0]
            rcv_params = prev_params[1][ind1] / float(len(current_ranks))
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    while shapes[name][1].stop > len(rcv_params):
                        ind1 += 1
                        prev_params[0][0].Wait()
                        del prev_params[0][0]
                        new_rcv_params = prev_params[1][ind1] / float(len(current_ranks))
                        rcv_params = torch.cat((rcv_params, new_rcv_params))
                    param[:] = (
                        rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                    )
        if len(self._prev_params) >= 1:
            print("len of prev_params is > 1!", self._prev_params)

    @torch.no_grad()
    def _update_parameters(self):
        """
        received the previously sent parameters for the last sending MPI group.
        this is also where the sent and local parameters are merged together.
        """
        # wait for the global sync data and update on the selected rank,
        # requires local torch param update after
        if self._send_mod_m1 is None:
            return
        prev_ranks = self.reduced_ranks[self._send_mod_m1]
        if self.comm.rank not in prev_ranks or len(self._prev_params) == 0:
            # receive previous gradients or
            # if no old gradients, return without doing anything
            return

        prev_params = self._prev_params.pop(0)
        batches_between = float(prev_params[3])
        shapes = prev_params[2]
        # add the weighted average to the received params
        numer = batches_between * 2.0 if batches_between > 0.0 else 1.0
        denom = float(len(prev_ranks) + numer)
        factor = numer / denom
        if not self.split:
            prev_params[0].Wait()
            rcv_params = prev_params[1] / denom
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    update = (
                        rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                    )
                    # NOTE: update here is the sum of the params across the processes
                    param *= factor
                    param += update
        else:
            # receive the first one, then when the end of the slice is higher than the amount received,
            #   then get the next one
            ind = 0
            prev_params[0][0].Wait()
            del prev_params[0][0]
            rcv_params = prev_params[1][ind] / denom
            # jit the parameter setting
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    if shapes[name][1].stop > len(rcv_params):
                        ind += 1
                        prev_params[0][0].Wait()
                        del prev_params[0][0]
                        new_rcv_params = prev_params[1][ind] / denom
                        rcv_params = torch.cat((rcv_params, new_rcv_params))
                    update = (
                        rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                    )
                    # NOTE: update here is the sum of the params across the processes
                    param *= factor
                    param += update

    def zero_grad(self) -> None:
        """
        Reset gradients of local optimizer's params.
        """
        # reset view onto params in order to reset all gradients
        self.local_optimizer.param_groups[0]["params"] = self.params_ref[:]
        self.local_optimizer.zero_grad()
