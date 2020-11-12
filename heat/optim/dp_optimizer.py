import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as tDDP
from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD

from typing import Union, List, Tuple

import time

__all__ = ["DataParallelOptimizer", "SkipBatches"]


def print0(*args, **kwargs):
    if MPI_WORLD.rank == 0:
        print(*args, **kwargs)


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
    nelem = torch.prod(torch.tensor(tens_b.shape)).item()
    new_buff = MPI.memory.fromaddress(tens_b.data_ptr(), nbytes=tens_b.element_size() * nelem)
    buffer_b[:] = new_buff


# create new OP
mpi_sum_f16 = MPI.Op.Create(__sum_f16_cb, commute=True)
mpi_sum_bfloat = MPI.Op.Create(__sum_bfloat_cb, commute=True)


def addCounter(counter1, counter2, datatype):
    for item in counter2:
        if item in counter1:
            counter1[item] += counter2[item]
        else:
            counter1[item] = counter2[item]
    return counter1


counterSumOp = MPI.Op.Create(addCounter, commute=True)


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


class SkipBatches:
    """
    Optimizer which skips batches
    """

    def __init__(
        self,
        local_optimizer: torch.optim.Optimizer = torch.optim.SGD,
        named_parameters=None,
        comm: MPICommunication = MPI_WORLD,
        skip_batches: Union[List, Tuple, int] = None,
        local_skip: int = None,
        loss_floor: Union[float, int] = 1.0,
        global_skip_delay: int = 4,
    ):
        self.lcl_optimizer = local_optimizer
        # reference of optimizer's params
        self.params_ref = local_optimizer.param_groups[0]["params"]
        self.named_params = named_parameters

        # TODO: MAKE SURE TO PUT THIS *AFTER* THE DDP MODEL??

        rank = comm.rank
        loc_gpus = torch.cuda.device_count()
        self.loc_gpus = loc_gpus
        local_rank = rank % loc_gpus
        self.local_skip = 1
        if loc_gpus > 1:
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

            self.device = "cuda:" + str(local_rank)
            torch.cuda.set_device(device=self.device)
            if skip_batches is None:
                skip_batches = 8
                self.local_skip = skip_batches // 2

        self.comm = comm

        self.current_batch, self.last_batch = 0, None

        self.og_global_skip = skip_batches
        self.global_skip = skip_batches
        self.local_skip = skip_batches // 2 if local_skip is None else local_skip
        self.og_local_skip = self.local_skip

        self._prev_params = []
        self.epoch = 0
        self._send_mod, self._send_mod_m1 = 0, None

        self._prev_losses_mean, self._prev_losses_std = [], []
        self._loss_wait = []
        self.start_loss = None
        self.loss_switch_target = None
        self.loss_floor = loss_floor
        self.batches_to_wait = global_skip_delay
        self._og_btw = global_skip_delay
        self._inc_ls = False
        self._param_send_shp = None

    def set_model(self, model):
        self.module = model

    def _stop_local_sync(self):
        # stop local synchronizations for tDDP
        if not isinstance(self.module, tDDP) or not self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        self.module.require_backward_grad_sync = False

    def _start_local_sync(self):
        # *restart* local synchronizations for tDDP
        if not isinstance(self.module, tDDP) or self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        self.module.require_backward_grad_sync = True

    def reset_skips(self):
        # need to reset the skips after the learning rate is adjusted
        # for step based learning
        print0("resetting skips", self.og_global_skip, self.og_local_skip, self._og_btw)
        self.global_skip = self.og_global_skip
        self.local_skip = self.og_local_skip
        self._prev_losses_mean = []
        self.batches_to_wait = self._og_btw

    @torch.no_grad()
    def new_loss_logic(self, loss):
        loss_send = torch.zeros(self.comm.size)
        # todo: check that its ``loss.data`` and not something else (why not just loss?)
        loss_send[self.comm.rank] = loss.data
        # todo: time this
        t_ls0 = time.perf_counter()
        self.comm.Allreduce(MPI.IN_PLACE, loss_send, MPI.SUM)
        print0(f"Loss allreduce time: {time.perf_counter() - t_ls0}")

        avg_loss = torch.mean(loss_send)
        self._prev_losses_mean.append(avg_loss)

        if self.loss_switch_target is None:
            self.loss_switch_target = avg_loss / 3.0

        epochs_to_wait = 4
        if len(self._prev_losses_mean) < epochs_to_wait:
            return

        means = torch.tensor(self._prev_losses_mean)
        diff = abs(means[-1] - means[-1 * epochs_to_wait])

        if avg_loss < self.loss_floor + 0.4:
            self.global_skip = 0
            self.local_skip = 0
            self.batches_to_wait = 0
            print0("all 0s", avg_loss)
            if len(self._prev_losses_mean) == 3:
                self._prev_losses_mean = []
                self.global_skip = 4
                self.local_skip = 1
                self.batches_to_wait = 1
            return
        elif avg_loss < self.loss_floor + 0.75:
            # todo: set the global skips to half? or just to 4?
            self.global_skip //= 2
            if self.global_skip < 4:
                self.global_skip = 4
            self.local_skip = 1
            self.batches_to_wait = 1
            print0(
                f"\tLoss floor: gs: {self.global_skip}, ls {self.local_skip}"
                f", btw {self.batches_to_wait}, {avg_loss}"
            )
            self._prev_losses_mean = []
            # if diff < 0.1:
            #     # if the loss is stable and in this range, do what???
            #     pass
        elif avg_loss < self.loss_switch_target:
            # todo: double global skips v reset global skips
            # self.global_skip = self.og_global_skip // 2
            self.global_skip *= 2
            self.local_skip *= 2
            self.batches_to_wait *= 2

            self.loss_switch_target /= 2.0
            self._inc_ls = True
            print0(
                f"\tbelow loss target: gs: {self.global_skip}, ls {self.local_skip}"
                f", btw {self.batches_to_wait}, {avg_loss}"
            )
            self._prev_losses_mean = []
        elif diff < 0.1:
            # drop gs by factor of 2
            self.global_skip //= 2
            if self._inc_ls:
                self.local_skip *= 2
            print0(
                f"\tabv loss target, stable:, gs: {self.global_skip}, ls {self.local_skip}"
                f", btw {self.batches_to_wait}, {avg_loss}"
            )
            self._prev_losses_mean = []
        if self.global_skip == 0:
            self.global_skip = 1
        if self.local_skip > self.global_skip:
            self.local_skip = self.global_skip
        if self.batches_to_wait > self.local_skip:
            self.batches_to_wait = self.local_skip

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

            # this will signal a LR switch when the loss is not changing or below a threshold
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
            epochs_to_wait = 3
            if len(self._prev_losses_mean) < epochs_to_wait:
                return False
            means = torch.tensor(self._prev_losses_mean)
            diff = abs(means[-1] - means[-1 * epochs_to_wait])
            diff_rec = abs(means[-1] - means[-2])

            lr_adjust = False
            if self.comm.rank == 0:
                print("\t\t\t", diff, diff_rec)
            if 0.0 <= diff <= 0.08:
                # and 0.0 <= diff_rec <= 0.08:  # or means[-1] < self.loss_switch_target:
                self._prev_losses_mean = []

                if self.global_skip > 4:
                    self.global_skip //= 2

                self.local_skip //= 2
                # self.loss_floor -= 0.3

                if self.local_skip == 0:
                    self.local_skip = 1
                if self.global_skip == 0:
                    self.global_skip = 1
                if self.comm.rank == 0:
                    print(
                        "\tadjust batches:",
                        diff,
                        means[-1],
                        " global:",
                        self.global_skip,
                        "local:",
                        self.local_skip,
                    )
            elif means[-1] <= self.loss_floor and self.local_skip < self.global_skip:
                # todo: tune this...
                if self.comm.rank == 0:
                    print(
                        "\t\t\t doubling local skips",
                        self.global_skip,
                        self.local_skip * 2,
                        "batches to wait:",
                        self.batches_to_wait - 1,
                        self.loss_floor,
                    )
                self._prev_losses_mean = []  # self._prev_losses_mean[-2:]
                self.batches_to_wait -= 1
                self.local_skip *= 2
                self.loss_floor -= 0.15
                self.global_skip //= 2
            elif means[-1] < self.loss_floor:
                print0(
                    "\t\t\t\t reducing skips and batches to wait -> hit loss floor",
                    self.global_skip // 2,
                    self.local_skip // 2,
                    1,
                )
                self._prev_losses_mean = self._prev_losses_mean[-2:]
                self.global_skip //= 2
                self.local_skip //= 2
                self.batches_to_wait = 1
                self._prev_losses_mean = []
            # if 70 <= self.epoch < 80:
            #    print0("\t\t\t override skips!", 16, 16, "batches to wait:", self.batches_to_wait + 1)
            #    self.local_skip = 16
            #    self.global_skip = 16
            #    self.batches_to_wait = 2
            # if 80 <= self.epoch < 87:
            #    self.local_skip = 4
            #    self.global_skip = 4
            #    self.batches_to_wait = 4
            # if self.epoch >= 87:
            #    # todo: tune this value and make it do this when it is near the loss floor
            #    self.local_skip = 2
            #    self.global_skip = 2
            #    self.batches_to_wait = 1
            return lr_adjust
        # todo: reset the skip after the learning rate is adjusted

    def step(self):
        # TODO: raise error is last batch is not set
        # collect the parameters from the current batch -> save + (non?)blocking send
        # test for receive from last batch,
        #   if yes: receive, update parameters with rcved stuff
        # copy and send the parameter dictionary
        self.lcl_optimizer.step()
        batch = self.current_batch
        next_batch = batch + 1
        gs = self.global_skip
        ls = self.local_skip

        gmod = batch % gs if gs > 0 else 0
        lmod = batch % ls if ls > 0 else 0

        # todo: make adjustments to logic if ls > gs

        # batches_to_wait = 1 if ls >= 1 else ls
        batches_to_wait = self.batches_to_wait
        # do full synce on global skips and on the last batch
        # todo: sync for last few batches before the end?
        if batch == self.last_batch or gmod == 0:
            btw = (
                batches_to_wait
                if batches_to_wait + batch <= self.last_batch
                else self.last_batch - batch
            )
            return self._full_global_sync(btw)

        if next_batch % gs == 0:
            # if self.comm.rank == 0:
            #     print(batch, "next batch is global sync, turning on local sync")
            self._start_local_sync()
            self.current_batch += 1
            return

        if gmod < batches_to_wait:
            # do nothing on these batches
            self.current_batch += 1
            # if self.comm.rank == 0:
            #     print(batch, "waiting for global sync")
            if next_batch == self.last_batch:
                self._start_local_sync()
            return
        elif gmod == batches_to_wait:
            # local updates should be on before this is called!
            self._update_parameters()
            self._local_torch_param_update(self._send_mod_m1)
            if ls > 1:
                self._stop_local_sync()
            # if self.comm.rank == 0:
            #     print(batch, "rcv global sync")

        if ls == 1 and next_batch != self.last_batch:
            # if self.comm.rank == 0:
            #     print(batch, "STARTING LOCAL SYNC")
            self.current_batch += 1
            self._start_local_sync()
            return

        if lmod == 0:
            # if self.comm.rank == 0:
            #     print(batch, "STOPPING local sync")
            self._stop_local_sync()
        elif next_batch % ls == 0:
            # if self.comm.rank == 0:
            #     print(batch, "STARTING local sync")
            self._start_local_sync()

        if next_batch == self.last_batch:
            self._start_local_sync()

        self.current_batch += 1

    @torch.no_grad()
    def _full_global_sync(self, batches_to_wait):
        current_comm = self.reduced_comms[self._send_mod]
        current_ranks = self.reduced_ranks[self._send_mod]

        if self.comm.rank in current_ranks:
            self._global_send_update(current_comm, batches_to_wait)
        # update parameters from the last sending (if there)
        self._update_parameters()  # -> splits off irrelevant ranks
        # needs to happen on all ranks:
        self._local_torch_param_update(self._send_mod_m1)

        if self.current_batch == self.last_batch or self.global_skip == 0:
            # print("last batch")
            # todo: abstract last batch?
            # receive the sent data to sync params across all ranks
            if self.comm.rank in current_ranks:
                if len(self._prev_params) > 1:
                    raise ValueError(f"length of previous params > 1! {len(self._prev_params)}")
                prev_params = self._prev_params.pop(0)
                shapes = prev_params[2]
                factor = 1.0 / float(len(current_ranks))
                prev_params[0].Wait()
                for name, param in self.module.named_parameters():
                    if param.requires_grad:
                        rcv_params = prev_params[1]
                        param *= 0
                        param += (
                            rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                            * factor
                        )
                self._prev_params = []
            else:
                if len(self._prev_params) > 1:
                    raise ValueError(f"OFF RANKS!len(prev_params) > 1! {len(self._prev_params)}")
            self._local_torch_param_update(self._send_mod)

            self._send_mod_m1 = None

            if self.current_batch == self.last_batch:
                self._send_mod = 0
                self.epoch += 1
                self.current_batch = 0
            else:
                self.current_batch += 1
                self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0
        else:
            self.current_batch += 1
            self._send_mod_m1 = self._send_mod
            self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0

    @torch.no_grad()
    def _global_send_update(self, current_comm, batches_to_wait):
        # pack and send the data required for a global synchronization
        op = MPI.SUM
        cast = False
        if self.global_skip == 0:
            op = mpi_sum_bfloat
            cast = True
        if self._param_send_shp is not None:
            return self._global_send_update_zeros(current_comm, batches_to_wait, op=op, cast=cast)

        params = []
        shapes = {}
        st = 0
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # flatten and prep the data for sending
                shapes[name] = [param.shape, slice(st, st + param.numel()), param.dtype]
                p = param.flatten()
                if cast:
                    p = p.to(torch.bfloat16)
                params.append(p)  # .to(torch.bfloat16))
                st += param.numel()
        params = torch.cat(params)
        self._param_send_shp = params.shape

        new_wait = current_comm.Iallreduce(MPI.IN_PLACE, params, op)  # mpi_sum_f16) #
        self._prev_params.append([new_wait, params, shapes, batches_to_wait])
        return new_wait

    @torch.no_grad()
    def _global_send_update_zeros(self, current_comm, batches_to_wait, op, cast):
        # pack and send the data required for a global synchronization
        # todo: device??, jit loop?
        params = torch.zeros(
            self._param_send_shp, device=self.device, dtype=torch.bfloat16 if cast else None
        )
        shapes = {}
        st = 0
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # flatten and prep the data for sending
                shapes[name] = [param.shape, slice(st, st + param.numel()), param.dtype]
                p = param.flatten()
                if cast:
                    p = p.to(torch.bfloat16)
                params[slice(st, st + param.numel())] = p
                st += param.numel()

        new_wait = current_comm.Iallreduce(MPI.IN_PLACE, params, op)  # mpi_sum_f16) #
        self._prev_params.append([new_wait, params, shapes, batches_to_wait])
        return new_wait

    @torch.no_grad()
    def _local_torch_param_update(self, mod_hold_pr):
        # mod_hold_pr is the process which has the updated gradients to be broadcast to the other local ranks
        # synchronize the local torch parameters
        if mod_hold_pr is None:
            return
        if torch.distributed.is_initialized():
            snds = {}
            # todo: test averaging in the update data instead of just setting it
            # nodes = self.comm.size // torch.cuda.device_count()
            # factor = nodes / float(self.comm.size) if torch.distributed.get_rank() == mod_hold_pr else \
            #          1 / float(self.comm.size)
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    snds[name] = torch.distributed.broadcast(param, mod_hold_pr, async_op=True)
                    # param /= float(torch.cuda.device_count())
                    # param *= factor
                    # snds[name] = torch.distributed.all_reduce(param, op=torch.distributed.ReduceOp.SUM, async_op=True)
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    snds[name].wait()
            del snds

    @torch.no_grad()
    def _update_parameters(self):
        # wait for the global sync data and update on the selected rank, requires local torch param update after
        if self._send_mod_m1 is None:
            return
        prev_ranks = self.reduced_ranks[self._send_mod_m1]
        if self.comm.rank not in prev_ranks:  # or self._prev_params[0][0] is None:
            # receive previous ones
            return
        if len(self._prev_params) == 0:  # or self._prev_params[0][0] is None:
            return
        prev_params = self._prev_params.pop(0)
        batches_between = float(prev_params[3])
        # add the weighted average to param
        shapes = prev_params[2]
        numer = batches_between
        denom = float(len(prev_ranks) + batches_between)
        factor = numer / denom
        prev_params[0].Wait()
        rcv_params = prev_params[1] / denom
        # todo: jit the parameter setting
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                update = rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                # print(update.mean())
                # NOTE: update here is the total sum of the params across the processes
                param *= factor
                param += update  # / denom

    def zero_grad(self) -> None:
        """
        Reset gradients of optimizer's params.
        """
        # reset view onto params in order to reset all gradients
        self.lcl_optimizer.param_groups[0]["params"] = self.params_ref[:]
        self.lcl_optimizer.zero_grad()
