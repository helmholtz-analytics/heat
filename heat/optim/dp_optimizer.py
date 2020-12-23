import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as tDDP
from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD

from typing import Union, List, Tuple, Dict

import time
import math

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
    #buffer_a[:] = new_buff


def __sum_bfloat_cb(buffer_a, buffer_b, _):
    print("start MPI op")
    tens_a = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_a, "native"))
    tens_b = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_b, "native"))
    #print("A within mpi, number of nans", tens_a.isnan().sum(), "length", tens_a.shape)
    #print("B within mpi, number of nans", tens_b.isnan().sum(), "length", tens_b.shape)
    #print("within mpi op a:", tens_a.mean(), "b", tens_b.mean())
    #print("within mpi op", tens_b.mean())

    ret = tens_b + tens_a
    nelem = int(tens_b.numel())
    #print("within mpi op", tens_b.mean())
    new_buff = MPI.memory.fromaddress(ret.data_ptr(), nbytes=int(ret.element_size() * ret.numel()))#nelem)
    #del tens_a, tens_b
    buffer_b[:] = new_buff
    print("end MPI op")
    #del tens_a, tens_b
    #buffer_a[:] = new_buff
    #del new_buff

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


class SkipBatches:
    """
    Optimizer which skips batches
    """

    def __init__(
        self,
        local_optimizer: torch.optim.Optimizer,
        total_epochs: int,
        comm: MPICommunication = MPI_WORLD,
        warmup_epochs: int = 4,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        self.comm = comm
        self.lcl_optimizer = local_optimizer
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
                color = 111 + i if rank in lp_ranks else 222 + i
                key = 0 + i if rank in lp_ranks else 444 + i
                reduced_comms.append(MPICommunication(MPI_WORLD.Split(color, key)))
                reduced_ranks.append(tuple(lp_ranks))
            self.reduced_comms, self.reduced_ranks = reduced_comms, reduced_ranks
            self.base_loc_ranks = base_loc_ranks

            self.device = "cuda:" + str(local_rank)
            torch.cuda.set_device(device=self.device)

        self.current_batch, self.last_batch = 0, None

        self._prev_params = []
        self.epoch = 0
        self._send_mod, self._send_mod_m1 = 0, None

        self._prev_losses_mean, self._prev_losses_std = [], []
        self.global_skip = 0
        self.local_skip = 0
        self.batches_to_wait = 0
        self.epochs_to_wait = 3

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # used in the sending of the params
        self._param_send_buffer_shape = None
        self.param_dict, self.shapes = None, None
        self._param_send_shp = None
        self.split = None

        self.split_val = 10_000_000 # 5?
        self.split_inds = None

        self.amp = False
        
        def __sum_bfloat_cb(buffer_a, buffer_b, _):
            print("start MPI op")
            tens_a = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_a, "native"))
            tens_b = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_b, "native"))
            #print("A within mpi, number of nans", tens_a.isnan().sum(), "length", tens_a.shape)
            #print("B within mpi, number of nans", tens_b.isnan().sum(), "length", tens_b.shape)
            #print("within mpi op a:", tens_a.mean(), "b", tens_b.mean())
            #print("within mpi op", tens_b.mean())

            ret = tens_b + tens_a
            nelem = int(tens_b.numel())
            #print("within mpi op", tens_b.mean())
            new_buff = MPI.memory.fromaddress(ret.data_ptr(), nbytes=int(ret.element_size() * ret.numel()))#nelem)
            #del tens_a, tens_b
            buffer_b[:] = new_buff
            #del tens_a, tens_b
            #buffer_a[:] = new_buff

        self.mpi_sum_bfloat = MPI.Op.Create(__sum_bfloat_cb, commute=True)


    def set_model(self, model):
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
    def epoch_loss_logic(self, loss):
        loss_send = torch.zeros(self.comm.size)
        # loss.data -> this will get the raw number from the lass value and nothing else
        loss_send[self.comm.rank] = loss.data

        self.comm.Allreduce(MPI.IN_PLACE, loss_send, MPI.SUM)

        avg_loss = torch.mean(loss_send)
        self._prev_losses_mean.append(avg_loss)

        if self.epoch < self.warmup_epochs:
            self.global_skip = 0
            self.local_skip = 0
            self.batches_to_wait = 0
            print0("\t\t", self.global_skip, self.local_skip, self.batches_to_wait)
            return
        elif 4 == self.epoch:
            self.global_skip = 4
            self.local_skip = 1
            self.batches_to_wait = 1
            print0("\t\t", self.global_skip, self.local_skip, self.batches_to_wait)
            self._prev_losses_mean = []

        if self.epoch >= self.total_epochs - 5:
            self.global_skip = 0
            self.local_skip = 0
            self.batches_to_wait = 0
            print0("\t\t", self.global_skip, self.local_skip, self.batches_to_wait)
            return

        # epochs_to_wait = 3
        if len(self._prev_losses_mean) < self.epochs_to_wait:
            return
        means = torch.tensor(self._prev_losses_mean)
        diff = abs(means[-1] - means[-1 * self.epochs_to_wait])
        stable = True if diff <= 0.075 else False
        # TODO: add something for when the loss is *increasing*?
        if stable and self.global_skip > 1:
            # drop gs by factor of 2
            self.global_skip //= 2
            self.local_skip //= 2
            self.batches_to_wait //= 2
            self.epochs_to_wait += 1
            self._prev_losses_mean = []
            print0("dropping skips, loss stable")
            if self.global_skip > 0:
                if self.batches_to_wait == 0:
                    self.batches_to_wait = 1
                if self.local_skip == 0:
                    self.local_skip = 1
        elif self.global_skip == 1 and stable:
            self.global_skip = 8
            self.local_skip = 2
            self.batches_to_wait = 3  # 2
            self._prev_losses_mean = []
            self.epochs_to_wait = 3

        print0("\t\t", self.global_skip, self.local_skip, self.batches_to_wait)

    def add_scaler(self, scaler):
        self.scaler = scaler
        self.amp = True

    def step(self):
        # TODO: raise error is last batch is not set
        # collect the parameters from the current batch -> save + (non?)blocking send
        # test for receive from last batch,
        #   if yes: receive, update parameters with rcved stuff
        # copy and send the parameter dictionary
        if self.amp:
            self.scaler.step(self.lcl_optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
        elif self.scheduler is None:
            self.lcl_optimizer.step()
            #li = []
            #for p in self.module.parameters():
            #    li.append(p.mean().item())
            #print0(torch.mean(torch.tensor(li)).item())
            #return
        else:
            self.scheduler.step()
        batch = self.current_batch
        next_batch = batch + 1
        gs = self.global_skip
        ls = self.local_skip

        gmod = batch % gs if gs > 0 else 0
        lmod = batch % ls if ls > 0 else 0

        batches_to_wait = self.batches_to_wait
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
            # do nothing on these batches
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
        current_comm = self.reduced_comms[self._send_mod]
        current_ranks = self.reduced_ranks[self._send_mod]

        if self.comm.rank in current_ranks:
            self._global_send_update(current_comm, batches_to_wait)

        if self.batches_to_wait != 0:
            # update parameters from the last sending (if there)
            self._update_parameters()  # -> splits off irrelevant ranks
            # needs to happen on all ranks:
            self._local_torch_param_update(self._send_mod_m1)

        if self.current_batch == self.last_batch or self.batches_to_wait == 0:
            # todo: abstract last batch?
            # receive the sent data to sync params across all ranks
            #if self.comm.rank in current_ranks:
            #    if len(self._prev_params) > 1:
            #        raise ValueError(f"length of previous params > 1! {len(self._prev_params)}")
            #    prev_params = self._prev_params.pop(0)
            #    shapes = prev_params[2]
            #    # factor = 1.0 / float(len(current_ranks))
            #    prev_params[0].Wait()
            #    rcv_params = prev_params[1] / float(len(current_ranks))  # * factor
            #    for name, param in self.module.named_parameters():
            #        if param.requires_grad:
            #            # rcv_params = prev_params[1]
            #            # param *= 0
            #            param[:] = (
            #                rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
            #            )
            #    self._prev_params = []
            if self.comm.rank in current_ranks:
                if len(self._prev_params) > 1:
                    raise ValueError(f"length of previous params > 1! {len(self._prev_params)}")
                prev_params = self._prev_params.pop(0)
                shapes = prev_params[2]
                if not self.split:
                    prev_params[0].Wait()
                    #print('\t', prev_params[1].mean(), prev_params[1].data_ptr())
                    rcv_params = prev_params[1] / float(len(current_ranks))
                    #print('\t\trcv params', rcv_params.mean())
                    for name, param in self.module.named_parameters():
                        if param.requires_grad:
                            param[:] = (
                                rcv_params[shapes[name][1]]
                                .reshape(shapes[name][0])
                                .to(shapes[name][2])
                            )
                else:
                    ind1 = 0
                    print("before first wait", prev_params[0][0])
                    #current_comm.Waitall(prev_params[0])
                    # prev_params[0][0].handle.Waitall([p.handle for p in prev_params[0]])
                    prev_params[0][0].handle.Waitany([p.handle for p in prev_params[0]])
                    del prev_params[0][0]
                    rcv_params = prev_params[1][ind1] / float(len(current_ranks))
                    print("after first wait")
                    for name, param in self.module.named_parameters():
                        if param.requires_grad:
                            #print0(shapes[name][1].stop, len(rcv_params))
                            while shapes[name][1].stop > len(rcv_params):
                                #print(ind1)
                                ind1 += 1
                                prev_params[0][0].Waitany()
                                del prev_params[0][0]
                                new_rcv_params = prev_params[1][ind1] / float(len(current_ranks))
                                rcv_params = torch.cat((rcv_params, new_rcv_params))
                            param[:] = (
                                rcv_params[shapes[name][1]]
                                .reshape(shapes[name][0])
                                .to(shapes[name][2])
                            )
                del rcv_params
                self._prev_params = []
            else:
                if len(self._prev_params) > 0:
                    raise ValueError(
                        f"DEBUG: OFF RANKS! len(prev_params) > 0! {len(self._prev_params)}"
                        f" batch number {self.current_batch}"
                    )
            #self.comm.Barrier()
            #print("here", self.current_batch)
            self._local_torch_param_update(self._send_mod)

            self._send_mod_m1 = None

            if self.current_batch == self.last_batch:
                self._send_mod = 0
                self.epoch += 1
                self.current_batch = 0
            else:
                self.current_batch += 1
                self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0
            li = []
            for p in self.module.parameters():
                li.append(p.mean().item())
            #print0(torch.mean(torch.tensor(li)).item())
        else:
            self.current_batch += 1
            self._send_mod_m1 = self._send_mod
            self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0

    @torch.no_grad()
    def _global_send_update2(self, current_comm, batches_to_wait):
        # pack and send the data required for a global synchronization
        op = MPI.SUM
        cast = False
        if self.global_skip < 1:
            op = self.mpi_sum_bfloat
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
        recv = torch.zeros_like(params)
        new_wait = current_comm.Iallreduce(params, recv, op)  # mpi_sum_f16) #
        self._prev_params.append([new_wait, recv, shapes, batches_to_wait])
        return new_wait
    
    @torch.no_grad()
    def _global_send_update_zeros(self, current_comm, batches_to_wait, op, cast):
        # pack and send the data required for a global synchronization
        # todo: jit loop?
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
        recv = torch.zeros_like(params)
        new_wait = current_comm.Iallreduce(params, recv, op)  # mpi_sum_f16) #
        self._prev_params.append([new_wait, recv, shapes, batches_to_wait])
        return new_wait




    @torch.no_grad()
    def _global_send_update(self, current_comm, batches_to_wait):
        # pack and send the data required for a global synchronization
        op = MPI.SUM
        cast = False
        if self.global_skip < 1:
            op = mpi_sum_bfloat
            cast = True
    
        param_dict, shapes = self._create_param_dict_n_shapes()
        #params = torch.zeros(
        #    self._param_send_buffer_shape,
        #    device=self.device,
        #    dtype=torch.bfloat16 if cast else None,
        #)

        params = self.__pack_data(param_dict, self._param_send_buffer_shape, cast, str(self.device)).contiguous()
        if params.isnan().sum():
            raise ValueError(f"{params.isnan().sum()} NaNs in `params` shit be fucked?")
        #print(f"\tbefore send {params.mean()}, end message")

        if self.split or params.numel() > self.split_val:
            # TODO: make this logic happen when the model is added
            self.split = True
            num_splits = math.ceil(len(params) / self.split_val)
            splits = [self.split_val] * (num_splits - 1)
            rem = len(params) - (self.split_val * (num_splits - 1))
            # first one will be smaller then the rest (the raminder is first
            splits = [rem] + splits
            self.split_inds = splits
            params_list = [None] * num_splits
            prev = 0
            waits = [None] * num_splits
            for s in range(num_splits):
                # need to slice the params at the split points
                #params_list[s] = torch.empty_like(params[prev : splits[s] + prev])
                params_list[s] = params[prev : splits[s] + prev]
                #print(params_list[s], prev, splits[s] + prev)
                #if s == 0:
                #print("before blocking test sneding", op)
                #time_test = time.perf_counter()
                #current_comm.Allreduce(MPI.IN_PLACE, params_list[s], op)
                #print("test blocking send", time.perf_counter() - time_test)

                #prev = splits[s]
                #print("send size", lp_par.shape, lp_par.dtype, prev, splits[s] + prev)
                prev += splits[s]
                #params_list.append(lp_par)
                # TODO: remove in place call from this
                waits[s] = current_comm.Iallreduce(MPI.IN_PLACE, params_list[s], op)
                #waits.append(lp_w) #current_comm.Iallreduce(MPI.IN_PLACE, lp_par, op))
                #print("before wait in send", waits[s])
                #w.Wait()
                #print(w)
                #del w
                #waits[s].Wait()
                #print("after wait", s)#, waits[s])
                #waits[s] = w # None
            self._prev_params.append([waits, params_list, shapes, batches_to_wait])
            #print("before blocking test sneding", op)
            #time_test = time.perf_counter()
            #current_comm.Allreduce(MPI.IN_PLACE, lp_par, op)
            #print("test blocking send", time.perf_counter() - time_test)
            #del params
        else:
            #params += 0.
            params.mean()
            new_wait = current_comm.Iallreduce(MPI.IN_PLACE, params, op)  # mpi_sum_f16) #
            self._prev_params.append([new_wait, params, shapes, batches_to_wait])
            #print(f"\tafter send {params.mean()}, end message")
        #del send

    def _create_param_dict_n_shapes(self):
        """
        create the shape and param dictionary used for sending parameters around the MPI world.
        this will also define the buffer size if it was not previously defined.
        """
        if self.shapes is not None:
            #param_dict = {n:p for n, p in self.module.named_parameters()}
            return self.param_dict, self.shapes
        # else:
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

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def __pack_data(iter_dict: Dict[str, torch.Tensor], buffer_shape: int, cast: bool, dev: str):
        """ jitted loop to pack the data into params to be sent"""
        st = 0
        params = torch.zeros(
            buffer_shape,
            device=dev,
            dtype=torch.bfloat16 if cast else None,
        )

        for name, param in iter_dict.items():
            if param.requires_grad:
                # flatten and prep the data for sending
                p = torch.flatten(param)
                if cast:
                    p = p.to(torch.bfloat16)
                params[st : st + param.numel()] = p
                st += param.numel()
        #print("w/in packing", params.mean(), params.isnan().sum())
        return params  #.contiguous()

    @torch.no_grad()
    def _local_torch_param_update(self, mod_hold_pr):
        # TODO: jit this?
        # send the globally updated gradients from `mod_hold_pr` to the other local processes
        if torch.distributed.is_initialized():
            snds = {}
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    #if torch.distributed.get_rank() != mod_hold_pr:
                    #    param2 = torch.zeros_like(param.data)
                    #else:
                    #    #print("send data", param.data.mean())
                    #    param2 = param.data.clone()
                    snds[name] = torch.distributed.broadcast(param, mod_hold_pr, async_op=True)  # default is SUM
                    #w = torch.distributed.broadcast(param2, mod_hold_pr, async_op=True)
                    #snds[name] = (w, param2)
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    snds[name].wait()
                    #param.data = snds[name][1]
            #print(param.mean())

            del snds

    @torch.no_grad()
    def _update_parameters(self):
        # wait for the global sync data and update on the selected rank, requires local torch param update after
        if self._send_mod_m1 is None:
            return
        prev_ranks = self.reduced_ranks[self._send_mod_m1]
        if self.comm.rank not in prev_ranks:
            # receive previous gradients
            return
        if len(self._prev_params) == 0:
            # if no old gradients, return without doing anything
            return

        prev_params = self._prev_params.pop(0)
        batches_between = float(prev_params[3])
        shapes = prev_params[2]
        # add the weighted average to param
        numer = batches_between * 2.0 if batches_between > 0.0 else 1.0
        denom = float(len(prev_ranks) + numer)
        factor = numer / denom
        if not self.split:
            prev_params[0].Wait()
            rcv_params = prev_params[1] / denom
            # todo: jit the parameter setting
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    update = (
                        rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                    )
                    # NOTE: update here is the sum of the params across the processes
                    param *= factor
                    param += update  # / denom
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
                        prev_params[0][0]
                        new_rcv_params = prev_params[1][ind] / denom
                        rcv_params = torch.cat((rcv_params, new_rcv_params))
                    update = (
                        rcv_params[shapes[name][1]].reshape(shapes[name][0]).to(shapes[name][2])
                    )
                    # NOTE: update here is the sum of the params across the processes
                    param *= factor
                    param += update  # / denom

    # @staticmethod
    # @torch.jit.script
    # def __set_params_after_recv(param_dict, factor):
    # todo: the slice to get the proper parameter numbers is a slice and
    #       cannot be passed into torch's jit function, same with dtype

    def zero_grad(self) -> None:
        """
        Reset gradients of optimizer's params.
        """
        # reset view onto params in order to reset all gradients
        self.lcl_optimizer.param_groups[0]["params"] = self.params_ref[:]
        self.lcl_optimizer.zero_grad()
