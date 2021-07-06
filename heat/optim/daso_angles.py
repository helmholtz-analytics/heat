"""
MPI enabled data parallel optimizers
"""

import inspect
import math
import torch
import torch.distributed
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as tDDP
from typing import Union, List, Tuple, Dict

from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD
from .utils import DetectMetricPlateau


__all__ = ["DASO_angles"]


def __sum_f16_cb(buffer_a, buffer_b, _):
    # MPI custom sum function to use torch.half
    tens_a = torch.HalfTensor().set_(torch.HalfStorage.from_buffer(buffer_a, "native"))
    tens_b = torch.HalfTensor().set_(torch.HalfStorage.from_buffer(buffer_b, "native"))
    tens_b += tens_a
    nelem = torch.prod(torch.tensor(tens_b.shape)).item()
    new_buff = MPI.memory.fromaddress(tens_b.data_ptr(), nbytes=tens_b.element_size() * nelem)
    buffer_b[:] = new_buff


def __sum_bfloat_cb(buffer_a, buffer_b, _):
    # MPI custom sum function to use torch.bfloat16
    tens_a = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_a, "native"))
    tens_b = torch.BFloat16Tensor().set_(torch.BFloat16Storage.from_buffer(buffer_b, "native"))
    tens_b += tens_a
    nelem = int(tens_b.numel())
    new_buff = MPI.memory.fromaddress(tens_b.data_ptr(), nbytes=nelem * tens_b.element_size())
    buffer_b[:] = new_buff


# create new MPI OPs
mpi_sum_f16 = MPI.Op.Create(__sum_f16_cb, commute=True)
mpi_sum_bfloat = MPI.Op.Create(__sum_bfloat_cb, commute=True)


class DASO_angles:
    def __init__(
        self,
        local_optimizer: torch.optim.Optimizer,
        module,
        total_epochs: int,
        comm: MPICommunication = MPI_WORLD,
        cooldown_epochs: int = 4,
        scheduler: torch.optim.lr_scheduler = None,
        downcast_type: torch.dtype = torch.bfloat16,
        verbose: bool = False,
    ):  # noqa: D107
        # check dtypes
        # frame = inspect.currentframe()
        # init_args = inspect.getargvalues(frame)[3]
        # self.__init_checktypes(init_args)

        self.cast_dtype = downcast_type
        if downcast_type == torch.bfloat16:
            self.cast_fn = mpi_sum_bfloat
        elif downcast_type == torch.half:
            self.cast_fn = mpi_sum_f16
        else:
            self.cast_fn = MPI.SUM

        self.comm = comm
        self.verbose = verbose
        self.local_optimizer = local_optimizer
        self.params_ref = local_optimizer.param_groups[0]["params"]
        # reference of optimizer's params
        self.scheduler = scheduler

        # todo: does it make sense to do local MPI groups again? (see original DASO)

        self.current_batch, self.last_batch = 0, None

        self.epoch = 0

        self.cooldown_epochs = cooldown_epochs
        self.total_epochs = total_epochs

        # used in the sending of the params

        self.amp = False

        self.last_synced_model = None
        self.cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.cosine_dists = []
        self.sum_diffs = []

        self.module = module
        layers = 0
        for l in module.parameters():
            if l.requires_grad():
                layers += 1
        self.num_layers = layers

        self.angles_wait = None
        self.cos_angles = None

        self.sent_layers = OrderedDict()

        self.angle_buffer = None  # torch.zeros(layers, )

        print("Finished DASO init")

    def add_scaler(self, scaler: torch.cuda.amp.GradScaler) -> None:
        """
        Create a reference to torch's `torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ used in torch's automatic mixed
        precision.

        Parameters
        ----------
        scaler: torch.cuda.amp.GradScaler
            the gradient scaler to be used
        """
        self.scaler = scaler
        self.amp = True

    @torch.no_grad()
    def save_master_model_state(self):
        self.last_synced_model = self.module._parameters.copy()

    # @torch.jit.script  TODO: jit this function for speed
    @torch.no_grad()
    def cos_dist(self):
        """
        calculate the cosine difference and sum difference of the current model state

        Returns
        -------
        a list/torch tensor if the
        """
        cos_dists = []

        # layer = 0
        for p_new, p_old in zip(self.module.parameters(), self.last_synced_model):
            # todo: more efficient to flatten and do it properly, or to just do it then take the
            #   average?
            if p_new.requires_grad:
                pnf = p_new.flatten()
                pof = p_old.flatten()
                cos_sim = self.cos_sim(pnf, pof)
                cos_dists.append(cos_sim)
            # layer += 1
            dev = p_new.device
        # todo: what dtype should this be? does it really matter since its small?
        return torch.tensor(cos_dists, device=dev)

    def zero_grad(self) -> None:
        """
        Reset gradients of local optimizer's parameters.
        """
        # reset view onto params in order to reset all gradients
        self.local_optimizer.param_groups[0]["params"] = self.params_ref[:]
        self.local_optimizer.zero_grad()

    # def stop_local_sync(self) -> None:
    #     """
    #     Stop local synchronizations for the next batches
    #     """
    #     if not isinstance(self.module, tDDP) or not self.module.require_backward_grad_sync:
    #         # this has no effect if the module is not locally distributed in torch
    #         return
    #     self.module.require_backward_grad_sync = False

    def step(self):

        # cos distance after the step function?? # todo: after or before the step?

        if self.amp:
            self.scaler.step(self.local_optimizer)
            # todo: add something to tell if the grads have infs or nans
            # Updates the scale for next iteration.
            self.scaler.update()
        elif self.scheduler is None:
            self.local_optimizer.step()
        else:
            self.scheduler.step()

        if self.last_synced_model is None:
            # should only happen on the first batch
            hold = []
            for par in self.module.named_parameters():
                hold.append(par)
            self.last_synced_model = tuple(hold)
            return

        # this wont get here if the top loop is true
        if self.angles_wait is None:
            # if there is not wait object for the angles then calc and send them
            self.cos_angles = self.cos_dist()
            self.angles_wait = self.comm.Iallreduce(MPI.IN_PLACE, self.cos_angles, MPI.SUM)
            return  # nothing else happens in this step now

        # this is the place to test if there are parameters being sent alreadyf

        # else: wait for the angles if they were sent last time
        self.angles_wait.Wait()
        self.cos_angles /= float(self.comm.size)

        # todo: skip the next bit if some params are already sent

        # cos angles is not the average
        # todo: average difference or max?
        # determine which layers need to be sent
        # TODO: THIS NUMBER IS WHAT DETERMINES WHAT IS SENT: need to tune/benchmark/exp
        layers_to_send = self.cos_angles > 0.005

        # todo: this should be outside of the gradients
        for num, layer in enumerate(self.module.parameters()):
            if layers_to_send[num]:
                # send the layer for this one
                # self.sent_layer_parameters -> [layer, wait object, params] for each layer being sent
                param = layer.clone().detach()  # clone or create an empty buffer?
                lwait = self.comm.Iallreduce(MPI.IN_PLACE, param, MPI.SUM)
                self.sent_layers[num] = [lwait, param]

        # have the cosine
