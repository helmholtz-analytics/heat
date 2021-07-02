"""
MPI enabled data parallel optimizers
"""

import inspect
import math
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as tDDP
from typing import Union, List, Tuple, Dict

from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD
from .utils import DetectMetricPlateau


__all__ = ["DataParallelOptimizer", "DASO", "DASO2"]


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


class DASO2:
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
        self.num_layers = len(self.module._parameters)

        print("Finished DASO init")

    @torch.no_grad()
    def save_master_model_state(self):
        self.last_synced_model = self.module._parameters.copy()

    @torch.no_grad()
    def cos_dist_and_sum_diff_calc(self):
        """
        calculate the cosine difference and sum difference of the current model state

        Returns
        -------

        """
        cos_dists = []
        sum_diffs = []
        layer = 0
        for p_new, p_old in zip(self.module.parameters(), self.last_synced_model.parameters):
            # todo: more efficient to flatten and do it properly, or to just do it then take the
            #   average?
            pnf = p_new.flatten()
            pof = p_old.flatten()
            cos_sim = self.cos_sim(pnf, pof)
            cos_dists.append(cos_sim)
            sdif = torch.sum(pnf - pof)
            sum_diffs.append(sdif)
            print(layer, cos_sim, sdif)
            layer += 1

    def zero_grad(self) -> None:
        """
        Reset gradients of local optimizer's parameters.
        """
        # reset view onto params in order to reset all gradients
        self.local_optimizer.param_groups[0]["params"] = self.params_ref[:]
        self.local_optimizer.zero_grad()

    def stop_local_sync(self) -> None:
        """
        Stop local synchronizations for the next batches
        """
        if not isinstance(self.module, tDDP) or not self.module.require_backward_grad_sync:
            # this has no effect if the module is not locally distributed in torch
            return
        self.module.require_backward_grad_sync = False