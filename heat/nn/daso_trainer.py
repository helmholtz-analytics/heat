"""
MPI enabled data parallel optimizers
"""

import inspect
import math
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as tDDP
from typing import Union, List, Tuple, Dict

from ..core.communication import MPICommunication
from ..core.communication import MPI
from ..core.communication import MPI_WORLD
from ..optim.utils import DetectMetricPlateau

try:
    import nvidia.dali as dali
    has_dali = True
except ImportError:
    has_dali = False


import gc

__all__ = ["DASOTrainer"]


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


class DASOTrainer(nn.Module):
    r"""
    Optimizer wrapper to use the Distributed Asynchronous and Selective Optimization (DASO) method.

    This optimizer uses a local torch optimizer combined with the :func:`nn.DataParallelMultiGPU <heat.nn.data_parallel.DataParallelMultiGPU>`
    to create local DPNNs on each node consisting of the GPUs on each node. Then those networks communicate
    globally with MPI groups, each of which has a single GPU on each node.

    DASO uses both local and global synchronization operations. Local synchronization operations are intended to be
    done very frequently while global synchronizations are conducted asynchronously as the next batches are
    computed.

    This implementation requires that all nodes have the name number of GPUs.

    There are four phases to training:

        1. initialization: steps 1 to 8 below
        2. Warmup phase: blocking averaging update occurs for global synchronization step
        3. Cycling phase: for the global synchronization, the data is sent after a number of batches. the number of batches between synchronizations is referred to as `global_skips`. After the data is sent a number of batches pass before it is received (`batches_to_wait`). both of these cycle downward from `max_global_skips` for the global skips and 1/4th this value for `batches_to_wait`. When both values are equal to 1 and the loss is stable it will be reset to the initial values, then will decay again.
        4. Cooldown phase: blocking averaging update occurs for global synchronization step

    As example usage of this can be found in `heat/examples/nn/imagenet-DASO.py <https://github.com/helmholtz-analytics/heat/blob/504-docstring-formatting/examples/nn/imagenet-DASO.py>`_.

    The recommended checklist for using this class is as follows:

        1. initialize the local PyTorch process group and set the default device of the local GPUs.
        2. define the torch network
        3. define the `local_optimizer` -> a torch optimizer of your choice (tested with SGD)
        4. optional, choose a learning rate scheduler. This is only for those learning rates which will also step the optimizer
        5. initialize DASO with the local optimizers and parameters
        6. initialize :func:`nn.DataParallelMultiGPU <heat.nn.data_parallel.DataParallelMultiGPU>` with the torch network and DASO
        7. If using automatic mixed precision (:class:`torch.cuda.amp`), initialize the gradient scaler and add it to DASO (:func:`add_scaler`)
        8. ensure that the DataLoaders evenly distribute the data between all the processes. This can be done by using the `torch.utils.data.distributed.DistributedSampler <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`_ with the `num_replicas` and `rank` parameters
        9. call `daso_optimizer.epoch_loss_logic(training_loss)` at the end of
        10. set the number of batches per epoch (`daso_optimizer.last_batch = number_of_batches`)
        11. ensure that the step function used in training is that of the DASO optimizer

    Parameters
    ----------
    local_optimizer: torch.optim.Optimizer
        This optimizer handles the optimization of the local NN. Example: `torch.optim.SGD`. \n
        This can be any optimizer, although tests were only completed with SGD. Other optimizers may show
        unexpected behavior.
    total_epochs: int
        The total number of epochs for training. Needed to determine when to enter the cooldown phase.
    comm: MPICommunication, optional
        The MPI communicator to use for training. \n
        Default: :func:`MPI_WORLD <heat.core.comm.MPI_WORLD>`
    warmup_steps: int, optional
        The number of epochs to complete with a blocking averaging operation after each batch before entering
        the cycling phase.\n
        Default: 4
    cooldown_epochs: int, optional
        The number of epochs with blocking averaging operations after each batch at the end of training.\n
        Default: 4
    scheduler: torch.optim.lr_scheduler, optional
        Local PyTorch learning rate scheduler. This must be used in the case that the scheduler's `step` function
        is supposed to be called instead of the optimizer's `step` function.\n
        Default: None
    stability_level: float, optional
        This can be viewed as the percent change threshold that the loss must exceed to be judged as improving.
        When the loss is within this percent change for 2 epochs, then it is judged as stable.\n
        Default: 0.05
    max_global_skips: int, optional
        The maximum number of batches between the beginning of a global synchronization process.\n
        Default: 8
    sending_chunk_size: int, optional
        During the global synchronization step, the network parameters are split into chunks of data to overlap
        communication and computation. This value is the maximum chunk size.\n
        Default: 10,000,000
    downcast_type: torch.dtype, optional
        Options: [torch.bfloat16, torch.half, torch.float]
        When the network parameters are sent during the global synchronization step, they are cast down to
        a smaller dtype, by default this is `torch.bfloat16`. Smaller torch dtypes are not implemented.
        torch.bfloat16.\n
        Default: torch.bfloat16
    use_mpi_groups: bool, optional
        Use MPI groups to divide the global communicator. If True, use MPI GROUPs, otherwise, use MPI SPLIT.\n
        Default: True
    skip_reduction_factor: int, optional
        How much to reduce the global/local skips by when the loss has stabilized.\n
        Default: 2
    local_skip_factor: int, optional
        How many local skips occur per global skip, i.e. number of local skips = global_skips // local_skip_factor.\n
        Default: 4
    verbose: bool, optional
        If true, print out a collection of debug messages.\n
        Default: False
    """

    def __init__(
        self,
        model: nn.Module,
        local_optimizer: torch.optim.Optimizer,
        cooldown_threshold: float,
        cooldown_up_down: str = "lt",
        ddp_kwargs: dict = None,
        comm: MPICommunication = MPI_WORLD,
        warmup_batches: int = 500,
        scheduler: torch.optim.lr_scheduler = None,
        stability_level: float = 0.05,
        max_global_skips: int = 4,
        sending_chunk_size: int = 20_000_000,  # 10 MB per bucket
        downcast_type: torch.dtype = torch.bfloat16,
        use_mpi_groups: bool = True,
        skip_reduction_factor: int = 2,
        local_skip_factor: int = 4,
        staleness_adjustment: float = 2.0,
        verbose: bool = False,
    ):  # noqa: D107
        # check dtypes
        # self.enable_jit =
        super(DASOTrainer, self).__init__()

        self.module, self.lcl_model, self.lcl_group = None, None, None
        frame = inspect.currentframe()
        init_args = inspect.getargvalues(frame)[3]
        self.__init_checktypes(init_args)

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

        rank = self.comm.rank
        loc_gpus = torch.cuda.device_count()
        # this assumes that there are an equal number of GPUs per node,
        #   if a change is desired a comm her to find the lowest number would work for this, however
        #   a change would also need to be made in heat.nn.DataParallelMultiGPU
        self.loc_gpus = loc_gpus
        local_rank = rank % loc_gpus
        self.local_skip = 1
        if loc_gpus < 1:
            raise RuntimeError(f"Not enough GPUs! local gpus: {loc_gpus}")

        # create MPI Groups
        base_loc_ranks = list(range(0, self.comm.size, loc_gpus))
        reduced_comms, reduced_ranks = [], []
        for i in range(loc_gpus):
            lp_ranks = [j + i for j in base_loc_ranks]
            if use_mpi_groups:
                new_group = self.comm.group.Incl(lp_ranks)
                new_comm = self.comm.Create_group(new_group)
                reduced_comms.append(MPICommunication(new_comm))
            else:
                color = 111 + i if rank in lp_ranks else 222 + i
                key = 0 + i if rank in lp_ranks else 444 + i
                reduced_comms.append(MPICommunication(self.comm.Split(color, key)))
            reduced_ranks.append(tuple(lp_ranks))
        self.reduced_comms, self.reduced_ranks = reduced_comms, reduced_ranks
        self.base_loc_ranks = base_loc_ranks
        self.device = "cuda:" + str(local_rank)
        torch.cuda.set_device(device=self.device)

        self._ddp_inits(model, ddp_kwargs, enable_graph=True)

        self.batch, self.last_batch = 0, None

        self._prev_params = []
        self.epoch = 0
        self._send_mod, self._send_mod_m1 = 0, None

        self.global_skip = 0
        self.local_skip = 0
        self.batches_to_wait = 0
        self.max_gs = max_global_skips

        self.warmup_steps = warmup_batches
        self.cooldown = False
        self.cooldown_threshold = cooldown_threshold
        self.cooldown_up_down = 1 if cooldown_up_down == "gt" else -1

        # used in the sending of the params
        self._param_send_buffer_size = None
        self.param_dict, self.shapes = None, None
        self._param_send_shp = None
        self.split = None

        self.skip_reduction_factor = skip_reduction_factor
        # the local_skip_factor is the factor by which the global skips are divided by initially
        # and upon reset
        self.local_skip_factor = local_skip_factor

        self.stale_weighting = staleness_adjustment

        self.stability = DetectMetricPlateau(patience=2, threshold=stability_level)

        self._gs8_waits = 3
        self._gs8_waited = 0

        self.split_val = sending_chunk_size

        # TODO: its possible that the split indexes could be used to avoid the concatenating method used currently
        self.split_inds = None
        self.amp = False

        # check for MP lamb
        try:
            from apex.optimizers.fused_mixed_precision_lamb import FusedMixedPrecisionLamb
            have_mp_lamb = True
        except ImportError:
            have_mp_lamb = False

        if have_mp_lamb and isinstance(self.optimizer, FusedMixedPrecisionLamb):
            self.enable_mp_lamb = True
        else:
            self.enable_mp_lamb = False

        self.enable_gpu_scheduler = (self.enable_distributed_lamb or self.enable_mp_lamb)

        # we need this for distlamb
        if self.enable_gpu_scheduler:
            # we need that in order for it to work with async graph capture
            self.lr_cpu = torch.tensor([0.], dtype=torch.float32, device='cpu').pin_memory()

        if self.enable_mp_lamb and (self.scheduler is not None):
            # we need to fix what the scheduler screwed up: initializing the
            # former removed the lr from the gpu:
            for group, lr_gpu in zip(self.optimizer.param_groups, self.scheduler.group_lrs_gpu):
                group["lr"] = lr_gpu
            # we need to update the backups as well
            self.scheduler.group_lrs_backup = [x.clone() for x in self.scheduler.group_lrs_gpu]

        self.entering_cooldown = False
        self.print0("Finished DASO init")

    def _ddp_inits(self, model, ddp_kwargs, enable_graph=True):
        # create new torch group for local models
        # get the rank which is rank 0 on every node
        node_start_rank = (self.comm.size // self.loc_gpus) * self.loc_gpus
        # create the list of the next four
        node_local_ranks = list(range(node_start_rank, node_start_rank + self.loc_gpus))
        self.lcl_group = dist.new_group(node_local_ranks)

        # self.rank = self.comm.rank
        local_rank = self.comm.rank % self.loc_gpus
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device=self.device)

        self.models = {}
        self.streams = {}

        if enable_graph:
            self.streams["local"] = torch.cuda.Stream()
            with self.streams["local"]:
                self.models["local"] = tDDP(
                    model, device_ids=[local_rank], process_group=self.lcl_group
                )

            self.streams["global"] = torch.cuda.Stream()
            with self.streams["global"]:
                self.models["global"] = tDDP(model, device_ids=[local_rank], **ddp_kwargs)
        else:
            self.streams["local"] = None
            self.models["local"] = tDDP(
                model, device_ids=[local_rank], process_group=self.lcl_group
            )
            self.streams["global"] = None
            self.models["global"] = tDDP(model, device_ids=[local_rank], **ddp_kwargs)

        self.jit_scriptable = True
        for m in model.modules():
            if hasattr(m, "jit_scriptable"):
                self.jit_scriptable = self.jit_scriptable and m.jit_scriptable
                if not self.jit_scriptable:
                    break

    def _compile_models(self, input_shape):
        # exit if we do not compile
        if not self.enable_jit:
            return
        for lg in ["local", "global"]:
            # set model to train just to be sure
            self.models[lg].train()

            # input example
            input_example = torch.zeros(
                (self.batch_size, *input_shape), dtype=torch.float32, device=self.device
            )
            input_example.normal_()

            # convert to half if requested
            if self.force_fp16:
                input_example = input_example.half()

            # we need to convert to NHWC if necessary
            if self.enable_nhwc:
                input_example = input_example.contiguous(memory_format=torch.channels_last)

            # compile the model
            with amp.autocast(enabled=self.enable_amp):
                # extract the right thing to jit
                # if isinstance(model, DistributedDataParallel): -> always DDP here!
                # GBN is not scriptable, we need to workaround here
                if self.jit_scriptable:
                    self.models[lg].module = torch.jit.script(self.models[lg].module)
                else:
                    self.models[lg].module = torch.jit.trace(
                        self.models[lg].module, input_example, check_trace=False
                    )

                # the criterion is always scriptable
                self.criterion = torch.jit.script(self.criterion)

    def _warmup(self, input_shape, label_shape, num_warmup=1):
        # create input:
        input_example = torch.zeros(
            (self.batch_size, *input_shape), dtype=torch.float32, device=self.device
        )
        input_example.normal_()
        label_example = torch.zeros(
            (self.batch_size, *label_shape), dtype=torch.int64, device=self.device
        )

        for lg in ["local", "global"]:
            # set model to train just to be sure
            self.models[lg].train()

            # extract or create stream
            # get the local and global streams

            # convert to half if requested
            if self.force_fp16:
                input_example = input_example.half()

            # we need to convert to NHWC if necessary
            if self.enable_nhwc:
                input_example = input_example.contiguous(memory_format=torch.channels_last)

            # wait for ambient stream before starting capture
            self.streams[lg].wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self.streams[lg]):
                # warmup:
                for _ in range(num_warmup):
                    self.optimizer.zero_grad()

                    with amp.autocast(enabled=self.enable_amp):
                        output = self.models[lg](input_example)
                        loss = self.criterion(output, label_example)

                    self.gscaler.scale(loss).backward()

            # FIXME: need to change the current stream multiple times??? what do?
            torch.cuda.current_stream().wait_stream(self.streams[lg])

    def capture_models(self, input_shape, label_shape, num_warmup=1, graph_pool=None):
        # create input:
        self.static_input = torch.zeros(
            (self.batch_size, *input_shape), dtype=torch.float32, device=self.device
        )
        self.static_input.normal_()
        self.static_label = torch.zeros(
            (self.batch_size, *label_shape), dtype=torch.int64, device=self.device
        )
        # convert to half if requested
        if self.force_fp16:
            self.static_input = self.static_input.half()

        # we need to convert to NHWC if necessary
        if self.enable_nhwc:
            self.static_input = self.static_input.contiguous(memory_format=torch.channels_last)

        self.graphs = {"local": None, "global": None}

        self._capture(local_global="local", num_warmup=num_warmup, graph_pool=graph_pool)
        self._capture(local_global="global", num_warmup=num_warmup, graph_pool=graph_pool)

    def _capture(self, local_global, num_warmup=1, graph_pool=None):
        # todo: what do to about the graph pool??
        # exit if we do not capture
        if not self.enable_graph:
            return

        # capture both models -> one for the local one for global
        # set model to train just to be sure

        self.models[local_global].train()

        # wait for ambient stream before starting capture
        self.streams[local_global].wait_stream(torch.cuda.current_stream())

        # enter stream context
        with torch.cuda.stream(self.streams[local_global]):
            # warmup:
            for _ in range(num_warmup):
                self.optimizer.zero_grad()

                # FW pass
                with amp.autocast(enabled=self.enable_amp):
                    output = self.models[local_global](self.static_input)
                    loss = self.criterion(output, self.static_label)

                    # BW pass
                self.gscaler.scale(loss).backward()

                # mixed precision lamb postprocessing
                if self.enable_mp_lamb:
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    lr = self.scheduler.group_lrs_gpu[0]
                    self.scheduler.step_gpu()
                    for idg, group in enumerate(self.optimizer.param_groups):
                        group["lr"].copy_(self.scheduler.group_lrs_gpu[idg])

            # sync streams
            self.streams[local_global].synchronize()

            # clean up
            if num_warmup > 0:
                del output, loss
                if self.enable_gpu_scheduler:
                    del lr
            gc.collect()
            torch.cuda.empty_cache()

            # create graph

            self.graphs[local_global] = torch.cuda.CUDAGraph()

            # zero grads before capture:
            self.model.zero_grad(set_to_none=True)

            # start capture
            if graph_pool is not None:
                self.graphs[local_global].capture_begin(pool=graph_pool)
            else:
                self.graphs[local_global].capture_begin()

            # FW pass
            with amp.autocast(enabled=self.enable_amp):
                self.static_output = self.model(self.static_input)
                self.static_loss = self.criterion(self.static_output, self.static_label)

            # BW pass
            self.gscaler.scale(self.static_loss).backward()

            # should also be done

            # mp lamb postprocessing
            if self.enable_mp_lamb:
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
                self.static_lr = self.scheduler.group_lrs_gpu[0]
                self.scheduler.step_gpu()
                for idg, group in enumerate(self.optimizer.param_groups):
                    group["lr"].copy_(self.scheduler.group_lrs_gpu[idg])

            # end capture
            self.graphs[local_global].capture_end()

            # sync up
            self.streams[local_global].synchronize()

        torch.cuda.current_stream().wait_stream(self.streams[local_global])

        return

    def compile_warmup_capture(self, input_shape, label_shape, num_warmup=1, graph_pool=None):
        self.print0("starting compile")
        self._compile_models(input_shape)
        self.print0("starting warmup")
        self._warmup(input_shape, label_shape, num_warmup)
        self.print0("starting capture")
        self.capture_models(input_shape, label_shape, num_warmup, graph_pool)
        self.print0("dont with compile/warmup/capture")

    def train_epoch(self, train_loader, batch_size, train_loader_len=None, to_gpu=False):
        self.batch_size = batch_size
        epoch_len = train_loader_len if train_loader_len is not None else self.last_batch
        for elems in train_loader:
            data = elems[0]
            labels = elems[1]
            if to_gpu:
                data = data.to(self.device)
                labels = labels.to(self.device)
            # loss, outputs, current_lr = self.step(data, labels, train_loader_len)
            loss, outputs, current_lr = self.step(data, labels, epoch_len)
        if self.last_batch is None:
            self.last_batch = self.batch
        return loss, outputs, current_lr

    def step(self, data, label, num_batches):
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
        self.last_batch must be set!
        """
        # stages: warmup -> full normal sync
        #         cycling -> cycle on the batches not the epochs
        #         cooldown -> full normal sync
        # need full sync at the end of every epoch!
        #   idea: run a blocking sync 1 batch before end, then normal full sync on the last batch
        # need number of steps in an epoch

        self.batch += 1
        if num_batches is None and self.last_batch is None:
            self.epoch_iter = self.batch
        else:
            self.epoch_iter = self.batch % num_batches

        if self.batch < self.warmup_steps:  # warmup
            case = 0
            model = "global"
        elif self.epoch_iter == self.last_batch - 2 or self.entering_cooldown:
            # end of epoch
            case = 1
            model = "global"
        elif self.cooldown or self.epoch_iter + self.batches_to_wait > self.last_batch:
            # cooldown phase
            case = 3
            model = "global"
        else:  # cycling
            case = 2
            model = "local"

        # key:
        #   warmup - 0
        #   end of epoch/end of cycling - 1
        #   cycling - 2
        #   cooldown - 3
        #       also: epoch cooldown -> 3 (happens when the batches to wait would go past the epoch)

        self.models[model].train()
        self.models[model].__getattr__(self.cast_dtype)()
        # to NHWC
        if self.enable_nhwc:
            # NOTE: this is designed for DEEPCAM! will need to be modified for each usecase.... :(
            shp = data.shape
            N, H, W, C = (self.batch_size, shp[1], shp[2], shp[3])
            data = torch.as_strided(data, size=[N, C, H, W], stride=[C * H * W, 1, W * C, C])
        # forward - backward - local opt - scheduler ==============================================
        if self.graphs[model] is None:
            with amp.autocast(enabled=self.enable_amp):
                outputs = self.models[model].forward(data)
                loss = self.criterion(outputs, label)

            # prepare optimizer
            self.local_optimizer.zero_grad()

            # backward pass
            self.scaler.scale(loss).backward()

            # # check for ddp mode
            # if self.ddp_mode == "sync":
            #     dist.barrier(
            #         device_ids=[comm.get_local_rank()], group=comm.get_data_parallel_group()
            #     )

            # update scaler
            self.scaler.step(self.local_optimizer)
            self.scaler.update()

            if self.enable_gpu_scheduler:
                self.static_lr = self.scheduler.group_lrs_gpu[0]
        else:
            # run graph
            self.static_input.copy_(data)
            self.static_label.copy_(label)
            self.graphs[model].replay()

            # # check for ddp mode
            # if self.ddp_mode == "sync":
            #     dist.barrier(
            #         device_ids=[comm.get_local_rank()], group=comm.get_data_parallel_group()
            #     )

            if not self.enable_gpu_scheduler:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # copy variables
            loss = self.static_loss.detach().clone()
            outputs = self.static_output.detach().clone()
        # get current learning rate
        if self.enable_gpu_scheduler:
            current_lr = self.static_lr.detach().clone()
        else:
            current_lr = self.optimizer.param_groups[0]['lr']

        # scheduler step if requested:
        if self.scheduler is not None:
            # switch depending on where scheduler runs
            if not self.enable_gpu_scheduler:
                self.scheduler.step()
            else:
                if self.graph is None:
                    self.scheduler.step_gpu()
                    if self.enable_distributed_lamb:
                        self.optimizer._lr.copy_(self.scheduler.group_lrs_gpu[0])
                    else:
                        for idg, group in enumerate(self.optimizer.param_groups):
                            group["lr"].copy_(self.scheduler.group_lrs_gpu[idg])
        # =========================================================================================
        # done at this point: forward - backward - optimizer step
        # determine when to next send/recv
        #   if the next send/recv is sooner, wait for it and put off the switch to the next batch

        if self.batch == num_batches:
            self.epoch += 1

        # key:
        #   warmup - 0
        #   end of epoch/end of cycling - 1
        #   cycling - 2
        #   cooldown - 3
        if case == 0 or case == 3:
            return loss, outputs, current_lr
        elif case == 1:  # before end of epoch / end of cycling
            self._global_sync(0)
            if self.entering_cooldown:
                self.cooldown = True
            return loss, outputs, current_lr
        elif case == 2:  # cycling
            # normal cycling logic
            # if the batches to wait goes beyond the end of epoch, do nothing
            if self.batch % 10 == 9:
                avg_loss = self.cycling_skip_logic(loss, loss_globally_averaged=False)
                c1 = self.cooldown_up_down * avg_loss
                c2 = self.cooldown_up_down * self.cooldown_threshold
                if c1 > c2:
                    self.entering_cooldown = True
            return loss, outputs, current_lr

        # next steps: start sending/receiving data
        # return loss, outputs, current_lr
        gs = self.global_skip
        # ls = self.local_skip
        # determine if to do the syncs
        # this logic is all based on the current epoch iteration
        gmod = self.epoch_iter % gs if gs > 0 else 0
        # lmod = self.epoch_iter % ls if ls > 0 else 0

        btw = (
            self.batches_to_wait
            if self.batches_to_wait + self.epoch_iter <= self.last_batch
            else 0
        )

        # do full sync on global skips and on the last batch
        if self.epoch_iter == self.last_batch or gmod == 0:
            return self._global_sync(btw)

        if gmod == btw:
            # local updates should be on before this is called!
            self._gs_rcv_update_params()
            self._local_update(self._send_mod_m1)

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

    @staticmethod
    def __init_checktypes(args: Dict) -> None:
        # this does all of the checks and raises for the parameters for init
        if not isinstance(args["local_optimizer"], torch.optim.Optimizer):
            raise TypeError(
                f"Local optimizer must be a torch optimizer object, currently {type(args['local_optimizer'])}"
            )
        if not isinstance(args["comm"], MPICommunication):
            raise TypeError(
                f"Comm object must be a ht.MPICommunication object, currently {type(args['comm'])}"
            )
        if not isinstance(args["cooldown_threshold"], float):
            raise TypeError(
                f"cooldown_threshold must be an int, "
                f"currently {type(args['cooldown_threshold'])}"
            )
        if not isinstance(args["warmup_steps"], int):
            raise TypeError(
                f"warmup_steps must be an int, currently {type(args['warmup_steps'])}"
            )
        if not isinstance(args["cooldown_epochs"], int):
            raise TypeError(
                f"cooldown_epochs must be an int, currently {type(args['cooldown_epochs'])}"
            )
        if args["scheduler"] is not None and not issubclass(
            args["scheduler"], torch.optim.lr_scheduler._LRScheduler
        ):
            raise TypeError(
                f"scheduler must be a torch learning rate scheduler, currently {args['scheduler']}"
            )
        if not isinstance(args["stability_level"], float):
            raise TypeError(
                f"stability_level must be a float, currently {type(args['stability_level'])}"
            )
        if not isinstance(args["max_global_skips"], int):
            raise TypeError(
                f"max_global_skips must be an int, currently {type(args['max_global_skips'])}"
            )
        if not isinstance(args["sending_chunk_size"], int):
            raise TypeError(
                f"sending_chunk_size must be an int, currently {type(args['sending_chunk_size'])}"
            )
        if not isinstance(args["verbose"], bool):
            raise TypeError(f"verbose must be a bool, currently {type(args['verbose'])}")
        if not isinstance(args["use_mpi_groups"], bool):
            raise TypeError(
                f"`use_mpi_grus` must be a bool, currently {type(args['use_mpi_groups'])}"
            )
        if not isinstance(args["downcast_type"], torch.dtype):
            raise TypeError(
                f"downcast_type must be a torch.dtype, currently {args['downcast_type']}"
            )
        if args["downcast_type"] not in [torch.bfloat16, torch.half, torch.float]:
            raise ValueError(
                f"downcast_type must be one of [torch.bfloat16, torch.half, torch.float], "
                f"currently {args['downcast_type']}"
            )
        if not isinstance(args["skip_reduction_factor"], int):
            raise TypeError(
                f"skip_reduction_factor must be an integer, currently {type(args['skip_reduction_factor'])}"
            )
        if not isinstance(args["local_skip_factor"], int):
            raise TypeError(
                f"local_skip_factor must be an integer, currently {type(args['local_skip_factor'])}"
            )

        if args["warmup_steps"] < 0:
            raise ValueError(f"warmup_steps must be >= 0, currently {args['warmup_steps']}")
        if args["cooldown_epochs"] < 0:
            raise ValueError(f"cooldown_epochs must be >= 0, currently {args['cooldown_epochs']}")
        if args["max_global_skips"] < 0:
            raise ValueError(f"stablitiy_level must be >= 0, currently {args['max_global_skips']}")
        if args["sending_chunk_size"] <= 0:
            raise ValueError(
                f"sending_chunk_size must be > 0, currently {args['sending_chunk_size']}"
            )
        if args["total_epochs"] <= 0:
            raise ValueError(f"total_epochs must be > 0, currently {args['total_epochs']}")
        if args["skip_reduction_factor"] <= 0:
            raise ValueError(
                f"skip_reduction_factor must be > 0, currently {args['skip_reduction_factor']}"
            )
        if args["local_skip_factor"] <= 0:
            raise ValueError(
                f"local_skip_factor must be > 0, currently {args['local_skip_factor']}"
            )

    @torch.no_grad()
    def cycling_skip_logic(
        self,
        loss: Union[torch.Tensor, int, float],
        loss_globally_averaged: bool = False
    ) -> torch.Tensor:
        """
        Function controlling the number of batches between global synchronizations and the batches to wait before
        receiving the sent parameters. The warm-up and cool-down phases are also controlled here.

        This function should be called at the end of each epoch with the training loss value at the end of the epoch.

        The number of batches between local synchronizations can also be modified here with minor code adjustments.

        Parameters
        ----------
        loss: torch.Tensor or float
            loss value of the current epoch
        loss_globally_averaged: bool, optional
            boolean if the loss is already globally averaged
        """
        if not loss_globally_averaged:
            # loss_send = torch.zeros(self.comm.size)
            # # loss.data -> this will get the raw number from the lass value and nothing else
            # loss_send[self.comm.rank] = loss.data if isinstance(loss, torch.Tensor) else loss
            # self.comm.Allreduce(MPI.IN_PLACE, loss_send, MPI.SUM)
            # avg_loss = torch.mean(loss_send)
            loss_send = loss.data if isinstance(loss, torch.Tensor) else loss
            loss_send /= self.comm.size
            avg_loss = self.comm.allreduce(loss_send)
        else:
            avg_loss = torch.tensor(loss)

        if self.global_skip == self.max_gs and self.max_gs > 4:
            self._gs8_waited += 1

        self.print0(
            f"Best loss value: {self.stability.best * (1.0 - self.stability.threshold):.4f}"
            f" Current loss: {avg_loss:.4f}, Worse epochs: {self.stability.num_bad_epochs}"
        )

        stable = self.stability.test_if_improving(avg_loss)

        if stable and self.global_skip > 1:
            # drop gs by factor of 2
            self.global_skip //= self.skip_reduction_factor
            self.batches_to_wait -= 1  # old was //= 2

            # self.print0("dropping skips")

            if self.global_skip > 0:
                if self.batches_to_wait == 0:  # avoid blocking case during cycling
                    self.batches_to_wait = 1
            self._gs8_waited = 0
            changed = True
        elif self.global_skip == 1 and stable:
            self.global_skip = self.max_gs
            self.batches_to_wait = self.max_gs // self.local_skip_factor

            self._gs8_waited = 0
            changed = True
        else:
            changed = False

        if changed:
            self.print0(
                f"\tNext Parameters: Global Skips: {self.global_skip}, Local Skips {self.local_skip}, "
                f" Batches to wait: {self.batches_to_wait}, \n\tCurrent loss: {avg_loss:.4f}, "
                f" Worse epochs: {self.stability.num_bad_epochs}"
            )
        return avg_loss

    @torch.no_grad()
    def _global_sync(self, batches_to_wait: int) -> None:
        """
        Performs a global synchronization. If `batches_to_wait > 0` this will wait for that many
        batches before received in the parameters.

        Full syncs are only performed on a single MPI group
        """
        current_comm = self.reduced_comms[self._send_mod]
        current_ranks = self.reduced_ranks[self._send_mod]

        if self.comm.rank in current_ranks:
            self._gs_send_params(current_comm, batches_to_wait)

        if self.batches_to_wait != 0:
            # update parameters from the last sending (if there)
            self._gs_rcv_update_params()  # -> splits off irrelevant ranks
            # needs to happen on all ranks:
            self._local_update(self._send_mod_m1)

        if self.batch == self.last_batch or batches_to_wait == 0:
            # receive the sent data to sync params across all ranks
            if self.comm.rank in current_ranks:
                self._gs_rcv_update_params_last_batch(current_ranks)
            else:
                if len(self._prev_params) > 0:
                    raise ValueError(
                        f"DEBUG: OFF RANKS! len(prev_params) > 0! {len(self._prev_params)}"
                        f" batch number {self.batch}"
                    )
            self._local_update(self._send_mod)

            self._send_mod_m1 = None

            if self.batch == self.last_batch:
                self._send_mod = 0
                self.epoch += 1
                self.batch = 0
            else:
                self.batch += 1
                self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0
        else:
            self.batch += 1
            self._send_mod_m1 = self._send_mod
            self._send_mod = self._send_mod + 1 if self._send_mod <= self.loc_gpus - 2 else 0

    def _gs_create_param_dict(self) -> Tuple[Dict, Dict]:
        """
        Create the shape and param dictionary used for sending parameters around the MPI world.
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

        if self._param_send_buffer_size is None:
            # use the total number of elements to define the sending buffer shape (single int)
            self._param_send_buffer_size = st

        self.param_dict = param_dict
        self.shapes = shapes
        return param_dict, shapes

    @torch.no_grad()
    def _gs_rcv_update_params(self) -> None:
        """
        Receive the previously sent parameters for the last sending MPI group.
        this is also where the sent and local parameters are merged together.
        """
        # wait for the global sync data and update on the selected rank,
        if self._send_mod_m1 is None:
            return
        prev_ranks = self.reduced_ranks[self._send_mod_m1]
        if self.comm.rank not in prev_ranks or len(self._prev_params) == 0:
            # if no old gradients, return without doing anything
            return

        prev_params = self._prev_params.pop(0)
        batches_between = float(prev_params[3])
        shapes = prev_params[2]
        # add the weighted average to the received params
        numer = batches_between * self.stale_weighting if batches_between > 0.0 else 1.0
        denom = float(len(prev_ranks) + numer)
        factor = numer / denom
        if not self.split:
            # only a single buffer
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
            # receive the first buffer
            ind = 0
            prev_params[0][0].Wait()
            del prev_params[0][0]
            rcv_params = prev_params[1][ind] / denom
            # jit the parameter setting
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    if shapes[name][1].stop > len(rcv_params):
                        # when the end of the slice is higher than the amount received, wait for the next buffer
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

    @torch.no_grad()
    def _gs_rcv_update_params_last_batch(self, current_ranks: Tuple) -> None:
        """
        Abstracted receive for the last batch (and if `global_skips` == 0)
        """
        if len(self._prev_params) > 1:
            raise ValueError(f"length of previous params > 1! {len(self._prev_params)}")
        prev_params = self._prev_params.pop(0)
        shapes = prev_params[2]
        if not self.split:
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

    @torch.no_grad()
    def _gs_send_params(self, current_comm: MPICommunication, batches_to_wait: int) -> None:
        """
        Pack and send the data required for a global synchronization on the `current_comm` group

        `batches_to_wait` is sent with the parameters to keep track of this between sending and receiving
        """
        op = MPI.SUM
        cast = False
        cast_int = 2
        if self.global_skip < 1:
            # op = mpi_sum_bfloat
            cast = True
            op = self.cast_fn
            if self.cast_dtype == torch.bfloat16:
                cast_int = 0
            elif self.cast_dtype == torch.half:
                cast_int = 1
            # else:  # keep as floats (default case, see above)

        param_dict, shapes = self._gs_create_param_dict()
        sndparams = torch.zeros(
            self._param_send_buffer_size,
            device=self.device,
            dtype=self.cast_dtype if cast else None,
        )

        sndparams = self.__pack_data(sndparams, param_dict, cast_int)
        try:
            nans = sndparams.isnan().sum()
        except RuntimeError:
            # the isnan function isnt implemented in some cuda / torch implementations
            nans = sndparams.to(torch.half).isnan().sum()

        if nans:
            # check if there are NaNs, if so, stuff is bad
            raise ValueError(f"{nans} NaNs in `params` shit be fucked.")

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

    @torch.no_grad()
    def _local_update(self, sending_process: Tuple) -> None:
        # use torch to send the network parameters of a single process to the other processes
        if sending_process is None:
            return
        snds = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                snds[name] = dist.broadcast(
                    param, sending_process, async_op=True
                )  # default is SUM
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                snds[name].wait()

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def __pack_data(
        jtparams: torch.Tensor, iter_dict: Dict[str, torch.Tensor], cast: int
    ) -> torch.Tensor:
        """
        Jitted loop to pack the data into a flattened buffer to be sent
        """
        st = 0
        cast_type = torch.float if cast == 2 else torch.bfloat16 if cast == 0 else torch.half
        for name, par in iter_dict.items():
            if par.requires_grad:
                # flatten and prep the data for sending
                p = torch.flatten(par)
                p = p.to(cast_type)
                jtparams[st : st + par.numel()] = p
                st += par.numel()
        return jtparams

    def print0(self, *args, **kwargs) -> None:
        """
        Print a message on rank 0 if the class parameter `verbose` is set.
        """
        if self.comm.rank == 0 and self.verbose:
            print(*args, **kwargs)

    def reset(self) -> None:
        """
        Reset the optimizer to its base state
        """
        self.stability.reset()
        self.global_skip = 0
        self.local_skip = 0
        self.batches_to_wait = 0
        self.batch = 0
        self._prev_params = []
        self.epoch = 0
        self._gs8_waited = 0
        self.zero_grad()

    # def step(self) -> None:
    #     """
    #     Perform a single optimization step.
    #     This will perform the `step` operations of the local optimizer,
    #     local learning rate scheduler (if defined), and the gradient scaler used in automatic mixed
    #     precision (if defined).
    #
    #     Also in the step is the logic used for when to send and receive the global/local synchronizations.
    #     Global Syncs occur on batches for which the modulus of the batch number and the `global_skip` number is 0.
    #     If `batches_to_wait` > 0, the next batches have only local syncs. After that number of batches,
    #     the data during the global sync phase is received.
    #
    #     Local synchronization can also be turned off if desired by increasing `local_skips` above 1.
    #
    #     Notes
    #     -----
    #     self.last_batch must be set!
    #     """
    #     if self.last_batch is None:
    #         raise ValueError(
    #             "self.last_batch must be set as the number of batches (len(dataloader))"
    #         )
    #
    #     if self.amp:
    #         self.scaler.step(self.local_optimizer)
    #         # todo: add something to tell if the grads have infs or nans
    #         # Updates the scale for next iteration.
    #         self.scaler.update()
    #     elif self.scheduler is None:
    #         self.local_optimizer.step()
    #     else:
    #         self.scheduler.step()
    #     batch = self.batch
    #     # knowing next_batch is important to make sure that the local sync is on
    #     #       or if the next is the last batch
    #     next_batch = batch + 1
    #     gs = self.global_skip
    #     ls = self.local_skip
    #     # determine if to do the syncs
    #     gmod = batch % gs if gs > 0 else 0
    #     lmod = batch % ls if ls > 0 else 0
    #
    #     batches_to_wait = self.batches_to_wait
    #     # ensure that the batch that will receive will be before the end of the training loop
    #     btw = (
    #         batches_to_wait
    #         if batches_to_wait + batch <= self.last_batch
    #         else self.last_batch - batch
    #     )
    #     # do full sync on global skips and on the last batch
    #     if batch == self.last_batch or gmod == 0:
    #         return self._global_sync(btw)
    #
    #     if next_batch % gs == 0:
    #         self._start_local_sync()
    #         self.batch += 1
    #         return
    #
    #     if gmod < btw:
    #         # do nothing on these batches (maintain the local sync)
    #         self.batch += 1
    #         if next_batch == self.last_batch:
    #             self._start_local_sync()
    #         return
    #     elif gmod == btw:
    #         # local updates should be on before this is called!
    #         self._gs_rcv_update_params()
    #         self._local_update(self._send_mod_m1)
    #         if ls > 1:
    #             self._stop_local_sync()
    #
    #     if ls == 1 and next_batch != self.last_batch:
    #         self.batch += 1
    #         self._start_local_sync()
    #         return
    #
    #     if lmod == 0:
    #         self._stop_local_sync()
    #     elif next_batch % ls == 0:
    #         self._start_local_sync()
    #
    #     if next_batch == self.last_batch:
    #         self._start_local_sync()
    #
    #     self.batch += 1
