"""
This file is for the BatchNorm classes for heat..
"""
import torch
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Optional, Any

from ..core.communication import MPICommunication


__all__ = ["HeatSyncBatchNorm"]


class HeatSyncBatchNorm(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.

    Currently :class:`SyncBatchNorm` only supports
    :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = ht.nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        comm: Optional[Any] = None,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.comm = comm

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError("expected at least 2D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        # currently only GPU input is supported by underlying kernel from PyTorch
        if not input.is_cuda:
            raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None

        need_sync = bn_training
        if need_sync:
            if self.comm:
                comm = self.comm
            else:
                comm = MPICommunication()
            need_sync = comm.size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            return SyncBatchNorm.apply(
                input,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                self.eps,
                exponential_average_factor,
                comm,
            )


class SyncBatchNorm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, comm):
        input = input.contiguous()
        self.comm = comm
        size = input.numel() // input.size(1)
        count = torch.tensor([size]).to(input.device)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        count_shape = count.shape
        count = count.unsqueeze(0)
        count_all = torch.zeros((comm.size,) + count_shape, device=count.device, dtype=torch.int64)
        comm.Allgather(count, count_all)

        mean_shape = mean.shape
        mean = mean.unsqueeze(0)
        mean_all = torch.zeros((comm.size,) + mean_shape, device=mean.device)
        comm.Allgather(mean, mean_all)

        invstd_shape = invstd.shape
        invstd = invstd.unsqueeze(0)
        invstd_all = torch.zeros((comm.size,) + invstd_shape, device=invstd.device)
        comm.Allgather(invstd, invstd_all)

        counts_for_bngswc = count_all.view(-1).float()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input, mean_all, invstd_all, running_mean, running_var, momentum, eps, counts_for_bngswc
        )

        self.save_for_backward(input, weight, running_mean, running_var, count)
        # apply element-wise normalization
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_all = self.saved_tensors
        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2],
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            sum_dy_reduced = torch.zeros_like(sum_dy, device=grad_output.device)
            self.comm.Allreduce(sum_dy, sum_dy_reduced)

            sum_dy_xmu_reduced = torch.zeros_like(sum_dy_xmu, device=grad_output.device)
            self.comm.Allreduce(sum_dy_xmu, sum_dy_xmu_reduced)

            mean_dy = sum_dy_reduced / self.comm.size
            mean_dy_xmu = sum_dy_xmu_reduced / self.comm.size

            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output, saved_input, mean, invstd, weight, mean_dy, mean_dy_xmu
            )

            grad_input = grad_output
        else:
            grad_input = None

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
