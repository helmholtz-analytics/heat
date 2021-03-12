import sys
import torch
import unittest
from . import functional
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import heat as ht



if sys.version_info.minor >= 7:
    from .data_parallel import *

    functional.__getattr__ = functional.func_getattr

    def __getattr__(name):
        torch_all = torch.nn.modules.__all__
        if name == "SyncBatchNorm":
            return HeatSyncBatchNorm
        elif name in torch_all:
            #if name in torch_all:
            return torch.nn.__getattribute__(name)
        else:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                raise AttributeError(f"module {name} not implemented in Torch or Heat")


else:
    from . import data_parallel
    from . import tests

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.torch_all = torch.nn.modules.__all__
            self.data_parallel_all = data_parallel.__all__

        def __getattr__(self, name):
            if name == "SyncBatchNorm":
                return HeatSyncBatchNorm
            elif name in self.torch_all:
                #if name in self.torch_all:
                return torch.nn.__getattribute__(name)
            elif name == "functional":
                return functional
            elif name in self.data_parallel_all:
                return data_parallel.__getattribute__(name)
            elif name == "tests":
                return tests
            else:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    raise AttributeError(f"module '{name}' not implemented in Torch or Heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])




class HeatSyncBatchNorm(_BatchNorm):
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        # currently only GPU input is supported by underlying kernel from PyTorch
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

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
            need_sync = ht.MPI_WORLD.size > 1


        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                bn_training, exponential_average_factor, self.eps)
        else:

            assert bn_training
            return SyncBatchNorm.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor)




class SyncBatchNorm(Function):
    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        count = torch.tensor([size]).to('cuda:0')
        
        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)
        comm = ht.MPICommunication()

        count_shape = count.shape
        count = count.unsqueeze(0)
        count_all = torch.zeros((ht.MPI_WORLD.size, ) + count_shape, device=count.device, dtype=torch.int64)
        comm.Allgather(count, count_all)

        
        mean_shape = mean.shape
        mean = mean.unsqueeze(0)
        mean_all = torch.zeros((ht.MPI_WORLD.size, ) + mean_shape, device=mean.device)
        comm.Allgather(mean, mean_all)


        invstd_shape = invstd.shape
        invstd = invstd.unsqueeze(0)
        invstd_all = torch.zeros((ht.MPI_WORLD.size, ) + invstd_shape, device=invstd.device)
        comm.Allgather(invstd, invstd_all)

        counts_for_bngswc = count_all.view(-1).float() 

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts_for_bngswc
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
            self.needs_input_grad[2]
        )
        

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            comm=ht.MPICommunication() 
            sum_dy_reduced = torch.zeros_like(sum_dy,device=grad_output.device)
            comm.Allreduce(sum_dy, sum_dy_reduced, op=ht.MPI.SUM)
            
            sum_dy_xmu_reduced = torch.zeros_like(sum_dy_xmu,device=grad_output.device)
            comm.Allreduce(sum_dy_xmu, sum_dy_xmu_reduced, op=ht.MPI.SUM)

            mean_dy = sum_dy_reduced / ht.MPI_WORLD.size 
            mean_dy_xmu = sum_dy_xmu_reduced / ht.MPI_WORLD.size 

            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
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

