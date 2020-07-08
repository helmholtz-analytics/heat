import bisect
import functools
import operator
from collections import OrderedDict

import heat as ht
import torch
import torch.nn as tnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from heat.core.communication import MPI

__all__ = ["DataParallel"]


class DataParallel(tnn.Module):
    # torch.nn.parallel.distributed
    def __init__(self, module, comm, optimizer=None):
        super(DataParallel, self).__init__()
        self.module = module
        self.comm = comm
        self.optimizer = optimizer

        self.DataLoader = None
        self.wait_handles = OrderedDict()
        self.fwd_hook_handles = list()
        # slices of parameters belonging to one and the same layer
        self.param_slices = dict()
        # pytorch internal parameter indexing
        self.param_indices = dict()
        # reference of optimizer's params
        self.params_ref = None

        # check if non-blocking
        if optimizer is not None:
            # check if optimizer matches module
            if list(module.parameters()) != optimizer.param_groups[0]["params"]:
                raise ValueError("given module and optimizer don't share same parameters.")
            else:
                # take reference of optimizer's params
                self.params_ref = optimizer.param_groups[0]["params"]
                optimizer.param_groups[0]["params"] = []

        # get parameter indexing and slices
        start_idx = 0
        layer_name_prev = None
        for idx, (name, param) in enumerate(module.named_parameters()):
            self.param_indices[name] = idx
            layer_name = name.split(sep=".", maxsplit=1)[0]
            if layer_name_prev is None:
                layer_name_prev = layer_name
            if layer_name_prev != layer_name:
                self.param_slices[layer_name_prev] = slice(start_idx, idx)
                layer_name_prev = layer_name
                start_idx = idx

            # register backward hooks for all model parameter tensors
            if optimizer is not None:
                param.register_hook(self.nonblocking_hook(layer_name, name))
            else:
                param.register_hook(self.blocking_hook)
        self.param_slices[layer_name_prev] = slice(start_idx, len(self.param_indices))
        # todo: batch sizes

    def forward(self, *inputs, **kwargs):
        data = inputs[0]

        if isinstance(data, ht.DNDarray):
            lcl_data = data._DNDarray__array
        elif isinstance(data, torch.Tensor):
            lcl_data = data
        else:
            lcl_data = torch.tensor(data)

        # check if non-blocking
        if self.optimizer is not None and self.module.training:
            # reset gradients before forward pass
            self.optimizer.zero_grad()

            # register forward hooks for all layers
            for name, submodule in self.module.named_modules():
                if name == "":
                    continue

                if name in self.wait_handles:
                    hook_handle = submodule.register_forward_pre_hook(self.forward_hook(name))
                    self.fwd_hook_handles.append(hook_handle)

        # perform forward pass
        ret = self.module(lcl_data, *inputs[1:], **kwargs)

        # clear dictionary after all wait handles are used up (dynamic computation graph)
        self.wait_handles.clear()

        # remove forward hooks (dynamic computation graph)
        for hook_handle in self.fwd_hook_handles:
            hook_handle.remove()
        self.fwd_hook_handles.clear()

        return ret

    def async_update(self, param_slice=None, layer_names=None):
        # perform update on the whole model
        if param_slice is None:
            param_slice = slice(len(self.params_ref))
        if layer_names is None:
            layer_names = list(self.wait_handles.keys())

        # update params that are visible for the optimizer
        self.optimizer.param_groups[0]["params"] = self.params_ref[param_slice]

        # iterate over layers
        for layer_name in layer_names:
            # iterate over layer's parameters/associated wait handles
            for _, param_name, wait_handle in self.wait_handles[layer_name]:
                # get internal index of selected parameter
                param_idx = self.param_indices[param_name]
                # synchronize, get parameter's global gradient
                wait_handle.wait()
                # check if shapes are matching
                if self.params_ref[param_idx].grad.data.shape != wait_handle.tensor.shape:
                    raise ValueError("Shapes must be equal.")
                # set parameter's global gradient
                self.params_ref[param_idx].grad.data = wait_handle.tensor

        # perform actual parameter update
        self.optimizer.step()

    def set_comm(self, comm):
        self.comm = comm

    def local_loss(self, loss_func, labels, outputs):
        if isinstance(outputs, ht.DNDarray):
            lcl_out = outputs._DNDarray__array
        elif not isinstance(outputs, torch.Tensor):
            raise TypeError("Outputs should be torch.Tensors")
        else:
            lcl_out = outputs

        if isinstance(labels, ht.DNDarray):
            lcl_labels = labels._DNDarray__array
        elif not isinstance(labels, torch.Tensor):
            raise TypeError("Outputs should be torch.Tensors")
        else:
            lcl_labels = labels

        return loss_func(lcl_out, lcl_labels)

    def blocking_grad_update(self, learning_rate):
        # need to send the self.parameters() to the other processes after the backwards loss step
        # then can use optimizer.step
        # print(list(self.parameters()))
        for f in self.parameters():
            c = torch.true_divide(f.grad.data, self.comm.size)
            self.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
            f.grad.data = c
            f.data.sub_(f.grad.data * learning_rate)

    def blocking_hook(self, grad_loc):
        # Pytorch Doc says, :attr:`grad` may not be modified itself, so it has to be cloned
        # (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
        # Seems to be true, since otherwise a Runtime Error is thrown
        grad_loc_cpy = grad_loc.clone()

        # counterbalance local gradient averaging
        # todo: should this be the number of processes? and should it be a division before the send
        # global gradient averaging
        grad_loc_cpy /= self.comm.size

        # perform MPI Allreduce to compute global gradient
        self.comm.Allreduce(ht.MPI.IN_PLACE, grad_loc_cpy, ht.MPI.SUM)

        return grad_loc_cpy

    # hook function for blocking gradient data exchange
    def nonblocking_hook(self, layer_name, param_name):
        def hook(grad_loc):
            # Pytorch Doc says, :attr:`grad` may not be modified itself, so it has to be cloned
            # (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
            # Seems to be true, since otherwise a Runtime Error is thrown when working on it

            grad_loc_cpy = grad_loc.clone()

            # counterbalance local gradient averaging
            grad_loc_cpy /= self.comm.size

            # perform MPI IAllreduce to compute global gradient, returns wait handle
            wait_handle = self.comm.Iallreduce(ht.MPI.IN_PLACE, grad_loc_cpy, ht.MPI.SUM)

            # if wait handle dictionary does not contain the layer yet, add it -> automatically tracks reversed layer
            # order
            if layer_name not in self.wait_handles:
                self.wait_handles[layer_name] = list()

            # get size of flattened tensor
            size1D = functools.reduce(operator.mul, grad_loc.shape, 1)

            # assign wait handle to its layer, layer-internal sorting by size
            bisect.insort(self.wait_handles[layer_name], (size1D, param_name, wait_handle))

            return grad_loc

        return hook

    # hook function for non-blocking parameter update
    def forward_hook(self, layer_name):
        def hook(_, input_):
            # update parameters of given layer
            param_slice = self.param_slices[layer_name]
            self.async_update(param_slice, [layer_name])
            return input_

        return hook

    class __LocalDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def __len__(self):
            return len(self.data)

    def init_data_loader_from_heat(self, ht_data, batch_size=None, DatasetClass=None):
        if DatasetClass is None:  # allow for other Dataset classes
            DatasetClass = self.__LocalDataset

        if ht_data.is_distributed():
            lshape_map = ht_data.create_lshape_map()
            min_data_sp = min(lshape_map[..., ht_data.split])
            # need to ensure that the data shape on all processes
            arb_slice = [slice(None, None, None)] * len(ht_data.gshape)
            arb_slice[ht_data.split] = slice(0, min_data_sp)
            loc_data = ht_data._DNDarray__array[tuple(arb_slice)]
        else:
            loc_data = ht_data._DNDarray__array

        dataset = DatasetClass(loc_data)
        self.DataLoader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    # def init_DataLoader_from_torch(self, datasets, batch_size):

    def data_shuffle_w_loader(self):
        # todo: implement at data shuffler across the different processes
        pass
