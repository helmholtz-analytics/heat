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
    def __init__(self, module, comm, nonblocking=False):
        super(DataParallel, self).__init__()
        self.module = module
        self.comm = comm
        self.DataLoader = None
        self.wait_handles = OrderedDict()

        # todo: remove batch size attributes when dataloader is finished
        self.global_batch_size = 0
        self.local_batch_size = 0

        # registering hooks for all model parameter tensors
        for name, param in module.named_parameters():
            layer_name = name.split(sep=".", maxsplit=1)[0]

            if nonblocking:
                param.register_hook(self.nonblocking_hook(layer_name))
            else:
                param.register_hook(self.blocking_hook)
        # todo: batch sizes

    def forward(self, *inputs, **kwargs):
        data = inputs[0]

        # todo: remove setting batch size attributes when dataloader is finished
        if isinstance(data, ht.DNDarray):
            lcl_data = data._DNDarray__array
            self.global_batch_size = data.gshape[0]
            self.local_batch_size = data.lshape[0]
        elif isinstance(data, torch.Tensor):
            lcl_data = data
            self.global_batch_size = data.size()[0]
            self.local_batch_size = self.global_batch_size
        else:
            lcl_data = torch.tensor(data)
            self.global_batch_size = lcl_data.size()[0]
            self.local_batch_size = self.global_batch_size
        ret = self.module(lcl_data, *inputs[1:], **kwargs)

        # clear dictionary after all wait handles are used up
        self.wait_handles.clear()

        return ret

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
        grad_loc_cpy *= self.local_batch_size / self.global_batch_size

        # perform MPI Allreduce to compute global gradient
        self.comm.Allreduce(ht.MPI.IN_PLACE, grad_loc_cpy, ht.MPI.SUM)

        return grad_loc_cpy

    # hook function for blocking gradient data exchange
    def nonblocking_hook(self, layer_name):
        def hook(grad_loc):
            # Pytorch Doc says, :attr:`grad` may not be modified itself, so it has to be cloned
            # (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
            # Seems to be true, since otherwise a Runtime Error is thrown when working on it

            grad_loc_cpy = grad_loc.clone()

            # counterbalance local gradient averaging
            grad_loc_cpy *= self.local_batch_size

            # perform MPI IAllreduce to compute global gradient, returns wait handle
            wait_handle = self.comm.Iallreduce(ht.MPI.IN_PLACE, grad_loc_cpy, ht.MPI.SUM)

            # if wait handle dictionary does not contain the layer yet, add it -> automatically tracks reversed layer
            # order
            if layer_name not in self.wait_handles:
                self.wait_handles[layer_name] = list()

            # get size of flattened tensor
            size1D = functools.reduce(operator.mul, grad_loc.shape, 1)

            # assign wait handle to its layer, layer-internal sorting by size
            bisect.insort(self.wait_handles[layer_name], (size1D, wait_handle))

            return grad_loc

        return hook

    class __LocalDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def __len__(self):
            return len(self.data)

    def init_data_loader_from_heat(self, ht_data, batch_size=None, DatasetClass=None):
        # create a torch.DataLoader
        # todo: drop_last = True
        # todo: active data is equal to the smallest local split value
        if batch_size is None:  # default for DataLoader
            batch_size = 1
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
