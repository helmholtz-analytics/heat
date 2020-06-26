import heat as ht
import torch
import torch.nn as tnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from heat.core.communication import MPI

__all__ = ["DataParallel"]


class DataParallel(tnn.Module):
    # torch.nn.parallel.distributed
    def __init__(self, module, comm):
        super(DataParallel, self).__init__()
        self.module = module
        self.comm = comm
        self.DataLoader = None
        # todo: batch sizes

    def forward(self, *inputs, **kwargs):
        data = inputs[0]
        if isinstance(data, ht.DNDarray):
            lcl_data = data._DNDarray__array
        elif isinstance(data, torch.Tensor):
            lcl_data = data
        else:
            lcl_data = torch.tensor(data)
        ret = self.module(lcl_data, *inputs[1:], **kwargs)
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
        for f in self.parameters():
            c = torch.true_divide(f.grad.data, self.comm.size)
            self.comm.Allreduce(MPI.IN_PLACE, c, MPI.SUM)
            f.grad.data = c
            f.data.sub_(f.grad.data * learning_rate)

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
