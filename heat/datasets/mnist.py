import torch
from torchvision import datasets
from PIL import Image
from ..core import factories

__all__ = ["MNISTDataset"]


class MNISTDataset(datasets.MNIST):
    # todo: implement iterable-style datasets
    # only map still datasets here
    # assumes that the items to train on are in the 0th axis
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        # slice the data at the smallest number of elements
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        array = factories.array(self.data, split=0)
        targets = factories.array(self.targets, split=0)
        min_data_split = array.gshape[0] // array.comm.size
        arb_slice = slice(min_data_split)
        self._cut_slice = arb_slice
        self.comm = array.comm
        self.htarray = array
        self.lcl_half = min_data_split // 2
        self.data = array._DNDarray__array[self._cut_slice]
        self.targets = targets._DNDarray__array[self._cut_slice]
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        # shuffled = self.data[torch.randperm(self.data.shape[0])]
        # snd = shuffled[: self.lcl_half].clone()
        # snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
        # comm = self.comm
        # dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
        # # send the top half of the data to the next process
        # comm.Send(snd, dest=dest)
        # del snd
        # new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
        # src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
        # comm.Recv(new_data, source=src)
        # self.data[: self.lcl_half] = new_data
        print("shuffle")
        comm = self.comm
        rd_perm = torch.randperm(self.data.shape[0])
        for dat in [self.data, self.targets]:
            shuffled = dat[rd_perm]
            snd = shuffled[: self.lcl_half].clone()
            snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
            dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
            # send the top half of the data to the next process
            send_wait = comm.Isend(snd, dest=dest)
            del snd
            new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
            src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
            rcv_w = comm.Irecv(new_data, source=src)
            send_wait.wait()
            rcv_w.wait()
            dat[: self.lcl_half] = new_data
