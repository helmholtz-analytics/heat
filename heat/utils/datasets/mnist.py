import torch
import torchvision
from torch.utils import data as torch_data
from PIL import Image

from ...core import factories

# from torch.MNIST dataloader


class MNISTDataset:  #(torchvision.datasets.vision.VisionDataset):
    # todo: implement iterable-style datasets
    # only map still datasets here
    # assumes that the items to train on are in the 0th axis
    def __init__(self, lcl_MNIST_Dataset):
        # create a globl heat array from the local data
        data_array = factories.array(lcl_MNIST_Dataset.data, split=0)
        train_array = factories.array(lcl_MNIST_Dataset.targets, split=0)
        # slice the data at the smallest number of elements
        self.comm = data_array.comm

        min_data = data_array.gshape[0] // data_array.comm.size
        self.lcl_half = min_data // 2
        self.data = []
        self.data.append(data_array._DNDarray__array[slice(min_data)])
        min_data_t = train_array.gshape[0] // train_array.comm.size
        self.data.append(data_array._DNDarray__array[slice(min_data_t)])
        self.transform = lcl_MNIST_Dataset.transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[0][index], int(self.targets[1][index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        comm = self.comm
        for d in range(2):
            shuffled = self.data[d][torch.randperm(self.data[d].shape[0])]
            snd = shuffled[: self.lcl_half].clone()
            snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
            dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
            # send the top half of the data to the next process
            comm.Send(snd, dest=dest)
            del snd
            new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
            src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
            comm.Recv(new_data, source=src)
            self.data[d][: self.lcl_half] = new_data

