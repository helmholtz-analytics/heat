import torch

from PIL import Image
from heat.core import factories
from torchvision import datasets
from typing import Callable, Union

__all__ = ["MNISTDataset"]


class MNISTDataset(datasets.MNIST):
    """
    Dataset wrapper for :class:`torchvision.datasets.MNIST`. This implements all of the required functions mentioned in
    :class:`heat.utils.data.Dataset`. The ``__getitem__`` and ``__len__`` functions are inherited from
    :class:`torchvision.datasets.MNIST`.

    Parameters
    ----------
    root : str
        directory containing the MNIST dataset
    train : bool, optional
        if the data is the training dataset or not, default is True
    transform : Callable, optional
        transform to be applied to the data dataset in the ``__getitem__`` function, default is None
    target_transform : Callable, optional
        transform to be applied to the target dataset in the ``__getitem__`` function, default is None
    download : bool, optional
        if the data does not exist in the directory, download it if True (default)
    split : int, optional
        on which access to split the data when it is loaded into a DNDarray

    Attributes
    ----------
    htdata : DNDarray
        full data
    httargets : DNDarray
        full target data
    comm : communication.MPICommunicator
        heat communicator for sending data between processes
    _cut_slice : slice
        slice to remove the last element if all are not equal in length
    lcl_half : int
        integer value of half of the data on the process
    data : torch.Tensor
        the local data on a process
    targets : torch.Tensor
        the local targets on a process

    For other attributes see :class:`torchvision.datasets.MNIST`.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = True,
        split: int = 0,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if split != 0 and split is not None:
            raise ValueError("split must be 0 or None")
        array = factories.array(self.data, split=split)
        targets = factories.array(self.targets, split=split)
        self.comm = array.comm
        self.htdata = array
        self.httargets = targets
        if split is not None:
            min_data_split = array.gshape[0] // array.comm.size
            arb_slice = slice(min_data_split)
            self._cut_slice = arb_slice
            self.lcl_half = min_data_split // 2
            self.data = array._DNDarray__array[self._cut_slice]
            self.targets = targets._DNDarray__array[self._cut_slice]
        else:
            self._cut_slice = None
            self.lcl_half = array.gshape[0] // 2
            self.data = array._DNDarray__array

            self.targets = targets._DNDarray__array

    def shuffle(self):
        """
        Blocking shuffle to send half of the local data to the next process in a ring (``self.comm.rank + 1`` or ``0``)
        """
        comm = self.comm
        rd_perm = torch.randperm(self.htdata.lshape[0])

        # for x in [["data", "htdata"]]:
        #     self.__getattribute__(x)
        ld = self.htdata._DNDarray__array
        snd = ld[: self.lcl_half].clone()
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
        self.htdata._DNDarray__array[: self.lcl_half] = new_data
        self.htdata._DNDarray__array = self.htdata._DNDarray__array[rd_perm]
        self.data = self.htdata._DNDarray__array[self._cut_slice]

        # shuffled = self.targets[rd_perm]
        ld2 = self.httargets._DNDarray__array
        snd = ld2[: self.lcl_half].clone()
        snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
        dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
        # send the top half of the data to the next process
        send_wait = comm.Isend(snd, dest=dest)
        del snd
        new_data2 = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
        src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
        rcv_w = comm.Irecv(new_data2, source=src)
        send_wait.wait()
        rcv_w.wait()
        self.httargets._DNDarray__array[: self.lcl_half] = new_data2
        self.httargets._DNDarray__array = self.httargets._DNDarray__array[rd_perm]
        self.targets = self.httargets._DNDarray__array[self._cut_slice]
