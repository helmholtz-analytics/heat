import torch

from PIL import Image
from torchvision import datasets
from typing import Callable, Union

from ...core import factories
from . import datatools

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
    ishuffle : bool (optional)
        flag indicating whether to use non-blocking communications for shuffling the data between epochs
        Note: if True, the ``Ishuffle()`` function must be defined within the class
        Default: False

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
    ishuffle : bool
        flag indicating if non-blocking communications are used for shuffling the data between epochs

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
        ishuffle: bool = False,
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
        self.partial_dataset = False
        self.comm = array.comm
        self.htdata = array
        self.httargets = targets
        self.ishuffle = ishuffle
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

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     print(index)
    #     img, target = self.data[index], int(self.targets[index])
    #
    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')
    #
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #     return img, target

    def Shuffle(self):
        """
        Blocking shuffle to send half of the local data to the next process in a ring (``self.comm.rank + 1`` or ``0``)
        """
        datatools.dataset_shuffle(
            dataset=self, attrs=[["htdata", "data"], ["httargets", "targets"]]
        )

    def Ishuffle(self):
        datatools.dataset_ishuffle(
            dataset=self, attrs=[["htdata", "data"], ["httargets", "targets"]]
        )
