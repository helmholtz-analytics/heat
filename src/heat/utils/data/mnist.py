"""
File for the MNIST dataset definition in heat
"""

import torch

from torchvision import datasets
from typing import Callable, Union

from ...core import factories
from . import datatools

__all__ = ["MNISTDataset"]


class MNISTDataset(datasets.MNIST):
    """
    Dataset wrapper for `torchvision.datasets.MNIST <https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST>`_.
    This implements all of the required functions mentioned in :class:`heat.utils.data.Dataset`. The ``__getitem__`` and ``__len__`` functions are inherited from
    `torchvision.datasets.MNIST <https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST>`_.

    Parameters
    ----------
    root : str
        Directory containing the MNIST dataset
    train : bool, optional
        If the data is the training dataset or not, default is True
    transform : Callable, optional
        Transform to be applied to the data dataset in the ``__getitem__`` function, default is ``None``
    target_transform : Callable, optional
        Transform to be applied to the target dataset in the ``__getitem__`` function, default is ``None``
    download : bool, optional
        If the data does not exist in the directory, download it if True (default)
    split : int, optional
        On which access to split the data when it is loaded into a ``DNDarray``
    ishuffle : bool, optional
        Flag indicating whether to use non-blocking communications for shuffling the data between epochs
        Note: if True, the ``Ishuffle()`` function must be defined within the class
        Default: ``False``
    test_set : bool, optional
        If this dataset is the testing set then keep all of the data local
        Default: ``False``

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
    test_set : bool
        if this dataset is the testing set then keep all of the data local

    Notes
    -----
    For other attributes see `torchvision.datasets.MNIST <https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST>`_.
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
        test_set: bool = False,
    ):  # noqa: D107
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if split != 0 and split is not None:
            raise ValueError("split must be 0 or None")
        split = None if test_set else split
        array = factories.array(self.data, split=split)
        targets = factories.array(self.targets, split=split)
        self.test_set = test_set
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
        # getitem and len are defined by torch's MNIST class

    def Shuffle(self):
        """
        Uses the :func:`datatools.dataset_shuffle` function to shuffle the data between the processes
        """
        if not self.test_set:
            datatools.dataset_shuffle(
                dataset=self, attrs=[["data", "htdata"], ["targets", "httargets"]]
            )

    def Ishuffle(self):
        """
        Uses the :func:`datatools.dataset_ishuffle` function to shuffle the data between the processes
        """
        if not self.test_set:
            datatools.dataset_ishuffle(
                dataset=self, attrs=[["data", "htdata"], ["targets", "httargets"]]
            )
