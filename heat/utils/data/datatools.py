import h5py
import numpy as np
import base64
import os
import torch
from torch.utils import data as torch_data
from typing import Callable, Iterator, Union

from heat.core import dndarray

__all__ = ["DataLoader", "Dataset"]


class DataLoader:
    """
    Data Loader. The combines either a ``DNDarray`` or a torch ``Dataset`` with a sampler. It will provide an iterable
    over the local dataset and it will have a ``shuffle()`` function which calls the ``shuffle()`` function of the
    given Dataset. If a HeAT DNDarray is given a general Dataset will be created.

    Currently, the DataLoader supports only map-style datasets with single-process loading, and users the random
    batch sampler. The rest of the DataLoader functionality mentioned in :func:`torch.utils.data.dataloader` applies.

    Arguments:
        data : Dataset or DNDarray
            dataset from which to load the data.
        batch_size : int, optional
            how many samples per batch to load (default: ``1``).
        num_workers : int, optional
            how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn : callable, optional
            merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.  If your
            data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type, see
            the example below.
        drop_last : bool, optional
            set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
            the last batch will be smaller. (default: ``False``)
        timeout : int or float, optional
            if positive, the timeout value for collecting a batch from workers. Should always be non-negative.
            (default: ``0``)
        worker_init_fn : callable, optional
            If not ``None``, this will be called on each worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: ``None``)
        lcl_dataset : torch.Dataset
            a PyTorch dataset from which the data will be returned by the created iterator
        transform : Callable
            transform to be given to Dataset creation if a Dataset is created

    Attributes
    ----------
    dataset : torch.data.utils.data.Dataset or heat.Dataset
        the dataset created from the local data
    DataLoader : torch.utils.data.DataLoader
        the local DataLoader object. Used in the creation of the iterable and the length
    _first_iter : bool
        flag indicating if the iterator created is the first one. If it is not, then the data will be shuffled before
        the iterator is created

    TODO: add data annotation
    """

    def __init__(
        self,
        data=None,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: Union[int, float] = 0,
        worker_init_fn: Callable = None,
        lcl_dataset: torch_data.Dataset = None,
        transform: Callable = None,
    ):
        if isinstance(data, dndarray.DNDarray) and lcl_dataset is not None:
            self.dataset = Dataset(array=data, transform=transform)
        elif lcl_dataset:
            self.dataset = lcl_dataset
        else:
            raise TypeError(
                f"data must be a DNDarray or lcl_dataset must be given, data is currently: {type(data)}"
            )
        # this is effectively setting ``shuffle`` to True
        # rand_sampler = torch_data.RandomSampler(self.dataset)
        # sampler = torch_data.BatchSampler(rand_sampler, batch_size, drop_last)
        self.DataLoader = torch_data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self._first_iter = True

    def __iter__(self) -> Iterator:
        # need a new iterator for each epoch
        if self._first_iter:
            self._first_iter = False
        else:
            # shuffle after the first epoch but before the iterator is generated
            self.shuffle()
        return self.DataLoader.__iter__()

    def __len__(self) -> int:
        return len(self.DataLoader)

    def shuffle(self):
        """
        Shuffle the data. This is a direct call to the shuffle function of the dataset.
        This function is blocking.
        """
        self.dataset.shuffle()


class Dataset(torch_data.Dataset):
    """
    An abstract class representing a given dataset. This inherits from torch.utils.data.Dataset.

    This class is a general example for what should be done to create a Dataset. When creating a dataset all of the
    standard attributes should be set, the ``__getitem__``, ``__len__``, and ``shuffle`` functions must be defined.

        - ``__getitem__`` : how an item is given to the network
        - ``__len__`` : the number of data elements to be given to the network in total
        - ``shuffle`` : how the data should be shuffled between the processes. The function shown below is for a dataset
            composed of only data and without targets

    As the amount of data across processes can be non-uniform, the dataset class will slice off the remaining elements
    on whichever processes have more data than the others. This should only be 1 element.
    The shuffle function will shuffle all of the data on the process.

    It is recommended that for DNDarrays, the split is either 0 or None

    Parameters
    ----------
    array : DNDarray
        DNDarray for which to great the dataset
    transform : Callable
        transformation to call before a data item is returned

    Attributes
    ----------
    These are the required attributes. Optional attributed are whatever are required for the Dataset
    (see :class:`heat.utils.data.mnist.py`)
    htdata : DNDarray
        full data
    _cut_slice : slice
        slice to cut off the last element to get a uniform amount of data on each process
    comm : MPICommunicator
        communication object used to send the data between processes
    lcl_half : int
        half of the number of data elements on the process
    data : torch.Tensor
        the local data to be used in training
    transform : Callable
        transform to be called during the getitem function

    TODO: type annotation for array
    """

    def __init__(self, array, transform: Callable = None):
        self.htdata = array
        self.comm = array.comm
        # create a slice to create a uniform amount of data on each process
        min_data_split = array.gshape[array.split] // array.comm.size
        self.lcl_half = min_data_split // 2
        arb_slice = [slice(None)] * array.ndim
        arb_slice[array.split] = slice(min_data_split)
        self._cut_slice = tuple(arb_slice)
        self.data = array._DNDarray__array[self._cut_slice]
        self.transform = transform

    def __getitem__(self, index: Union[int, slice, tuple, list, torch.Tensor]) -> torch.Tensor:
        if self.transform:
            return self.transform(self.data[index])
        return self.data[index]

    def __len__(self) -> int:
        return self.data.shape[0]

    def shuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        lc = self.htdata._DNDarray__array
        prm = torch.randperm(lc.shape[0])
        snd = lc[: self.lcl_half].clone()
        snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
        comm = self.comm
        dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
        # send the top half of the data to the next process
        sw = comm.Isend(snd, dest=dest)
        del snd
        new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
        src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
        rw = comm.Irecv(new_data, source=src)
        sw.wait()
        rw.wait()
        self.htdata._DNDarray__array[: self.lcl_half] = new_data
        self.htdata._DNDarray__array = self.htdata._DNDarray__array[prm]
        self.data = self.htdata._DNDarray__array[self._cut_slice]
