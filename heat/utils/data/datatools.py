"""
Function and classes useful for loading data into neural networks
"""

import torch
from torch.utils import data as torch_data
from typing import Callable, List, Iterator, Union, Optional, Sized

from ...core.dndarray import DNDarray
from ...core.communication import MPI_WORLD
from . import partial_dataset

__all__ = ["DataLoader", "Dataset", "dataset_shuffle", "dataset_ishuffle"]


class DataLoader:
    r"""
    The combines either a :func:`DNDarray <heat.core.dndarray.DNDarray>` or a torch `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_
    with a sampler. This provides an iterable over the local dataset and it will shuffle the data at the end of the
    iterator. If a :func:`DNDarray <heat.core.dndarray.DNDarray>` is given, then a :func:`Dataset` will be created
    internally.

    Currently, this only supports only map-style datasets with single-process loading. It uses the random
    batch sampler. The rest of the ``DataLoader`` functionality mentioned in `torch.utils.data.dataloader <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader>`_ applies.

    Arguments:
        dataset : :func:`Dataset`, torch `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_, :func:`heat.utils.data.partial_dataset.PartialH5Dataset`
            A torch dataset from which the data will be returned by the created iterator
        batch_size : int, optional
            How many samples per batch to load\n
             Default: 1
        num_workers : int, optional
            How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.\n
            Default: 0
        collate_fn : callable, optional
            Merges a list of samples to form a mini-batch of torch.Tensor(s).  Used when using batched loading from a
            map-style dataset.\n
            Default: None
        pin_memory : bool, optional
            If ``True``, the data loader will copy torch.Tensors into CUDA pinned memory before returning them.
            If your data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below. \n
            Default: False
        drop_last : bool, optional
            Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
            the last batch will be smaller.\n
            Default: ``False``
        timeout : int or float, optional
            If positive, the timeout value for collecting a batch from workers. Should always be non-negative.\n
            Default: 0
        worker_init_fn : callable, optional
            If not ``None``, this will be called on each worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading.\n
            default: None

    Attributes
    ----------
    dataset : :func:`Dataset`, torch `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_, :func:`heat.utils.data.partial_dataset.PartialH5Dataset`
        The dataset created from the local data
    DataLoader : `torch.utils.data.dataloader <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader>`_
        The local DataLoader object. Used in the creation of the iterable and the length
    _first_iter : bool
        Flag indicating if the iterator created is the first one. If it is not, then the data will be shuffled before
        the iterator is created
    last_epoch : bool
        Flag indicating last epoch
    """

    def __init__(
        self,
        dataset: Union[torch_data.Dataset, partial_dataset.PartialH5Dataset],
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: Union[int, float] = 0,
        worker_init_fn: Callable = None,
    ):  # noqa: D107
        if not isinstance(dataset, (torch_data.Dataset, Dataset, partial_dataset.PartialH5Dataset)):
            raise TypeError(
                f"dataset must be a torch Dataset, heat Dataset, heat PartialH5Dataset, currently: {type(dataset)}"
            )
        self.dataset = dataset
        self.ishuffle = self.dataset.ishuffle
        if isinstance(self.dataset, partial_dataset.PartialH5Dataset):
            drop_last = True

        self.DataLoader = torch_data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self._first_iter = True
        self.last_epoch = False

    def __iter__(self) -> Iterator:
        """
        Generate a new iterator of a type dependent on the type of dataset.
        Returns a :class:`partial_dataset.PartialH5DataLoaderIter` if the dataset is a :class:`partial_dataset.PartialH5Dataset`
        :func:`self._full_dataset_shuffle_iter` otherwise
        """
        if isinstance(self.dataset, partial_dataset.PartialH5Dataset):
            return partial_dataset.PartialH5DataLoaderIter(self)
        if hasattr(self, "_full_dataset_shuffle_iter"):
            # if it is a normal heat dataset then this is defined
            self._full_dataset_shuffle_iter()
        return self.DataLoader.__iter__()

    def __len__(self) -> int:
        """
        Get the length of the dataloader. Returns the number of batches.
        """
        return self.DataLoader.__len__()

    def _full_dataset_shuffle_iter(self):
        # logic for when to shuffle the data
        if not self.ishuffle:
            if self._first_iter:
                self._first_iter = False
            else:
                # shuffle after the first epoch but before the iterator is generated
                self.dataset.Shuffle()
        else:
            # start the shuffling for the next iteration
            if not self.last_epoch:
                self.dataset.Ishuffle()

            if self._first_iter:
                self._first_iter = False
            else:
                dataset_irecv(self.dataset)


class Dataset(torch_data.Dataset):
    r"""
    An abstract class representing a given dataset. This inherits from torch.utils.data.Dataset.

    This class is a general example for what should be done to create a Dataset. When creating a dataset all of the
    standard attributes should be set, the ``__getitem__``, ``__len__``, and ``shuffle`` functions must be defined.

        - ``__getitem__`` : how an item is given to the network
        - ``__len__`` : the number of data elements to be given to the network in total
        - ``Shuffle()`` : how the data should be shuffled between the processes. The function shown below is for a dataset composed of only data and without targets. The function :func:`dataset_shuffle` abstracts this. For this function only the dataset and a list of attributes to shuffle are given.\n
        - ``Ishuffle()`` : A non-blocking version of ``Shuffle()``, this is handled in the abstract function :func:`dataset_ishuffle`. It works similarly to :func:`dataset_shuffle`.

    As the amount of data across processes can be non-uniform, the dataset class will slice off the remaining elements
    on whichever processes have more data than the others. This should only be 1 element.
    The shuffle function will shuffle all of the data on the process.

    It is recommended that for ``DNDarray`` s, the split is either 0 or None

    Parameters
    ----------
    array : DNDarray
        DNDarray for which to great the dataset
    transform : Callable
        Transformation to call before a data item is returned
    ishuffle : bool, optional
        flag indicating whether to use non-blocking communications for shuffling the data between epochs
        Note: if ``True``, the ``Ishuffle()`` function must be defined within the class\n
        Default: False

    Attributes
    ----------
    These are the required attributes.

    htdata : DNDarray
        Full data
    _cut_slice : slice
        Slice to cut off the last element to get a uniform amount of data on each process
    comm : MPICommunicator
        Communication object used to send the data between processes
    lcl_half : int
        Half of the number of data elements on the process
    data : torch.Tensor
        The local data to be used in training
    transforms : Callable
        Transform to be called during the getitem function
    ishuffle : bool
        Flag indicating if non-blocking communications are used for shuffling the data between epochs
    """

    def __init__(
        self,
        array,
        transforms: Optional[Union[List, Callable]] = None,
        ishuffle: Optional[bool] = False,
        test_set: Optional[bool] = False,
    ):  # noqa: D107
        self.htdata = array
        self.comm = array.comm
        self.test_set = test_set
        # create a slice to create a uniform amount of data on each process
        min_data_split = array.gshape[array.split] // array.comm.size
        self.lcl_half = min_data_split // 2
        arb_slice = [slice(None)] * array.ndim
        arb_slice[array.split] = slice(min_data_split)
        self._cut_slice = tuple(arb_slice)
        self.data = array._DNDarray__array[self._cut_slice]
        if not isinstance(transforms, (list, tuple)) and transforms is not None:
            transforms = [transforms]
        self.transforms = transforms
        self.ishuffle = ishuffle

    def __getitem__(self, index: Union[int, slice, tuple, list, torch.Tensor]) -> torch.Tensor:
        """
        This is the most basic form of getitem. As the dataset is often very specific to the dataset,
        this should be overwritten by the user. In this form it only gets the raw items from the data.
        """
        if self.transforms:
            return self.transforms[0](self.data[index])
        return self.data[index]

    def __len__(self) -> int:
        """
        Get the number of items in the dataset. This should be overwritten by custom datasets
        """
        return self.data.shape[0]

    def Shuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        if not self.test_set:
            dataset_shuffle(dataset=self, attrs=[["data", "htdata"]])

    def Ishuffle(self):
        """
        Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
        """
        if not self.test_set:
            dataset_ishuffle(dataset=self, attrs=[["data", "htdata"]])


def dataset_shuffle(dataset: Union[Dataset, torch_data.Dataset], attrs: List[list]):
    """
    Shuffle the given attributes of a dataset across multiple processes. This will send half of the data to rank + 1.
    Once the new data is received, it will be shuffled into the existing data on the process.
    This function will be called by the DataLoader automatically if ``dataset.ishuffle = False``.
    attrs should have the form [[torch.Tensor, DNDarray], ... i.e. [['data', 'htdata`]] assume that all of the attrs have the same dim0 shape as the local data

    Parameters
    ----------
    dataset : Dataset
        the dataset to shuffle
    attrs : List[List[str, str], ... ]
        List of lists each of which contains 2 strings. The strings are the handles corresponding to the Dataset
        attributes corresponding to the global data DNDarray and the local data of that array, i.e. [["data, "htdata"],]
        would shuffle the htdata around and set the correct amount of data for the ``dataset.data`` attribute. For
        multiple parameters multiple lists are required. I.e. [["data", "htdata"], ["targets", "httargets"]]

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.
    """
    # attrs -> [[torch.Tensor, DNDarray], ...]
    if attrs[0][1] is not None:
        prm = torch.randperm(getattr(dataset, attrs[0][1])._DNDarray__array.shape[0])
    else:
        prm = torch.randperm(getattr(dataset, attrs[0][0]).shape[0])
    comm = dataset.comm
    for att in attrs:
        ld = getattr(dataset, att[0])
        snd = ld[: dataset.lcl_half].clone()
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
        # set the DNDarray data
        if att[1] is not None:
            getattr(dataset, att[1])._DNDarray__array[: dataset.lcl_half] = new_data
            # shuffle all of the data around
            shuffled = getattr(dataset, att[1])._DNDarray__array[prm]
            getattr(dataset, att[1])._DNDarray__array = shuffled
            # set the torch data
            setattr(dataset, att[0], shuffled[dataset._cut_slice])
        else:
            getattr(dataset, att[0])[: dataset.lcl_half] = new_data
            # shuffle all of the data around
            shuffled = getattr(dataset, att[0])[prm]
            setattr(dataset, att[0], shuffled[dataset._cut_slice])


def dataset_ishuffle(dataset: Union[Dataset, torch_data.Dataset], attrs: List[list]):
    """
    Shuffle the given attributes of a dataset across multiple processes, using non-blocking communications.
    This will send half of the data to rank + 1. The data must be received by the :func:`dataset_irecv` function.

    This function will be called by the DataLoader automatically if ``dataset.ishuffle = True``. This is set either
    during the definition of the class of its initialization by a given paramete.

    Parameters
    ----------
    dataset : Dataset
        the dataset to shuffle
    attrs : List[List[str, str], ... ]
        List of lists each of which contains 2 strings. The strings are the handles corresponding to the Dataset
        attributes corresponding to the global data DNDarray and the local data of that array, i.e. [["htdata, "data"],]
        would shuffle the htdata around and set the correct amount of data for the ``dataset.data`` attribute. For
        multiple parameters multiple lists are required. I.e. [["htdata", "data"], ["httargets", "targets"]]

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.
    """
    # attrs should have the form [[heat array, sliced array], [...], ...]
    #       i.e. [['data', 'htdata']]
    # assume that all of the attrs have the same dim0 shape as the local data
    comm = dataset.comm
    ret_list = []
    for att in attrs:
        snd = getattr(dataset, att[0])[: dataset.lcl_half].clone()
        snd_shape, snd_dtype, snd_dev = snd.shape, snd.dtype, snd.device
        dest = comm.rank + 1 if comm.rank + 1 != comm.size else 0
        # send the top half of the data to the next process
        send_wait = comm.Isend(snd, dest=dest, tag=99999)
        new_data = torch.empty(snd_shape, dtype=snd_dtype, device=snd_dev)
        src = comm.rank - 1 if comm.rank != 0 else comm.size - 1
        wait = comm.Irecv(new_data, source=src, tag=99999)
        ret_list.append([att, wait, new_data])
        send_wait.wait()
        del snd
    setattr(dataset, "rcv_list", ret_list)


def dataset_irecv(dataset: Union[Dataset, torch_data.Dataset]):
    """
    Receive the data sent by the :func:`dataset_ishuffle` function. This will wait for the data and then shuffle the
    data into the existing data on the process

    This function will be called by the DataLoader automatically if ``dataset.ishuffle = True``. This is set either
    during the definition of the class of its initialization by a given paramete.

    Parameters
    ----------
    dataset : Dataset
        the dataset to shuffle

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.
    """
    setattr(dataset, "shuffle_prm", torch.randperm(dataset.data.shape[0]))
    rcv_list = getattr(dataset, "rcv_list")
    prm = getattr(dataset, "shuffle_prm")
    for rcv in rcv_list:
        rcv[1].wait()
        if rcv[0][1] is not None:
            getattr(dataset, rcv[0][1])._DNDarray__array[: dataset.lcl_half] = rcv[2]
            # shuffle all of the data around
            shuffled = getattr(dataset, rcv[0][1])._DNDarray__array[prm]
            getattr(dataset, rcv[0][1])._DNDarray__array = shuffled
            # set the torch data
            setattr(dataset, rcv[0][0], shuffled[dataset._cut_slice])
        else:
            getattr(dataset, rcv[0][0])[: dataset.lcl_half] = rcv[2]
            # shuffle all of the data around
            shuffled = getattr(dataset, rcv[0][0])[prm]
            setattr(dataset, rcv[0][0], shuffled[dataset._cut_slice])
