"""
Function and classes useful for loading data into neural networks
"""

from collections import deque
from dataclasses import dataclass
import random
import torch
import torch.distributed
from torch.utils import data as torch_data
from typing import Callable, List, Iterator, Union, Optional
from mpi4py import MPI

import torch.utils

from ...core.dndarray import DNDarray
from ...core.communication import MPICommunication
from ...core.manipulations import take, Itake, TakeRequest, moveaxis
from . import partial_dataset

__all__ = [
    "DataLoader",
    "Dataset",
    "dataset_shuffle",
    "dataset_ishuffle",
    "DistributedDataset",
    "create_train_val_split",
    "DistributedDataLoader",
]


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
        if hasattr(self.dataset, "ishuffle"):
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
        if hasattr(self, "_full_dataset_shuffle_iter") and hasattr(self.dataset, "ishuffle"):
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
        Basic form of __getitem__. As the dataset is often very specific to the dataset,
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


def create_train_val_split(
    *data: DNDarray,
    p: float = 0.95,
    seed: int | None = None,
    sample_axis: int | tuple[int, ...] = 0,
) -> tuple[DNDarray, ...]:
    """Shuffle aligned arrays and split each into training and validation sets.

    Parameters
    ----------
    *data : DNDarray
        One or more arrays with at least one dimension and the same positive global
        number of samples along their sample axes. The arrays may be split along any
        axis. The same permutation is applied to every array to preserve alignment.
        Their communicators must have the same rank and world size.
    p : float, optional
        Fraction of samples assigned to training, in the interval [0, 1]. Default
        is 0.95. For N input samples, the training set contains ``int(N * p)`` samples
        and the validation set contains the remaining samples. Either set may be empty.
    seed : int or None, optional
        Seed for the local PyTorch generator. Equal seeds produce equal permutations
        without modifying the global PyTorch random state. If None, a random seed
        from rank 0 is broadcast using the first input array's communicator.
        Default is None.
    sample_axis : int or tuple of int, optional
        Axis containing samples in each input. An integer applies to all inputs; a tuple
        specifies one axis per input. Negative axes are supported. Default is 0.

    Returns
    -------
    tuple of DNDarray
        Training and validation arrays for each input, in input order:
        ``(data0_train, data0_val, data1_train, data1_val, ...)``. Each output
        preserves the corresponding input's axis order, dtype, device, communicator,
        and split axis.

    Raises
    ------
    TypeError
        If an input is not a DNDarray or a sample axis is not an integer. Boolean
        sample axes are not accepted.
    ValueError
        If no arrays are given, an input has no dimensions or no global samples, the
        sample counts differ, a sample axis is out of range, the axis tuple has the
        wrong length, the communicators have different ranks or world sizes, or ``p``
        is outside [0, 1].

    Notes
    -----
    All processes holding distributed parts of the arrays must call this function
    with matching parameters. Explicit seeds must be equal across these processes.
    When ``seed`` is None, all processes in the first input array's communicator must
    participate in the call, including when the input arrays are unsplit.
    """
    data, sample_axes = _sanitize_data_input(data, sample_axis=sample_axis)
    comm = _sanitize_communicators(*(array.comm for array in data))

    if not 0 <= p <= 1:
        raise ValueError(f"{p=} must be within [0, 1]")

    if seed is None:
        seed = random.randint(-0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF)
        seed = comm.bcast(seed, root=0)

    num_samples = data[0].gshape[sample_axes[0]]
    train_rows = int(num_samples * p)
    val_rows = num_samples - train_rows

    assert train_rows + val_rows == num_samples  # sanity check

    generator = torch.Generator().manual_seed(seed)

    perm = torch.randperm(num_samples, generator=generator)

    train_idx = perm[:train_rows]
    val_idx = perm[train_rows:]

    return tuple(
        part
        for arr, axis in zip(data, sample_axes)
        for part in (take(arr, train_idx, axis=axis), take(arr, val_idx, axis=axis))
    )


def _sanitize_data_input(
    data: DNDarray | tuple[DNDarray, ...],
    *,
    sample_axis: int | tuple[int, ...] = 0,
) -> tuple[tuple[DNDarray, ...], tuple[int, ...]]:
    """
    Check the input arrays and normalize their sample axes.

    Parameters
    ----------
    data : DNDarray or tuple of DNDarray
        One or more arrays with at least one dimension and the same positive global
        number of samples. The arrays may be split along different axes or be unsplit.
    sample_axis : int or tuple of int, optional
        Sample axis shared by all inputs or one axis per input. Negative axes are
        supported. Default is 0.

    Returns
    -------
    arrays : tuple of DNDarray
        The input arrays in their original order, without modifying their layout or data.
    sample_axes : tuple of int
        Non-negative sample axes, one per input array.

    Raises
    ------
    TypeError
        If ``data`` is not a DNDarray or a tuple of DNDarrays, or a sample axis is not
        an integer. Boolean sample axes are not accepted.
    ValueError
        If the input tuple is empty, an array has no dimensions or no global samples,
        the sample counts differ, a sample axis is out of range, or the axis tuple has
        the wrong length.
    """
    if isinstance(data, DNDarray):
        arrays = (data,)
    elif isinstance(data, tuple):
        arrays = data
    else:
        raise TypeError("'data' must be a DNDarray or a tuple of DNDarrays.")

    if len(arrays) == 0:
        raise ValueError("'data' cannot be an empty tuple.")

    if not all(isinstance(x, DNDarray) for x in arrays):
        raise TypeError("Every element of 'data' must be a DNDarray.")

    if any(x.ndim == 0 for x in arrays):
        raise ValueError("Data must have at least one dimension.")

    if isinstance(sample_axis, int):
        axes = (sample_axis,) * len(arrays)
    elif isinstance(sample_axis, tuple):
        axes = sample_axis
    else:
        raise TypeError("'sample_axis' must be an integer or a tuple of integers.")

    if len(axes) != len(arrays):
        raise ValueError("'sample_axis' must contain one axis per input array.")
    if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axes):
        raise TypeError("Sample axes must be integers, not booleans.")
    if any(not -arr.ndim <= axis < arr.ndim for arr, axis in zip(arrays, axes)):
        raise ValueError("Each sample axis must be in range for its input array.")

    axes = tuple(axis % arr.ndim for arr, axis in zip(arrays, axes))
    if len({arr.gshape[axis] for arr, axis in zip(arrays, axes)}) != 1:
        raise ValueError("All DNDarrays must have the same global number of samples.")
    if arrays[0].gshape[axes[0]] == 0:
        raise ValueError("Input arrays must contain at least one sample.")

    return arrays, axes


def _local_batch(batch: DNDarray, sample_axis: int, distributed: bool) -> torch.Tensor:
    """
    Assemble complete local samples with the sample axis first.

    Parameters
    ----------
    batch : DNDarray
        Newly selected batch, with indices grouped by process when distributed.
    sample_axis : int
        Axis containing the selected samples.
    distributed : bool
        Whether to distribute the samples across processes, including for replicated inputs.

    Returns
    -------
    torch.Tensor
        Local samples with their remaining dimensions in their original relative order.
    """
    if sample_axis != 0:
        batch = moveaxis(batch, sample_axis, 0)
    if distributed and batch.split != 0:
        batch.resplit_(0)
    return batch.larray


def _sanitize_communicators(*comms: MPICommunication) -> MPICommunication:
    """
    Check that communicators have the same rank and world size and return the first.

    Parameters
    ----------
    *comms : MPICommunication
        One or more communicators to validate.

    Returns
    -------
    MPICommunication
        The first input communicator.

    Raises
    ------
    ValueError
        If no communicators are provided, or if their ranks or world sizes differ.
    """
    if not comms:
        raise ValueError("One or more Communicators must be passed.")

    ranks = {comm.rank for comm in comms}
    sizes = {comm.size for comm in comms}

    if len(ranks) != 1:
        raise ValueError("All communicators must have the same rank.")
    if len(sizes) != 1:
        raise ValueError("All communicators must have the same world size.")

    return comms[0]


def _create_distributed_indices(
    comm: MPICommunication,
    num_elements: int,
    shuffle: bool,
    seed: int,
    *,
    distributed: bool = True,
    drop_last: bool = False,
) -> torch.Tensor:
    """
    Create sample indices for each process.

    Parameters
    ----------
    comm : MPICommunication
        Communication object used to broadcast the global indices from rank 0.
    num_elements : int
        Number of samples in the global dataset. Must be positive.
    shuffle : bool
        If ``True``, create a random permutation of the sample indices. Otherwise, return
        the indices in increasing order.
    seed : int
        Seed of a dedicated torch random number generator. Only used if ``shuffle`` is
        ``True``. For distributed indices, rank 0's seed determines the global order.
    distributed : bool, optional
        If ``True``, return one row of indices for each process. Otherwise, return only the
        local indices in a single row. Default is ``True``.
    drop_last : bool, optional
        If ``True``, truncate the global indices to a multiple of the number of processes.
        Otherwise, repeat indices from the beginning to reach the next multiple. Only
        applies to distributed indices. Default is ``False``.

    Returns
    -------
    torch.Tensor
        A CPU tensor of dtype ``torch.int64`` and shape ``(comm.size, samples_per_rank)``
        if ``distributed`` is ``True``, or ``(1, num_elements)`` otherwise. Each row
        contains a contiguous portion of the global sample order for that process.

    Raises
    ------
    ValueError
        If ``num_elements`` is zero or negative.

    Notes
    -----
    For distributed indices, rank 0 creates the global order and broadcasts it to all
    processes before partitioning. All processes in ``comm`` must participate with the
    same ``num_elements`` and ``drop_last``. No communication is needed for non-distributed
    indices. The global torch random number generator state is unchanged.
    """
    N = num_elements
    if N <= 0:
        raise ValueError("num_elements must be greater than 0.")
    P = comm.size

    if shuffle:
        idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    else:
        idx = torch.arange(N, dtype=torch.int64)

    if not distributed:
        return idx.unsqueeze(0)

    if P > 1:
        comm.Bcast(idx, root=0)

    if drop_last:
        idx = idx[: N // P * P]
    else:
        total_size = (N + P - 1) // P * P
        padding_size = total_size - N
        if padding_size > 0:
            padding = idx.repeat((padding_size + N - 1) // N)[:padding_size]
            idx = torch.cat((idx, padding))

    return idx.reshape(P, -1).contiguous()


def _batch_indices(
    indices: torch.Tensor,
    batch_size: int,
    drop_last: bool,
) -> Iterator[torch.Tensor]:
    """
    Generate batches from the sample indices of each process.

    Parameters
    ----------
    indices : torch.Tensor
        Two-dimensional tensor of shape ``(P, N)``, where each row contains the sample
        indices for one process.
    batch_size : int
        Number of samples per process in each batch. Must be positive.
    drop_last : bool
        If ``True``, skip the last batch if it contains fewer than ``batch_size`` samples
        per process.

    Yields
    ------
    torch.Tensor
        A one-dimensional tensor containing the indices of a batch. The indices of each
        process are kept together in row order. A full batch contains ``P * batch_size``
        indices; the last batch may contain fewer if ``drop_last`` is ``False``.
    """
    for start in range(0, indices.shape[1], batch_size):
        batch = indices[:, start : start + batch_size]

        if drop_last and batch.shape[1] < batch_size:
            break

        yield batch.flatten()


class DistributedDataset[T](torch.utils.data.Dataset):
    """
    Create a PyTorch dataset from one or more DNDarrays.

    Samples are retrieved along each input's sample axis and returned with samples on
    axis 0. If any input is distributed, the requested indices are gathered from each
    process and complete samples are distributed back to the processes. If all inputs
    are unsplit, they are indexed locally.

    Parameters
    ----------
    data : DNDarray
        One or more arrays passed as positional arguments. The arrays must have at least
        one dimension and the same positive global number of samples. They may be split
        along any axis, including different axes, or be unsplit. Their communicators must
        have the same rank and world size.
    sample_axis : int or tuple of int, optional
        Axis containing samples in each input. An integer applies to all inputs; a tuple
        specifies one axis per input. Negative axes are supported. Default is 0.
    post_processing : Callable, optional
        Function called for each local batch with the position of the input array and the
        corresponding torch tensor, i.e. ``post_processing(i, batch)``. Its return value
        replaces that batch. Default is ``None``.

    Raises
    ------
    TypeError
        If an input is not a DNDarray, ``post_processing`` is not callable or ``None``,
        or a sample axis is not an integer. Boolean sample axes are not accepted.
    ValueError
        If no arrays are given, an array has no dimensions or no global samples, the
        sample counts differ, a sample axis is out of range, the axis tuple has the
        wrong length, or the communicators have different ranks or world sizes.

    Notes
    -----
    Batch retrieval returns a tensor for one input array and a tuple for multiple arrays.
    If ``post_processing`` is given, the returned batches have the type produced by that
    function. Before post-processing, batches have shape ``(batch_size, ...)``, with
    the non-sample dimensions in their original relative order. Inputs retain their
    layout and distribution; feature-split inputs are redistributed one batch at a time.

    Use a torch DataLoader with ``num_workers=0`` and an identity ``collate_fn``, as this
    dataset already returns assembled batches. If any input is distributed, all processes
    must retrieve batches in the same order and request the same number of samples in
    each batch, including for replicated inputs in the same call.

    Examples
    --------
    >>> import heat as ht
    >>> from torch.utils.data import DataLoader
    >>> data = ht.arange(8, split=0)
    >>> dataset = ht.utils.data.DistributedDataset(data)
    >>> loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=lambda batch: batch)
    >>> batches = list(loader)

    To assign each process a subset of the global dataset, use PyTorch's
    ``DistributedSampler`` as the DataLoader's ``sampler`` argument with the MPI
    communicator's size and rank. Use the same seed on all processes. For multiple epochs,
    call ``loader.sampler.set_epoch(epoch)`` before iterating the loader to change the
    shuffle order. By default, the sampler repeats some indices when the sample count is not
    divisible by the number of processes; use ``drop_last=True`` on the sampler to drop
    the remainder instead.

    >>> from torch.utils.data import DistributedSampler
    >>> loader = DataLoader(
    ...     dataset,
    ...     batch_size=2,
    ...     sampler=DistributedSampler(
    ...         dataset,
    ...         num_replicas=data.comm.size,
    ...         rank=data.comm.rank,
    ...         shuffle=True,
    ...         seed=0,
    ...     ),
    ...     num_workers=0,
    ...     collate_fn=lambda batch: batch,
    ... )
    >>> batches = list(loader)
    """

    def __init__(
        self,
        *data: DNDarray,
        sample_axis: int | tuple[int, ...] = 0,
        post_processing: Callable[[int, torch.Tensor], T] | None = None,
    ) -> None:
        if post_processing is not None and not callable(post_processing):
            raise TypeError("post_processing must be a callable function or None.")

        self._data, self._sample_axes = _sanitize_data_input(data, sample_axis=sample_axis)
        self._distributed = any(array.is_distributed() for array in self._data)

        self._comm: MPICommunication = _sanitize_communicators(*(d.comm for d in self._data))
        self._post_processing = post_processing

    def __getitems__(
        self, indices: list[int]
    ) -> Union[torch.Tensor, T, tuple[torch.Tensor, ...], tuple[T, ...]]:
        """
        Get a local batch of samples from each input array.

        Parameters
        ----------
        indices : list of int
            Global indices requested by this process along each input's sample axis.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor or T or tuple of T
            The local batch for each input array, after post-processing if given. A single
            input returns one batch; multiple inputs return a tuple in their original order.

        Raises
        ------
        ValueError
            If any input is distributed and processes request different numbers of indices.

        Notes
        -----
        If any input is distributed, all processes must call this method with the same
        number of indices so that the gathered samples can be distributed evenly.
        The counts are checked collectively before gathering the indices.
        """
        indices: torch.Tensor = torch.tensor(indices, dtype=torch.int64)

        if self._distributed:
            counts = self._comm.allgather(indices.numel())
            if len(set(counts)) != 1:
                raise ValueError("All processes must request the same number of sample indices.")
            index_buffer = torch.empty(indices.numel() * self._comm.size, dtype=torch.int64)
            self._comm.Allgather(indices, index_buffer)
        else:
            index_buffer = indices

        results = tuple(
            _local_batch(take(d, index_buffer, axis=axis), axis, self._distributed)
            for d, axis in zip(self._data, self._sample_axes)
        )

        if self._post_processing is not None:
            results = tuple(self._post_processing(i, result) for i, result in enumerate(results))

        return results if len(results) > 1 else results[0]

    def __len__(self) -> int:
        return self._data[0].gshape[self._sample_axes[0]]


class DistributedDataLoader[T]:
    """
    Create an iterable over batches from one or more DNDarrays.

    If any input is distributed, each process iterates over a portion of a shared global
    sample order, receiving ``batch_size`` complete samples in a full batch. Arrays are
    accessed using :func:`Itake <heat.core.manipulations.Itake>` requests. If all inputs
    are unsplit, they are indexed locally and each process traverses their complete
    sample order. Batches have samples on axis 0.

    Parameters
    ----------
    data : DNDarray
        One or more arrays passed as positional arguments. The arrays must have at least
        one dimension and the same positive global number of samples. They may be split
        along any axis, including different axes, or be unsplit. Their communicators must
        have the same rank and world size.
    batch_size : int
        Number of samples per process in each batch. Must be a positive integer.
    sample_axis : int or tuple of int, optional
        Axis containing samples in each input. An integer applies to all inputs; a tuple
        specifies one axis per input. Negative axes are supported. Default is 0.
    shuffle : bool, optional
        If ``True``, shuffle the global sample indices before each iteration.
        Default is ``False``.
    drop_last : bool, optional
        If ``True``, truncate distributed sample indices to give every process the same
        number of samples, then drop the last incomplete local batch. Otherwise, pad
        distributed indices by repeating samples from the beginning of the global order;
        the last batch may contain fewer than ``batch_size`` samples per process.
        Default is ``False``.
    prefetch_factor : int, optional
        Number of additional batches to queue beyond the current batch. Must be non-negative.
        The queue holds at most ``prefetch_factor + 1`` batches. Default is 0.
    seed : int, optional
        Base seed used to shuffle the sample indices. Each new iterator uses ``seed + epoch``
        and increments the epoch counter, which starts at 0. For distributed arrays,
        rank 0's seed and epoch determine the shared global order. Default is 0.
    post_processing : Callable, optional
        Function called for each local batch with the position of the input array and the
        corresponding torch tensor, i.e. ``post_processing(i, batch)``. Its return value
        replaces that batch. Default is ``None``.

    Attributes
    ----------
    seed : int
        Base seed for shuffling. Assigning a new value resets the epoch counter to 0.

    Raises
    ------
    TypeError
        If an input is not a DNDarray, ``post_processing`` is not callable or ``None``,
        or a sample axis is not an integer. Boolean sample axes are not accepted.
    ValueError
        If the arrays are missing, scalar, have no global samples or different sample
        counts, a sample axis is out of range, the axis tuple has the wrong length, or
        the communicators have different ranks or world sizes. Also raised if ``batch_size``
        is not a positive integer, ``prefetch_factor`` is not a non-negative integer,
        ``drop_last`` is not a bool, or ``seed`` is not an integer. Boolean values are not
        accepted for ``batch_size`` or ``seed``.

    Notes
    -----
    One input array returns a tensor per batch; multiple arrays return a tuple in their
    original order. If ``post_processing`` is given, the returned batches have the type
    produced by that function. The length of the loader is the number of batches per process.

    Before post-processing, batches have shape ``(batch_size, ...)``, with the non-sample
    dimensions in their original relative order. Inputs retain their layout and distribution.
    Selection along a split sample axis uses non-blocking communication. Feature-split
    inputs require additional redistribution of each selected batch when it is retrieved.

    If any input is distributed, all processes in its communicator must iterate together and use
    matching batch sizes, remainder policies, and iteration counts. Each process receives
    a contiguous portion of the shared global sample order. If an iteration is stopped
    early, close the iterator on all participating processes, or use it as a context
    manager to finish outstanding requests.

    Examples
    --------
    >>> import heat as ht
    >>> data = ht.arange(8, split=0)
    >>> labels = ht.arange(8, split=0)
    >>> loader = ht.utils.data.DistributedDataLoader(data, labels, batch_size=2, shuffle=True)
    >>> with iter(loader) as batches:
    ...     for data_batch, label_batch in batches:
    ...         pass
    """

    def __init__(
        self,
        *data: DNDarray,
        batch_size: int,
        sample_axis: int | tuple[int, ...] = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        prefetch_factor: int = 0,
        seed: int = 0,
        post_processing: Callable[[int, torch.Tensor], T] | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"'{batch_size=}' Value must be a positive integer.")
        if not isinstance(prefetch_factor, int) or prefetch_factor < 0:
            raise ValueError(f"'{prefetch_factor=}' must be a non-negative integer.")
        if not isinstance(drop_last, bool):
            raise ValueError(f"'{drop_last=}' must be a bool.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"'{seed=}' must be an integer.")
        if post_processing is not None and not callable(post_processing):
            raise TypeError("post_processing must be a callable function or None.")

        self._data, self._sample_axes = _sanitize_data_input(data, sample_axis=sample_axis)
        self._distributed = any(array.is_distributed() for array in self._data)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._prefetch_factor = prefetch_factor
        self._post_processing = post_processing

        self._comm: MPICommunication = _sanitize_communicators(*(d.comm for d in self._data))

        self._seed = seed
        self._epoch = 0

        self._num_elements = self._data[0].gshape[self._sample_axes[0]]

    @property
    def seed(self) -> int:
        """
        Returns the base seed used. Independent of the epoch counter.
        """
        return self._seed

    @property
    def epoch(self) -> int:
        """
        Returns the epoch counter.
        """
        return self._epoch

    @seed.setter
    def seed(self, value: int) -> None:
        """
        Set the base seed and reset the epoch counter.

        Parameters
        ----------
        value : int
            Seed used for the next iterator. Boolean values are not accepted.

        Raises
        ------
        ValueError
            If ``value`` is not an integer or is a bool.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Seed must be an integer not {type(value)}.")
        self._seed = value
        self._epoch = 0

    def __len__(self) -> int:
        N = self._num_elements
        B = self._batch_size

        if self._distributed:
            B *= self._comm.size

        if self._drop_last:
            return N // B
        else:
            return (N + B - 1) // B

    def __iter__(self) -> "_DistributedDataLoaderIter":
        """
        Create a new iterator using the base seed and the current epoch counter.
        Creating the iterator increments the epoch.
        """
        seed = self._seed + self._epoch
        self._epoch += 1
        return _DistributedDataLoaderIter(self, seed)


@dataclass
class _PendingBatch:
    """
    Store the index tensors and requests for a batch being loaded.

    Attributes
    ----------
    indices : tuple of torch.Tensor
        Index tensors kept alive until the requests are finished, one for each input device.
    requests : tuple of TakeRequest
        Pending :func:`Itake <heat.core.manipulations.Itake>` requests in input array order.
    """

    indices: tuple[torch.Tensor, ...]
    requests: tuple[TakeRequest, ...]


class _DistributedDataLoaderIter:
    """
    Iterate over local batches while queuing requests for upcoming batches.

    Parameters
    ----------
    loader : DistributedDataLoader
        Loader holding the input arrays, communicator, and batch loading parameters.
    seed : int
        Seed used to create the sample indices for this iteration.

    Notes
    -----
    The iterator holds at most ``loader._prefetch_factor + 1`` pending batches. Each batch
    contains one request per input array. Use :meth:`close` or a context manager to finish
    pending requests when iteration is stopped early. All participating processes must
    close their iterators consistently.
    """

    def __init__(
        self,
        loader: DistributedDataLoader,
        seed: int,
    ) -> None:
        self._loader = loader
        self._closed = False

        self._indices = _create_distributed_indices(
            comm=loader._comm,
            num_elements=loader._num_elements,
            shuffle=loader._shuffle,
            seed=seed,
            distributed=loader._distributed,
            drop_last=loader._drop_last,
        )

        self._batch_iterator = _batch_indices(
            indices=self._indices,
            batch_size=loader._batch_size,
            drop_last=loader._drop_last,
        )

        self._pending: deque[_PendingBatch] = deque()

        self._capacity = loader._prefetch_factor + 1

        try:
            self._fill_queue()
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return self

    def _start_batch(self, indices: torch.Tensor) -> _PendingBatch:
        """
        Start a sample selection request for each input array.

        Parameters
        ----------
        indices : torch.Tensor
            One-dimensional tensor of sample indices for the batch, grouped by process.

        Returns
        -------
        _PendingBatch
            The requests and the index tensors copied to the input devices. If starting a
            request fails, previously started requests are finished before the error is raised.
        """
        requests = []
        cached_indices_by_device: dict[torch.device, torch.Tensor] = {}

        try:
            # Itake normally handles the transfer of the index tensor, but this saves some unnecessary copies.
            for array, axis in zip(self._loader._data, self._loader._sample_axes):
                device = array.larray.device

                if device not in cached_indices_by_device:
                    cached_indices_by_device[device] = indices.to(device)

                requests.append(Itake(array, cached_indices_by_device[device], axis=axis))

        except BaseException:
            # When one of the request fail, wait for the MPI comms to finish
            for request in requests:
                request.Wait()
            raise

        return _PendingBatch(
            indices=tuple(indices_by_device.values()),
            requests=tuple(requests),
        )

    def _fill_queue(self) -> None:
        while len(self._pending) < self._capacity:
            try:
                indices = next(self._batch_iterator)
            except StopIteration:
                break

            self._pending.append(self._start_batch(indices))

    def _finish_batch(
        self,
        pending: _PendingBatch,
    ) -> tuple[torch.Tensor, ...]:
        """
        Wait for all requests of a batch and return the local tensors.

        Parameters
        ----------
        pending : _PendingBatch
            Batch containing the requests to finish.

        Returns
        -------
        tuple of torch.Tensor
            Local samples in input array order, before post-processing.

        Notes
        -----
        If a request fails, the remaining requests are waited on before the first error
        is raised.
        """
        results = []
        error = None

        for request in pending.requests:
            try:
                results.append(request.Wait())
            except Exception as exc:
                if error is None:
                    error = exc

        if error is not None:
            raise error

        return tuple(
            _local_batch(result, axis, self._loader._distributed)
            for result, axis in zip(results, self._loader._sample_axes)
        )

    def __next__(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Finish the next batch, refill the queue, and apply post-processing.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor or T or tuple of T
            Local batch for one input array, or a tuple of local batches for multiple arrays.
            If post-processing is given, return its results in input array order.

        Raises
        ------
        StopIteration
            If the iterator is closed or no batches remain.

        Notes
        -----
        If loading or post-processing fails, the iterator is closed to finish pending requests
        before the error is raised.
        """
        if self._closed:
            raise StopIteration

        if len(self._pending) == 0:
            self.close()
            raise StopIteration

        pending = self._pending.popleft()

        try:
            batch = self._finish_batch(pending)

            self._fill_queue()

            if self._loader._post_processing is not None:
                batch = tuple(
                    self._loader._post_processing(i, tensor) for i, tensor in enumerate(batch)
                )

            return batch if len(batch) > 1 else batch[0]

        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """
        Drain all outstanding Itake requests.

        All participating ranks must close their iterators
        consistently to avoid incomplete MPI collectives.
        """
        if self._closed:
            return

        self._closed = True
        error = None

        while self._pending:
            pending = self._pending.popleft()

            for request in pending.requests:
                try:
                    request.Wait()
                except Exception as exc:
                    if error is None:
                        error = exc

        self._batch_iterator.close()
        self._indices = None

        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
