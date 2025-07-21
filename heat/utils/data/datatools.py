"""
Function and classes useful for loading data into neural networks
"""

from functools import reduce
import itertools
import random
import warnings
import mpi4py
import torch
import torch.distributed
from torch.utils import data as torch_data
from typing import Callable, List, Iterator, Literal, Union, Optional, Sized

import torch.utils
import torchvision

from ...core.dndarray import DNDarray
from ...core.communication import CUDA_AWARE_MPI, MPI_WORLD, MPICommunication
from . import partial_dataset

__all__ = [
    "DataLoader",
    "Dataset",
    "dataset_shuffle",
    "dataset_ishuffle",
    "DistributedDataset",
    "DistributedSampler",
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


class DistributedDataset(torch_data.Dataset):
    """
    A DistributedDataset for usage in PyTorch. Saves the dndarray and the larray tensor. Uses the larray tensor
    for the distribution and getting the items. Intented to be used with DistributedSampler.
    """

    def __init__(self, dndarray: DNDarray, transforms: torchvision.transforms.Compose = None):
        if not isinstance(dndarray, DNDarray):
            raise TypeError(f"Expected DNDarray but got {type(dndarray)}")
        if dndarray.split != 0:
            raise ValueError("DistributedDataset only works with a DNDarray split of 0")

        self.dndarray = dndarray
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.dndarray.larray)

    def __getitem__(self, index):
        item = self.dndarray.larray[index]
        if self.transforms is not None:
            return self.transforms(item)
        return item

    def __getitems__(self, indices):
        if self.transforms is not None:
            return tuple(self.transforms(self.dndarray.larray[index]) for index in indices)
        return tuple(self.dndarray.larray[index] for index in indices)


class DistributedSampler(torch_data.Sampler):
    """
    A DistributedSampler for usage in PyTorch with Heat Arrays. Uses the nature of the Heat DNDArray
    to give the locally stored data on the larray. Shuffling is done by shuffling the indices.
    The given Indices corrospond to the index of the larray tensor.
    Works only with DNDarray that are split on axis 0
    """

    def __init__(
        self,
        dataset: DistributedDataset,
        shuffle: bool = False,
        seed: Optional[int] = None,
        shuffle_type: Literal["global"] | Literal["local"] = "global",
        correction: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        dataset : DistributedDataset
            Dataset to be shuffled
        shuffle : bool, optional
            If the underlying DNDarray should be shuffled, by default False
        seed : int, optional
            seed for shuffling, by default None
        shuffle_type : Literal[&quot;global&quot;] | Literal[&quot;local&quot;], optional
            Wether to shuffle process local or get new data using by shuffling globally across all processes, by default "global"
        correction : bool, optional
            If index correction is wanted after an global shuffle, by default False
        """
        if not isinstance(dataset, DistributedDataset):
            raise TypeError(f"Expected DistributedDataset for dataset not {type(dataset)}")
        if not isinstance(shuffle, bool):
            raise TypeError(f"Expected bool for shuffle not {type(shuffle)}")
        if not isinstance(seed, int) and seed is not None:
            raise TypeError(f"Expected int or None for seed not {type(shuffle)}")
        if not isinstance(shuffle_type, str):
            raise TypeError("Shuffle Type needs to be an string")
        if not isinstance(correction, bool):
            raise TypeError("Correction Parameter needs to be an bool")

        self.dataset = dataset
        self.dndarray = dataset.dndarray
        self.shuffle = shuffle
        self.linked_sampler = None
        self.correction = correction
        self.set_shuffle_type(shuffle_type)
        self.set_seed(seed)

        if self.dndarray.split != 0:
            raise ValueError("DistributedSampler only works with a DNDarray split of 0")

    @staticmethod
    def _in_slice(idx: int, a_slice: slice) -> bool:
        """Check if the given index is inside the given slice

        Parameters
        ----------
        idx : int
            Index to check
        a_slice : slice
            Slice to check

        Returns
        -------
        bool
            Wether index is in slice
        """
        if idx < a_slice.start or idx >= a_slice.stop:
            return False
        step = a_slice.step if a_slice.step else 1
        if (idx - a_slice.start) % step == 0:
            return True
        else:
            return False

    def _shuffle(self) -> None:
        """Shuffles the given dndarray at creation across processes."""
        if self.shuffle_type == "local":
            return

        if self.shuffle_type != "global":
            raise ValueError("Shuffle type is not 'local' nor 'global'")

        dtype = self.dndarray.dtype.torch_type()
        comm: MPICommunication = self.dndarray.comm
        rank: int = comm.rank
        world_size: int = comm.size
        N: int = self.dndarray.gshape[0]
        mpi_type: mpi4py.MPI.Datatype = comm._MPICommunication__mpi_type_mappings[dtype]

        if rank == 0:
            indices = torch.randperm(N, dtype=torch.int64)
        else:
            indices = torch.empty(N, dtype=torch.int64)
        mpi4py.MPI.COMM_WORLD.Bcast(indices, root=0)

        indice_buffers: List[List[int]] = [list() for _ in range(world_size)]
        rank_slices: List[slice] = [
            comm.chunk((N,), split=0, rank=i)[-1][0] for i in range(world_size)
        ]

        block_length: int = reduce(lambda a, b: a * b, self.dndarray.gshape[1:], 1)
        local_slice: slice = rank_slices[rank]
        local_displacement: int = self.dndarray.counts_displs()[1][rank] * block_length

        # Now figure out which rank needs to send what to each rank and what this rank will receive
        for i, idx in enumerate(indices):
            idx = idx.item()
            for data_send_rank, tslice in enumerate(rank_slices):
                if not self._in_slice(idx, tslice):
                    continue
                break
            for data_recv_rank, tslice in enumerate(rank_slices):
                if not self._in_slice(i, tslice):
                    continue
                break
            if data_recv_rank == rank:
                indice_buffers[rank].append(idx)
            elif data_send_rank == rank:
                indice_buffers[data_recv_rank].append(idx)

        # print("RECV BUFFER creating...", flush=True)
        send_elems_dtype: List[mpi4py.MPI.Datatype] = list()
        local_recv_buffer: torch.Tensor = torch.empty(self.dndarray.larray.shape, dtype=dtype)

        for current_rank in range(world_size):
            if current_rank == rank:
                send_indice = [
                    idx for idx in indice_buffers[current_rank] if self._in_slice(idx, local_slice)
                ]
            else:
                send_indice = indice_buffers[current_rank]
            displacements = [
                mpi_type.Get_size() * (disp * block_length - local_displacement)
                for disp in send_indice
            ]
            block_lengths = [block_length] * len(displacements)
            send_type = mpi_type.Create_struct(
                blocklengths=block_lengths,
                displacements=displacements,
                datatypes=[mpi_type] * len(displacements),
            )
            send_type.Commit()
            send_elems_dtype.append(send_type)

        recv_counts = torch.zeros(world_size, dtype=torch.int64)
        for idx in indice_buffers[rank]:
            for i, tslice in enumerate(rank_slices):
                if not self._in_slice(idx, tslice):
                    continue
                recv_counts[i] += 1
                break

        send_elems = self.dndarray.larray
        send_elems = send_elems if CUDA_AWARE_MPI else send_elems.cpu()

        recv_types: List[mpi4py.MPI.Datatype] = []

        total_displ = 0

        for i in range(world_size):
            if recv_counts[i] == 0:
                recv_type = mpi_type.Create_contiguous(0)
            else:
                types = [mpi_type.Create_contiguous(block_length) for _ in range(recv_counts[i])]

                displ = torch.zeros(len(types))
                displ[1:] = torch.cumsum(torch.tensor([t.Get_size() for t in types])[:-1], 0)
                displ += total_displ

                recv_type = mpi_type.Create_struct(
                    blocklengths=[1] * len(types), displacements=displ, datatypes=types
                )
                total_displ += sum([t.Get_size() for t in types])

            recv_type.Commit()
            recv_types.append(recv_type)

        mpi4py.MPI.COMM_WORLD.Alltoallw(
            (send_elems, send_elems_dtype),
            (local_recv_buffer, recv_types),
        )

        for elem in itertools.chain(recv_types, send_elems_dtype):
            elem.Free()

        # As MPI indirectly sorts the data according to the rank we need
        # to change that to represent the permutation
        if self.correction:

            def get_from_rank(idx):
                for i, rslice in enumerate(rank_slices):
                    if self._in_slice(idx, rslice):
                        return i
                raise RuntimeError("IDX not found in slices")

            idx_to_rank_map = [get_from_rank(idx) for idx in indices[local_slice]]

            sort_idx = torch.argsort(torch.tensor(idx_to_rank_map), stable=True)
            local_slices_sorted = indices[local_slice][sort_idx]

            reverse_index = {idx.item(): i for i, idx in enumerate(indices[local_slice])}
            idxmap = {i: reverse_index[idx.item()] for i, idx in enumerate(local_slices_sorted)}

            for i, dest in idxmap.items():
                self.dndarray.larray[dest] = local_recv_buffer[i].to(self.dndarray.larray.device)
        else:
            self.dndarray.larray = local_recv_buffer.to(self.dndarray.larray.device)

    def set_shuffle_type(self, shuffle_type: Literal["global"] | Literal["local"]) -> None:
        """Sets the Shuffle type for the Sampler.

        Parameters
        ----------
        shuffle_type : Literal[&quot;global&quot;] | Literal[&quot;local&quot;]
            - Local Shuffle means the shuffle of the larray only.
            - Global Shuffle means the shuffle across all processes

        Raises
        ------
        TypeError
            Shuffle type needs to be a string
        ValueError
            Only Global/Local shuffle types exist
        """
        if not isinstance(shuffle_type, str):
            raise TypeError("Shuffle type needs to be an string")
        if not (shuffle_type == "global" or shuffle_type == "local"):
            raise ValueError("only 'global' or 'local' allowed as shuffle type")

        self.shuffle_type: Literal["global"] | Literal["local"] = shuffle_type

        if self.linked_sampler is not None:
            self.linked_sampler.set_shuffle_type(shuffle_type)

    def set_seed(self, value: int | None) -> None:
        """Sets the seed for the torch.randperm

        Parameters
        ----------
        value : int
            seed to set
        """
        self._seed = value
        if value is not None:
            torch.manual_seed(value)
        if self.shuffle:
            self._shuffle()

        if self.linked_sampler is not None:
            self.linked_sampler.set_seed(value)

    def link(self, sampler: "DistributedSampler") -> None:
        """
        Links another DistributedSampler to this one, to automatically sets the seed/shuffle_type of this and the linked one,
        rather than manually setting both seperately. Usefull when one Sampler contains training data and the
        linked one the label data.
        """
        if not isinstance(sampler, DistributedSampler):
            raise TypeError(f"Sampler of type {type(sampler)} needs to be an DistributedSampler")
        self.linked_sampler = sampler

    def unlink(self) -> None:
        """
        Removes an established link. For more info view :link: function
        """
        self.linked_sampler = None

    def __iter__(self) -> Iterator[int]:
        if self.shuffle_type == "local":
            self.indices = torch.randperm(len(self.dndarray.larray)).tolist()
        else:
            self.indices = list(range(len(self.dndarray.larray)))
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.dndarray.larray)


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
