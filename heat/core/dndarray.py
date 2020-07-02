from __future__ import annotations

import numpy as np
import math
import torch
import warnings
from typing import List, Dict, Any, TypeVar, Union, Tuple

from . import devices
from . import factories
from . import io
from . import linalg
from . import manipulations
from . import memory
from . import stride_tricks
from . import tiling

from .devices import Device
from .types import datatype, canonical_heat_type

from .communication import MPI, Communication
from .stride_tricks import sanitize_axis

warnings.simplefilter("always", ResourceWarning)

__all__ = ["DNDarray"]


class LocalIndex:
    """
    Indexing class for local operations (primarily for :func:`lloc` function)
    For docs on ``__getitem__`` and ``__setitem__`` see :func:`lloc`
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, value):
        self.obj[key] = value


class DNDarray:
    """
    Distributed N-Dimensional array. The core element of HeAT. It is composed of
    PyTorch tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor
        Local array elements
    gshape : tuple
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        The axis on which the array is divided between processes
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    """

    def __init__(self, array, gshape, dtype, split, device, comm):
        self.__array = array
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__ishalo = False
        self.__halo_next = None
        self.__halo_prev = None

        # handle inconsistencies between torch and heat devices
        if (
            isinstance(array, torch.Tensor)
            and isinstance(device, Device)
            and array.device.type != device.device_type
        ):
            self.__array = self.__array.to(devices.sanitize_device(self.__device).torch_device)

    @property
    def halo_next(self):
        return self.__halo_next

    @property
    def halo_prev(self):
        return self.__halo_prev

    @property
    def comm(self):
        """
        The :class:`~heat.core.communication.Communication` of the ``DNDarray``
        """
        return self.__comm

    @property
    def device(self):
        """
        The :class:`~heat.core.devices.Device` of the ``DNDarray``
        """
        return self.__device

    @property
    def dtype(self):
        """
        The :class:`~heat.core.types.datatype` of the ``DNDarray``
        """
        return self.__dtype

    @property
    def gshape(self):
        return self.__gshape

    @property
    def numdims(self) -> int:
        """
        Number of dimensions of the ``DNDarray``

        Returns
        -------
        number_of_dimensions : int
            The number of dimensions of the DNDarray

        .. deprecated:: 0.5.0
          ``numdims`` will be removed in HeAT 1.0.0, it is replaced by ``ndim`` because the latter is numpy API compliant.
        """
        warnings.warn("numdims is deprecated, use ndim instead", DeprecationWarning)
        return len(self.__gshape)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``DNDarray``

        """
        return len(self.__gshape)

    @property
    def size(self) -> int:
        """
        Number of total elements of the ``DNDarray``
        """
        try:
            return np.prod(self.__gshape)
        except TypeError:
            return 1

    @property
    def gnumel(self) -> int:
        """
        Number of total elements of the ``DNDarray``
        """
        return self.size

    @property
    def lnumel(self) -> int:
        """
        Number of elements of the ``DNDarray`` on each process
        """
        return np.prod(self.__array.shape)

    @property
    def lloc(self) -> Union[DNDarray, None]:
        """
        Local item setter and getter. i.e. this function operates on a local
        level and only on the PyTorch tensors composing the :class:`DNDarray`.
        This function uses the LocalIndex class. As getter, it returns a ``DNDarray``
        with the indices selected at a *local* level

        Parameters
        ----------
        key : int or slice or Tuple[int,...]
            Indices of the desired data.
        value : scalar, optional
            All types compatible with pytorch tensors, if none given then this is a getter function

        Examples
        --------
        >>> a = ht.zeros((4, 5), split=0)
        (1/2) tensor([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]])
        (2/2) tensor([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.]])
        >>> a.lloc[1, 0:4]
        (1/2) tensor([0., 0., 0., 0.])
        (2/2) tensor([0., 0., 0., 0.])
        >>> a.lloc[1, 0:4] = torch.arange(1, 5)
        >>> a
        (1/2) tensor([[0., 0., 0., 0., 0.],
                      [1., 2., 3., 4., 0.]])
        (2/2) tensor([[0., 0., 0., 0., 0.],
                      [1., 2., 3., 4., 0.]])
        """
        return LocalIndex(self.__array)

    @property
    def lshape(self) -> Tuple[int]:
        """
        Returns the shape of the ``DNDarray`` on each node
        """
        return tuple(self.__array.shape)

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns the shape of the ``DNDarray`` as a whole
        """
        return self.__gshape

    @property
    def split(self) -> int:
        """
        Returns the axis on which the ``DNDarray`` is split
        """
        return self.__split

    @property
    def stride(self) -> Tuple[int]:
        """
        Returns the steps in each dimension when traversing a ``DNDarray``. torch-like usage: ``self.stride()``
        """
        return self.__array.stride

    @property
    def strides(self) -> Tuple[int]:
        """
        Returns bytes to step in each dimension when traversing a ``DNDarray``. numpy-like usage: ``self.strides()``
        """
        steps = list(self._DNDarray__array.stride())
        itemsize = self._DNDarray__array.storage().element_size()
        strides = tuple(step * itemsize for step in steps)
        return strides

    @property
    def T(self):
        """
        Transpose the ``DNDarray``
        """
        return linalg.transpose(self, axes=None)

    @property
    def array_with_halos(self):
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next``/``self.halo_prev``
        in case they are not already stored. If ``halo_size`` differs from the size of already stored halos,
        the are overwritten.
        """
        return self.__cat_halo()

    def __prephalo(self, start, end) -> torch.Tensor:
        """
        Extracts the halo indexed by start, end from ``self.array`` in the direction of ``self.split``

        Parameters
        ----------
        start : int
            Start index of the halo extracted from ``self.array``
        end : int
            End index of the halo extracted from ``self.array``
        """
        ix = [slice(None, None, None)] * len(self.shape)
        try:
            ix[self.split] = slice(start, end)
        except IndexError:
            print("Indices out of bound")

        return self.__array[ix].clone().contiguous()

    def get_halo(self, halo_size):
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next/self.halo_prev``
        in case they are not already stored. If ``halo_size`` differs from the size of already stored halos,
        the are overwritten.

        Parameters
        ----------
        halo_size : int
            Size of the halo.
        """
        if not isinstance(halo_size, int):
            raise TypeError(
                "halo_size needs to be of Python type integer, {} given)".format(type(halo_size))
            )
        if halo_size < 0:
            raise ValueError(
                "halo_size needs to be a positive Python integer, {} given)".format(type(halo_size))
            )

        if self.comm.is_distributed() and self.split is not None:
            min_chunksize = self.shape[self.split] // self.comm.size
            if halo_size > min_chunksize:
                raise ValueError(
                    "halo_size {} needs to smaller than chunck-size {} )".format(
                        halo_size, min_chunksize
                    )
                )

            a_prev = self.__prephalo(0, halo_size)
            a_next = self.__prephalo(-halo_size, None)

            res_prev = None
            res_next = None

            req_list = list()

            if self.comm.rank != self.comm.size - 1:
                self.comm.Isend(a_next, self.comm.rank + 1)
                res_prev = torch.zeros(a_prev.size(), dtype=a_prev.dtype)
                req_list.append(self.comm.Irecv(res_prev, source=self.comm.rank + 1))

            if self.comm.rank != 0:
                self.comm.Isend(a_prev, self.comm.rank - 1)
                res_next = torch.zeros(a_next.size(), dtype=a_next.dtype)
                req_list.append(self.comm.Irecv(res_next, source=self.comm.rank - 1))

            for req in req_list:
                req.wait()

            self.__halo_next = res_prev
            self.__halo_prev = res_next
            self.__ishalo = True

    def __cat_halo(self) -> Tuple[torch.tensor, torch.tensor]:
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next``/``self.halo_prev``
        in case they are not already stored. If ``halo_size`` differs from the size of already stored halos,
        the are overwritten.

        """
        return torch.cat(
            [_ for _ in (self.__halo_prev, self.__array, self.__halo_next) if _ is not None],
            self.split,
        )

    def astype(self, dtype, copy=True) -> DNDarray:
        """
        Returns a casted version of this array.
        Casted array is a new array of the same shape but with given type of this array. If copy is ``True``, the
        same array is returned instead.

        Parameters
        ----------
        dtype : datatype
            HeAT type to which the array is cast
        copy : bool, optional
            By default the operation returns a copy of this array. If copy is set to false the cast is performed
            in-place and this array is returned

        """
        dtype = canonical_heat_type(dtype)
        casted_array = self.__array.type(dtype.torch_type())
        if copy:
            return DNDarray(casted_array, self.shape, dtype, self.split, self.device, self.comm)

        self.__array = casted_array
        self.__dtype = dtype

        return self

    def balance_(self):
        """
        Function for balancing a :class:`DNDarray` between all nodes. To determine if this is needed use the is_balanced function.
        If the ``DNDarray`` is already balanced this function will do nothing. This function modifies the ``DNDarray``
        itself and will not return anything.

        Examples
        --------
        >>> a = ht.zeros((10, 2), split=0)
        >>> a[:, 0] = ht.arange(10)
        >>> b = a[3:]
        [0/2] tensor([[3., 0.],
        [1/2] tensor([[4., 0.],
                      [5., 0.],
                      [6., 0.]])
        [2/2] tensor([[7., 0.],
                      [8., 0.],
                      [9., 0.]])
        >>> b.balance_()
        >>> print(b.gshape, b.lshape)
        [0/2] (7, 2) (1, 2)
        [1/2] (7, 2) (3, 2)
        [2/2] (7, 2) (3, 2)
        >>> b
        [0/2] tensor([[3., 0.],
                     [4., 0.],
                     [5., 0.]])
        [1/2] tensor([[6., 0.],
                      [7., 0.]])
        [2/2] tensor([[8., 0.],
                      [9., 0.]])
        >>> print(b.gshape, b.lshape)
        [0/2] (7, 2) (3, 2)
        [1/2] (7, 2) (2, 2)
        [2/2] (7, 2) (2, 2)
        """
        if self.is_balanced():
            return
        self.redistribute_()

    def __bool__(self) -> bool:
        """
        Boolean scalar casting.
        """
        return self.__cast(bool)

    def __cast(self, cast_function) -> Union[float, int]:
        """
        Implements a generic cast function for HeAT ``DNDarray`` objects.

        Parameters
        ----------
        cast_function : function
            The actual cast function, e.g. ``float`` or ``int``

        Raises
        ------
        TypeError
            If the ``DNDarray`` object cannot be converted into a scalar.

        """
        if np.prod(self.shape) == 1:
            if self.split is None:
                return cast_function(self.__array)

            is_empty = np.prod(self.__array.shape) == 0
            root = self.comm.allreduce(0 if is_empty else self.comm.rank, op=MPI.SUM)

            return self.comm.bcast(None if is_empty else cast_function(self.__array), root=root)

        raise TypeError("only size-1 arrays can be converted to Python scalars")

    def __complex__(self):
        """
        Complex scalar casting.
        """
        return self.__cast(complex)

    def cpu(self) -> DNDarray:
        """
        Returns a copy of this object in main memory. If this object is already in main memory, then no copy is
        performed and the original object is returned.
        """
        self.__array = self.__array.cpu()
        self.__device = devices.cpu
        return self

    def create_lshape_map(self) -> torch.Tensor:
        """
        Generate a 'map' of the lshapes of the data on all processes.
        Units are ``(process rank, lshape)``
        """
        lshape_map = torch.zeros(
            (self.comm.size, len(self.gshape)), dtype=torch.int, device=self.device.torch_device
        )
        lshape_map[self.comm.rank, :] = torch.tensor(self.lshape, device=self.device.torch_device)
        self.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)
        return lshape_map

    def __float__(self) -> float:
        """
        Float scalar casting.
        """
        return self.__cast(float)

    def fill_diagonal(self, value) -> DNDarray:
        """
        Fill the main diagonal of a 2D :class:`DNDarray`.
        This function modifies the input tensor in-place, and returns the input array.

        Parameters
        ----------
        value : float
            The value to be placed in the ``DNDarrays`` main diagonal
        """
        # Todo: make this 3D/nD
        if len(self.shape) != 2:
            raise ValueError("Only 2D tensors supported at the moment")

        if self.split is not None and self.comm.is_distributed:
            counts, displ, _ = self.comm.counts_displs_shape(self.shape, self.split)
            k = min(self.shape[0], self.shape[1])
            for p in range(self.comm.size):
                if displ[p] > k:
                    break
                proc = p
            if self.comm.rank <= proc:
                indices = (
                    displ[self.comm.rank],
                    displ[self.comm.rank + 1] if (self.comm.rank + 1) != self.comm.size else k,
                )
                if self.split == 0:
                    self._DNDarray__array[:, indices[0] : indices[1]] = self._DNDarray__array[
                        :, indices[0] : indices[1]
                    ].fill_diagonal_(value)
                elif self.split == 1:
                    self._DNDarray__array[indices[0] : indices[1], :] = self._DNDarray__array[
                        indices[0] : indices[1], :
                    ].fill_diagonal_(value)

        else:
            self._DNDarray__array = self._DNDarray__array.fill_diagonal_(value)

        return self

    def __getitem__(self, key) -> DNDarray:
        """
        Global getter function for DNDarrays.
        Returns a new DNDarray composed of the elements of the original tensor selected by the indices
        given. This does *NOT* redistribute or rebalance the resulting tensor. If the selection of values is
        unbalanced then the resultant tensor is also unbalanced!
        To redistributed the tensor use balance() (issue #187)

        Parameters
        ----------
        key : int, slice, Tuple[int,...], List[int,...]
            Indices to get from the tensor.

        Examples
        --------
        >>> a = ht.arange(10, split=0)
        (1/2) >>> tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        >>> a[1:6]
        (1/2) >>> tensor([1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5], dtype=torch.int32)
        >>> a = ht.zeros((4,5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1]
        (1/2) >>> tensor([0.])
        (2/2) >>> tensor([0., 0.])
        """
        l_dtype = self.dtype.torch_type()
        kgshape_flag = False
        if isinstance(key, DNDarray) and key.ndim == self.ndim:
            """ if the key is a DNDarray and it has as many dimensions as self, then each of the entries in the 0th
                dim refer to a single element. To handle this, the key is split into the torch tensors for each dimension.
                This signals that advanced indexing is to be used. """
            lkey = [slice(None, None, None)] * self.ndim
            kgshape_flag = True
            kgshape = [0] * len(self.gshape)
            if key.ndim > 1:
                for i in range(key.ndim):
                    kgshape[i] = key.gshape[i]
                    lkey[i] = key._DNDarray__array[..., i]
            else:
                kgshape[0] = key.gshape[0]
                lkey[0] = key._DNDarray__array
            key = tuple(lkey)
        elif not isinstance(key, tuple):
            """ this loop handles all other cases. DNDarrays which make it to here refer to advanced indexing slices,
                as do the torch tensors. Both DNDaarrys and torch.Tensors are cast into lists here by PyTorch.
                lists mean advanced indexing will be used"""
            h = [slice(None, None, None)] * self.ndim
            if isinstance(key, DNDarray):
                h[0] = key._DNDarray__array.tolist()
            elif isinstance(key, torch.Tensor):
                h[0] = key.tolist()
            else:
                h[0] = key
            key = tuple(h)

        gout_full = [None] * len(self.gshape)
        # below generates the expected shape of the output.
        #   If lists or torch.Tensors remain in the key, some change may be made later
        key = list(key)
        for c, k in enumerate(key):
            if isinstance(k, slice):
                new_slice = stride_tricks.sanitize_slice(k, self.gshape[c])
                gout_full[c] = math.ceil((new_slice.stop - new_slice.start) / new_slice.step)
            elif isinstance(k, list):
                gout_full[c] = len(k)
            elif isinstance(k, (DNDarray, torch.Tensor)):
                gout_full[c] = k.shape[0] if not kgshape_flag else kgshape[c]
            if isinstance(k, DNDarray):
                key[c] = k._DNDarray__array
        if all(g == 1 for g in gout_full):
            gout_full = [1]
        else:
            # delete the dimensions from gout_full if they are not touched (they will be removed)
            for i in range(len(gout_full) - 1, -1, -1):
                if gout_full[i] is None:
                    del gout_full[i]

        key = tuple(key)
        if not self.is_distributed():
            if not self.comm.size == 1:
                return factories.array(
                    self.__array[key],
                    dtype=self.dtype,
                    split=self.split,
                    device=self.device,
                    comm=self.comm,
                )
            else:
                gout = tuple(self.__array[key].shape)
                if self.split is not None and self.split >= len(gout):
                    new_split = len(gout) - 1 if len(gout) - 1 >= 0 else None
                else:
                    new_split = self.split

                return DNDarray(
                    self.__array[key], gout, self.dtype, new_split, self.device, self.comm
                )

        else:
            rank = self.comm.rank
            ends = []
            for pr in range(self.comm.size):
                _, _, e = self.comm.chunk(self.shape, self.split, rank=pr)
                ends.append(e[self.split].stop - e[self.split].start)
            ends = torch.tensor(ends, device=self.device.torch_device)
            chunk_ends = ends.cumsum(dim=0)
            chunk_starts = torch.tensor([0] + chunk_ends.tolist(), device=self.device.torch_device)
            chunk_start = chunk_starts[rank]
            chunk_end = chunk_ends[rank]
            arr = torch.Tensor()
            # all keys should be tuples here
            gout = [0] * len(self.gshape)
            # handle the dimensional reduction for integers
            ints = sum(isinstance(it, int) for it in key)
            gout = gout[: len(gout) - ints]
            if self.split >= len(gout):
                new_split = len(gout) - 1 if len(gout) - 1 > 0 else 0
            else:
                new_split = self.split
            if len(key) == 0:  # handle empty list
                # this will return an array of shape (0, ...)
                arr = self.__array[key]
                gout_full = list(arr.shape)

            """ At the end of the following if/elif/elif block the output array will be set.
                each block handles the case where the element of the key along the split axis is a different type
                and converts the key from global indices to local indices.
            """
            if isinstance(key[self.split], (list, torch.Tensor, DNDarray)):
                # advanced indexing, elements in the split dimension are adjusted to the local indices
                lkey = list(key)
                if isinstance(key[self.split], DNDarray):
                    lkey[self.split] = key[self.split]._DNDarray__array
                inds = (
                    torch.tensor(
                        lkey[self.split], dtype=torch.long, device=self.device.torch_device
                    )
                    if not isinstance(lkey[self.split], torch.Tensor)
                    else lkey[self.split]
                )

                loc_inds = torch.where((inds >= chunk_start) & (inds < chunk_end))
                if len(loc_inds[0]) != 0:
                    # if there are no local indices on a process, then `arr` is empty
                    inds = inds[loc_inds] - chunk_start
                    lkey[self.split] = inds
                    arr = self.__array[tuple(lkey)]
            elif isinstance(key[self.split], slice):
                # standard slicing along the split axis,
                #   adjust the slice start, stop, and step, then run it on the processes which have the requested data
                key = list(key)
                key_start = key[self.split].start if key[self.split].start is not None else 0
                key_stop = (
                    key[self.split].stop
                    if key[self.split].stop is not None
                    else self.gshape[self.split]
                )
                if key_stop < 0:
                    key_stop = self.gshape[self.split] + key[self.split].stop
                key_step = key[self.split].step
                og_key_start = key_start
                st_pr = torch.where(key_start < chunk_ends)[0]
                st_pr = st_pr[0] if len(st_pr) > 0 else self.comm.size
                sp_pr = torch.where(key_stop >= chunk_starts)[0]
                sp_pr = sp_pr[-1] if len(sp_pr) > 0 else 0
                actives = list(range(st_pr, sp_pr + 1))
                if rank in actives:
                    key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
                    key_stop = ends[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
                    if key_step is not None and rank > actives[0]:
                        offset = (chunk_ends[rank - 1] - og_key_start) % key_step
                        if key_step > 2 and offset > 0:
                            key_start += key_step - offset
                        elif key_step == 2 and offset > 0:
                            key_start += (chunk_ends[rank - 1] - og_key_start) % key_step
                    if isinstance(key_start, torch.Tensor):
                        key_start = key_start.item()
                    if isinstance(key_stop, torch.Tensor):
                        key_stop = key_stop.item()
                    key[self.split] = slice(key_start, key_stop, key_step)
                    arr = self.__array[tuple(key)]

            elif isinstance(key[self.split], int):
                # if there is an integer in the key along the split axis, adjust it and then get `arr`
                key = list(key)
                key[self.split] = (
                    key[self.split] + self.gshape[self.split]
                    if key[self.split] < 0
                    else key[self.split]
                )
                if key[self.split] in range(chunk_start, chunk_end):
                    key[self.split] = key[self.split] - chunk_start
                    arr = self.__array[tuple(key)]

            if 0 in arr.shape:
                # arr is empty
                # gout is all 0s as is the shape
                warnings.warn(
                    "This process (rank: {}) is without data after slicing, "
                    "running the .balance_() function is recommended".format(self.comm.rank),
                    ResourceWarning,
                )

            return DNDarray(
                arr.type(l_dtype),
                gout_full if isinstance(gout_full, tuple) else tuple(gout_full),
                self.dtype,
                new_split,
                self.device,
                self.comm,
            )

    if torch.cuda.device_count() > 0:

        def gpu(self) -> DNDarray:
            """
            Returns a copy of this object in GPU memory. If this object is already in GPU memory, then no copy is
            performed and the original object is returned.

            """
            self.__array = self.__array.cuda(devices.gpu.torch_device)
            self.__device = devices.gpu
            return self

    def __int__(self) -> int:
        """
        Integer scalar casting.
        """
        return self.__cast(int)

    def is_balanced(self) -> bool:
        """
        Determine if ``self`` is balanced evenly (or as evenly as possible) across all nodes
        """
        _, _, chk = self.comm.chunk(self.shape, self.split)
        test_lshape = tuple([x.stop - x.start for x in chk])
        balanced = 1 if test_lshape == self.lshape else 0

        out = self.comm.allreduce(balanced, MPI.SUM)
        return True if out == self.comm.size else False

    def is_distributed(self) -> bool:
        """
        Determines whether the data of this array is distributed across multiple processes.
        """
        return self.split is not None and self.comm.is_distributed()

    def item(self):
        """
        Returns the only element of a 1-element :class:`DNDarray`.
        Mirror of the pytorch command by the same name. If size of array is >1 element, then a ``ValueError`` is
        raised (by pytorch)

        Examples
        -------
        >>> import heat as ht
        >>> x = ht.zeros((1))
        >>> x.item()
        0.0
        """
        return self.__array.item()

    def __len__(self) -> int:
        """
        The length of the DNDarray, i.e. the number of items in the first dimension.
        """
        return self.shape[0]

    def numpy(self) -> np.array:
        """
        Convert :class:`DNDarray` to numpy array. If the tensor is distributed it will be merged beforehand. If the array
        resides on the GPU, it will be copied to the CPU first.

        Examples
        --------
        >>> import heat as ht
        T1 = ht.random.randn((10,8))
        T1.numpy()
        """
        dist = manipulations.resplit(self, axis=None)
        return dist._DNDarray__array.cpu().numpy()

    def qr(self, tiles_per_proc=1, calc_q=True, overwrite_a=False) -> Tuple[DNDarray, DNDarray]:
        """
        Calculates the QR decomposition of a 2D :class:`DNDarray`.
        Returns a Tuple of Q and R
        The algorithms are based on the CAQR and TSQR algorithms. For more information see the references.

        Parameters
        ----------
        a : DNDarray
            DNDarray which will be decomposed
        tiles_per_proc : int or torch.Tensor, optional
            Number of tiles per process to operate on
        calc_q : bool, optional
            whether or not to calculate Q.
            If ``True``, function returns ``(Q, R)``.
            If ``False``, function returns ``(None, R)``.
        overwrite_a : bool, optional
            If ``True``, function overwrites the DNDarray a, with R.
            If ``False``, a new array will be created for R

        References
        ----------
        [0] W. Zheng, F. Song, L. Lin, and Z. Chen, “Scaling Up Parallel Computation of Tiled QR
        Factorizations by a Distributed Scheduling Runtime System and Analytical Modeling,”
        Parallel Processing Letters, vol. 28, no. 01, p. 1850004, 2018.

        [1] Bilel Hadri, Hatem Ltaief, Emmanuel Agullo, Jack Dongarra. Tile QR Factorization with
        Parallel Panel Processing for Multicore Architectures. 24th IEEE International Parallel
        and DistributedProcessing Symposium (IPDPS 2010), Apr 2010, Atlanta, United States.
        inria-00548899

        [2] Gene H. Golub and Charles F. Van Loan. 1996. Matrix Computations (3rd Ed.).
        """
        return linalg.qr(
            self, tiles_per_proc=tiles_per_proc, calc_q=calc_q, overwrite_a=overwrite_a
        )

    def __repr__(self, *args):
        # TODO: document me
        # TODO: generate none-PyTorch repr
        return self.__array.__repr__(*args)

    def redistribute_(self, lshape_map=None, target_map=None):
        """
        Redistributes the data of the :class:`DNDarray` *along the split axis* to match the given target map.
        This function does not modify the non-split dimensions of the ``DNDarray``.
        This is an abstraction and extension of the balance function.

        Parameters
        ----------
        lshape_map : torch.Tensor, optional
            The current lshape of processes.
            Units are ``[rank, lshape]``.
        target_map : torch.Tensor, optional
            The desired distribution across the processes.
            Units are ``[rank, target lshape]``.
            Note: the only important parts of the target map are the values along the split axis,
            values which are not along this axis are there to mimic the shape of the ``lshape_map``.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int)
        >>> target_map[0, 2] = 67
        >>> print(target_map)
        [0/2] tensor([[ 0,  0, 67],
        [0/2]         [ 0,  0,  0],
        [0/2]         [ 0,  0,  0]], dtype=torch.int32)
        [1/2] tensor([[ 0,  0, 67],
        [1/2]         [ 0,  0,  0],
        [1/2]         [ 0,  0,  0]], dtype=torch.int32)
        [2/2] tensor([[ 0,  0, 67],
        [2/2]         [ 0,  0,  0],
        [2/2]         [ 0,  0,  0]], dtype=torch.int32)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> st.redistribute_(target_map=target_map)
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        """
        if not self.is_distributed():
            return
        snd_dtype = self.dtype.torch_type()
        # units -> {pr, 1st index, 2nd index}
        if lshape_map is None:
            # NOTE: giving an lshape map which is incorrect will result in an incorrect distribution
            lshape_map = self.create_lshape_map()
        else:
            if not isinstance(lshape_map, torch.Tensor):
                raise TypeError(
                    "lshape_map must be a torch.Tensor, currently {}".format(type(lshape_map))
                )
            if lshape_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    "lshape_map must have the shape ({}, {}), currently {}".format(
                        self.comm.size, len(self.gshape), lshape_map.shape
                    )
                )

        if target_map is None:  # if no target map is given then it will balance the tensor
            target_map = torch.zeros(
                (self.comm.size, len(self.gshape)), dtype=int, device=self.device.torch_device
            )
            _, _, chk = self.comm.chunk(self.shape, self.split)
            target_map = lshape_map.clone()
            target_map[..., self.split] = 0
            for pr in range(self.comm.size):
                target_map[pr, self.split] = self.comm.chunk(self.shape, self.split, rank=pr)[1][
                    self.split
                ]
        else:
            if not isinstance(target_map, torch.Tensor):
                raise TypeError(
                    "target_map must be a torch.Tensor, currently {}".format(type(target_map))
                )
            if target_map[..., self.split].sum() != self.shape[self.split]:
                raise ValueError(
                    "Sum along the split axis of the target map must be equal to the "
                    "shape in that dimension, currently {}".format(target_map[..., self.split])
                )
            if target_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    "target_map must have the shape {}, currently {}".format(
                        (self.comm.size, len(self.gshape)), target_map.shape
                    )
                )

        lshape_cumsum = torch.cumsum(lshape_map[..., self.split], dim=0)
        chunk_cumsum = torch.cat(
            (
                torch.tensor([0], device=self.device.torch_device),
                torch.cumsum(target_map[..., self.split], dim=0),
            ),
            dim=0,
        )
        # need the data start as well for process 0
        for rcv_pr in range(self.comm.size - 1):
            st = chunk_cumsum[rcv_pr].item()
            sp = chunk_cumsum[rcv_pr + 1].item()
            # start pr should be the next process with data
            if lshape_map[rcv_pr, self.split] >= target_map[rcv_pr, self.split]:
                # if there is more data on the process than the start process than start == stop
                st_pr = rcv_pr
                sp_pr = rcv_pr
            else:
                # if there is less data on the process than need to get the data from the next data
                # with data
                # need processes > rcv_pr with lshape > 0
                st_pr = (
                    torch.nonzero(input=lshape_map[rcv_pr:, self.split] > 0, as_tuple=False)[
                        0
                    ].item()
                    + rcv_pr
                )
                hld = (
                    torch.nonzero(input=sp <= lshape_cumsum[rcv_pr:], as_tuple=False).flatten()
                    + rcv_pr
                )
                sp_pr = hld[0].item() if hld.numel() > 0 else self.comm.size

            # st_pr and sp_pr are the processes on which the data sits at the beginning
            # need to loop from st_pr to sp_pr + 1 and send the pr
            for snd_pr in range(st_pr, sp_pr + 1):
                if snd_pr == self.comm.size:
                    break
                data_required = abs(sp - st - lshape_map[rcv_pr, self.split].item())
                send_amt = (
                    data_required
                    if data_required <= lshape_map[snd_pr, self.split]
                    else lshape_map[snd_pr, self.split]
                )
                if (sp - st) <= lshape_map[rcv_pr, self.split].item() or snd_pr == rcv_pr:
                    send_amt = 0
                # send amount is the data still needed by recv if that is available on the snd
                if send_amt != 0:
                    self.__redistribute_shuffle(
                        snd_pr=snd_pr, send_amt=send_amt, rcv_pr=rcv_pr, snd_dtype=snd_dtype
                    )
                lshape_cumsum[snd_pr] -= send_amt
                lshape_cumsum[rcv_pr] += send_amt
                lshape_map[rcv_pr, self.split] += send_amt
                lshape_map[snd_pr, self.split] -= send_amt
            if lshape_map[rcv_pr, self.split] > target_map[rcv_pr, self.split]:
                # if there is any data left on the process then send it to the next one
                send_amt = lshape_map[rcv_pr, self.split] - target_map[rcv_pr, self.split]
                self.__redistribute_shuffle(
                    snd_pr=rcv_pr, send_amt=send_amt.item(), rcv_pr=rcv_pr + 1, snd_dtype=snd_dtype
                )
                lshape_cumsum[rcv_pr] -= send_amt
                lshape_cumsum[rcv_pr + 1] += send_amt
                lshape_map[rcv_pr, self.split] -= send_amt
                lshape_map[rcv_pr + 1, self.split] += send_amt

        if any(lshape_map[..., self.split] != target_map[..., self.split]):
            # sometimes need to call the redistribute once more,
            # (in the case that the second to last processes needs to get data from +1 and -1)
            self.redistribute_(lshape_map=lshape_map, target_map=target_map)

    def __redistribute_shuffle(self, snd_pr, send_amt, rcv_pr, snd_dtype):
        """
        Function to abstract the function used during redistribute for shuffling data between
        processes along the split axis

        Parameters
        ----------
        snd_pr : int or torch.Tensor
            Sending process
        send_amt : int or torch.Tensor
            Amount of data to be sent by the sending process
        rcv_pr : int or torch.Tensor
            Recieving process
        snd_dtype : torch.dtype
            Torch type of the data in question
        """
        rank = self.comm.rank
        send_slice = [slice(None)] * self.ndim
        keep_slice = [slice(None)] * self.ndim
        if rank == snd_pr:
            if snd_pr < rcv_pr:  # data passed to a higher rank (off the bottom)
                send_slice[self.split] = slice(
                    self.lshape[self.split] - send_amt, self.lshape[self.split]
                )
                keep_slice[self.split] = slice(0, self.lshape[self.split] - send_amt)
            if snd_pr > rcv_pr:  # data passed to a lower rank (off the top)
                send_slice[self.split] = slice(0, send_amt)
                keep_slice[self.split] = slice(send_amt, self.lshape[self.split])
            data = self.__array[send_slice].clone()
            self.comm.Send(data, dest=rcv_pr, tag=685)
            self.__array = self.__array[keep_slice]
        if rank == rcv_pr:
            shp = list(self.gshape)
            shp[self.split] = send_amt
            data = torch.zeros(shp, dtype=snd_dtype, device=self.device.torch_device)
            self.comm.Recv(data, source=snd_pr, tag=685)
            if snd_pr < rcv_pr:  # data passed from a lower rank (append to top)
                self.__array = torch.cat((data, self.__array), dim=self.split)
            if snd_pr > rcv_pr:  # data passed from a higher rank (append to bottom)
                self.__array = torch.cat((self.__array, data), dim=self.split)

    def resplit_(self, axis=None):
        """
        In-place option for resplitting a :class:`DNDarray`.

        Parameters
        ----------
        axis : int
            The new split axis, ``None`` denotes gathering, an int will set the new split axis

        Examples
        --------
        >>> a = ht.zeros((4, 5,), split=0)
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, None)
        >>> a.split
        None
        >>> a.lshape
        (0/2) (4, 5)
        (1/2) (4, 5)
        >>> a = ht.zeros((4, 5,), split=0)
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, 1)
        >>> a.split
        1
        >>> a.lshape
        (0/2) (4, 3)
        (1/2) (4, 2)
        """
        # sanitize the axis to check whether it is in range
        axis = sanitize_axis(self.shape, axis)

        # early out for unchanged content
        if axis == self.split:
            return self
        if axis is None:
            gathered = torch.empty(
                self.shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )
            counts, displs, _ = self.comm.counts_displs_shape(self.shape, self.split)
            self.comm.Allgatherv(self.__array, (gathered, counts, displs), recv_axis=self.split)
            self.__array = gathered
            self.__split = axis
            return self
        # tensor needs be split/sliced locally
        if self.split is None:
            # new_arr = self
            _, _, slices = self.comm.chunk(self.shape, axis)
            temp = self.__array[slices]
            self.__array = torch.empty((1,), device=self.device.torch_device)
            # necessary to clear storage of local __array
            self.__array = temp.clone().detach()
            self.__split = axis
            return self

        tiles = tiling.SplitTiles(self)
        new_tile_locs = tiles.set_tile_locations(
            split=axis, tile_dims=tiles.tile_dimensions, arr=self
        )
        rank = self.comm.rank
        # receive the data with non-blocking, save which process its from
        rcv = {}
        for rpr in range(self.comm.size):
            # need to get where the tiles are on the new one first
            # rpr is the destination
            new_locs = torch.where(new_tile_locs == rpr)
            new_locs = torch.stack([new_locs[i] for i in range(self.ndim)], dim=1)
            for i in range(new_locs.shape[0]):
                key = tuple(new_locs[i].tolist())
                spr = tiles.tile_locations[key].item()
                to_send = tiles[key]
                if spr == rank and spr != rpr:
                    self.comm.Send(to_send.clone(), dest=rpr, tag=rank)
                    del to_send
                elif spr == rpr and rpr == rank:
                    rcv[key] = [None, to_send]
                elif rank == rpr:
                    sz = tiles.get_tile_size(key)
                    buf = torch.zeros(
                        sz, dtype=self.dtype.torch_type(), device=self.device.torch_device
                    )
                    w = self.comm.Irecv(buf=buf, source=spr, tag=spr)
                    rcv[key] = [w, buf]
        dims = list(range(self.ndim))
        del dims[axis]
        sorted_keys = sorted(rcv.keys())
        # todo: reduce the problem to 1D cats for each dimension, then work up
        sz = self.comm.size
        arrays = []
        for prs in range(int(len(sorted_keys) / sz)):
            lp_keys = sorted_keys[prs * sz : (prs + 1) * sz]
            lp_arr = None
            for k in lp_keys:
                if rcv[k][0] is not None:
                    rcv[k][0].wait()
                if lp_arr is None:
                    lp_arr = rcv[k][1]
                else:
                    lp_arr = torch.cat((lp_arr, rcv[k][1]), dim=dims[-1])
                del rcv[k]
            if lp_arr is not None:
                arrays.append(lp_arr)
        del dims[-1]
        # for 4 prs and 4 dims, arrays is now 16 elements long,
        # next need to group the each 4 (sz) and cat in the next dim
        for d in reversed(dims):
            new_arrays = []
            for prs in range(int(len(arrays) / sz)):
                new_arrays.append(torch.cat(arrays[prs * sz : (prs + 1) * sz], dim=d))
            arrays = new_arrays
            del d
        if len(arrays) == 1:
            arrays = arrays[0]

        self.__array = arrays
        self.__split = axis
        return self

    def save(self, path, *args, **kwargs):
        """
        Save the array's data to disk. Attempts to auto-detect the file format by determining the extension.

        Parameters
        ----------
        self : DNDarray
            The array holding the data to be stored
        path : str
            Path to the file to be stored.
        args/kwargs : list/dict
            Additional options passed to the particular functions.

        Raises
        -------
        ValueError
            If the file extension is not understood or known.

        Examples
        --------
        >>> a = ht.arange(100, split=0)
        >>> a.save('data.h5', 'DATA', mode='a')
        >>> a.save('data.nc', 'DATA', mode='w')
        """
        return io.save(self, path, *args, **kwargs)

    if io.supports_hdf5():

        def save_hdf5(self, path, dataset, mode="w", **kwargs):
            """
            Saves data to an HDF5 file. Attempts to utilize parallel I/O if possible.

            Parameters
            ----------
            path : str
                Path to the HDF5 file to be written.
            dataset : str
                Name of the dataset the data is saved to.
            mode : str,
                File access mode, one of ``'w', 'a', 'r+'``
            kwargs : dict
                Additional arguments passed to the created dataset.

            Raises
            -------
            TypeError
                If any of the input parameters are not of correct type.
            ValueError
                If the access mode is not understood.

            Examples
            --------
            >>> ht.arange(100, split=0).save_hdf5('data.h5', dataset='DATA')
            """
            return io.save_hdf5(self, path, dataset, mode, **kwargs)

    if io.supports_netcdf():

        def save_netcdf(self, path, variable, mode="w", **kwargs):
            """
            Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

            Parameters
            ----------
            path : str
                Path to the netCDF4 file to be written.
            variable : str
                Name of the variable the data is saved to.
            mode : str
                File access mode, one of ``'w', 'a', 'r+'``
            kwargs : dict
                Additional arguments passed to the created dataset.

            Raises
            -------
            TypeError
                If any of the input parameters are not of correct type.
            ValueError
                If the access mode is not understood.

            Examples
            --------
            >>> ht.arange(100, split=0).save_netcdf('data.nc', dataset='DATA')
            """
            return io.save_netcdf(self, path, variable, mode, **kwargs)

    def __setitem__(self, key, value):
        """
        Global item setter

        Parameters
        ----------
        key : int or Tuple or List or Slice
            Index/indices to be set
        value: np.scalar or tensor or torch.Tensor
            value to be set to the specified positions in the DNDarray (self)
        Notes
        -----
        If a DNDarray is given as the value to be set then the split axes are assumed to be equal.
        If they are not, PyTorch will raise an error when the values are attempted to be set on the local array

        Examples
        --------
        >>> a = ht.zeros((4,5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1] = 1
        >>> a
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 1., 0., 0., 0.],
                          [0., 1., 0., 0., 0.]])
        """
        if isinstance(key, DNDarray) and key.ndim == self.ndim:
            # this splits the key into torch.Tensors in each dimension for advanced indexing
            lkey = [slice(None, None, None)] * self.ndim
            for i in range(key.ndim):
                lkey[i] = key._DNDarray__array[..., i]
            key = tuple(lkey)
        elif not isinstance(key, tuple):
            h = [slice(None, None, None)] * self.ndim
            h[0] = key
            key = tuple(h)

        if not self.is_distributed():
            self.__setter(key, value)
        else:
            # raise RuntimeError("split axis of array and the target value are not equal") removed
            # this will occur if the local shapes do not match
            rank = self.comm.rank
            ends = []
            for pr in range(self.comm.size):
                _, _, e = self.comm.chunk(self.shape, self.split, rank=pr)
                ends.append(e[self.split].stop - e[self.split].start)
            ends = torch.tensor(ends, device=self.device.torch_device)
            chunk_ends = ends.cumsum(dim=0)
            chunk_starts = torch.tensor([0] + chunk_ends.tolist(), device=self.device.torch_device)
            _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
            chunk_start = chunk_slice[self.split].start
            chunk_end = chunk_slice[self.split].stop

            if isinstance(key, tuple):
                if isinstance(key[self.split], slice):
                    key = list(key)
                    key_start = key[self.split].start if key[self.split].start is not None else 0
                    key_stop = (
                        key[self.split].stop
                        if key[self.split].stop is not None
                        else self.gshape[self.split]
                    )
                    if key_stop < 0:
                        key_stop = self.gshape[self.split] + key[self.split].stop
                    key_step = key[self.split].step
                    og_key_start = key_start
                    st_pr = torch.where(key_start < chunk_ends)[0]
                    st_pr = st_pr[0] if len(st_pr) > 0 else self.comm.size
                    sp_pr = torch.where(key_stop >= chunk_starts)[0]
                    sp_pr = sp_pr[-1] if len(sp_pr) > 0 else 0
                    actives = list(range(st_pr, sp_pr + 1))
                    if rank in actives:
                        key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
                        key_stop = (
                            ends[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
                        )
                        if key_step is not None and rank > actives[0]:
                            offset = (chunk_ends[rank - 1] - og_key_start) % key_step
                            if key_step > 2 and offset > 0:
                                key_start += key_step - offset
                            elif key_step == 2 and offset > 0:
                                key_start += (chunk_ends[rank - 1] - og_key_start) % key_step
                        if isinstance(key_start, torch.Tensor):
                            key_start = key_start.item()
                        if isinstance(key_stop, torch.Tensor):
                            key_stop = key_stop.item()
                        key[self.split] = slice(key_start, key_stop, key_step)
                        # todo: need to slice the values to be the right size...
                        if isinstance(value, (torch.Tensor, type(self))):
                            value_slice = [slice(None, None, None)] * value.ndim
                            step2 = key_step if key_step is not None else 1
                            key_start = chunk_starts[rank] - og_key_start
                            key_stop = key_start + key_stop
                            slice_loc = (
                                value.ndim - 1 if self.split > value.ndim - 1 else self.split
                            )
                            value_slice[slice_loc] = slice(
                                key_start.item(), math.ceil(torch.true_divide(key_stop, step2)), 1
                            )
                            self.__setter(tuple(key), value[tuple(value_slice)])
                        else:
                            self.__setter(tuple(key), value)

                elif isinstance(key[self.split], torch.Tensor):
                    key = list(key)
                    key[self.split] -= chunk_start
                    self.__setter(tuple(key), value)

                elif key[self.split] in range(chunk_start, chunk_end):
                    key = list(key)
                    key[self.split] = key[self.split] - chunk_start
                    self.__setter(tuple(key), value)

                elif key[self.split] < 0:
                    key = list(key)
                    if self.gshape[self.split] + key[self.split] in range(chunk_start, chunk_end):
                        key[self.split] = key[self.split] + self.shape[self.split] - chunk_start
                        self.__setter(tuple(key), value)
            else:
                self.__setter(key, value)

    def __setter(self, key, value):
        if np.isscalar(value):
            self.__array.__setitem__(key, value)
        elif isinstance(value, DNDarray):
            self.__array.__setitem__(key, value.__array)
        elif isinstance(value, torch.Tensor):
            self.__array.__setitem__(key, value.data)
        elif isinstance(value, (list, tuple)):
            value = torch.tensor(value, device=self.device.torch_device)
            self.__array.__setitem__(key, value.data)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            self.__array.__setitem__(key, value.data)
        else:
            raise NotImplementedError("Not implemented for {}".format(value.__class__.__name__))

    def __str__(self, *args):
        """
        String representation of the array
        """
        # TODO: generate none-PyTorch str
        return self.__array.__str__(*args)

    def tolist(self, keepsplit=False) -> List:
        """
        Return a copy of the local array data as a (nested) Python list. For scalars, a standard Python number is returned.

        Parameters
        ----------
        keepsplit: bool
            Whether the list should be returned locally or globally.

        Examples
        --------
        >>> a = ht.array([[0,1],[2,3]])
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0,1],[2,3]], split=0)
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0,1],[2,3]], split=1)
        >>> a.tolist(keepsplit=True)
        (1/2) [[0], [2]]
        (2/2) [[1], [3]]
        """

        if not keepsplit:
            return manipulations.resplit(self, axis=None).__array.tolist()

        return self.__array.tolist()
