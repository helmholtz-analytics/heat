"""Provides HeAT's core data structure, the DNDarray, a distributed n-dimensional array"""
from __future__ import annotations

import math
import numpy as np
import torch
import warnings

from inspect import stack
from mpi4py import MPI
from pathlib import Path
from typing import List, Union, Tuple, TypeVar, Optional

warnings.simplefilter("always", ResourceWarning)

# NOTE: heat module imports need to be placed at the very end of the file to avoid cyclic dependencies
__all__ = ["DNDarray"]

Communication = TypeVar("Communication")


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
    gshape : Tuple[int,...]
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        The axis on which the array is divided between processes
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.
        If this information is not available (``self.balanced is None``), it
        can be gathered via the :func:`is_balanced()` method (requires communication).
    """

    def __init__(
        self,
        array: torch.Tensor,
        gshape: Tuple[int, ...],
        dtype: datatype,
        split: Union[int, None],
        device: Device,
        comm: Communication,
        balanced: bool,
    ):
        self.__array = array
        self.__gshape = gshape
        self.__dtype = dtype
        self.__split = split
        self.__device = device
        self.__comm = comm
        self.__balanced = balanced
        self.__ishalo = False
        self.__halo_next = None
        self.__halo_prev = None
        self.__partitions_dict__ = None
        self.__lshape_map = None

        # check for inconsistencies between torch and heat devices
        assert str(array.device) == device.torch_device

    @property
    def balanced(self) -> bool:
        """
        Boolean value indicating if the DNDarray is balanced between the MPI processes
        """
        return self.__balanced

    @property
    def comm(self) -> Communication:
        """
        The :class:`~heat.core.communication.Communication` of the ``DNDarray``
        """
        return self.__comm

    @property
    def device(self) -> Device:
        """
        The :class:`~heat.core.devices.Device` of the ``DNDarray``
        """
        return self.__device

    @property
    def dtype(self) -> datatype:
        """
        The :class:`~heat.core.types.datatype` of the ``DNDarray``
        """
        return self.__dtype

    @property
    def gshape(self) -> Tuple:
        """
        Returns the global shape of the ``DNDarray`` across all processes
        """
        return self.__gshape

    @property
    def halo_next(self) -> torch.Tensor:
        """
        Returns the halo of the next process
        """
        return self.__halo_next

    @property
    def halo_prev(self) -> torch.Tensor:
        """
        Returns the halo of the previous process
        """
        return self.__halo_prev

    @property
    def larray(self) -> torch.Tensor:
        """
        Returns the underlying process-local ``torch.Tensor`` of the ``DNDarray``
        """
        return self.__array

    @larray.setter
    def larray(self, array: torch.Tensor):
        """
        Setter for ``self.larray``, the underlying local ``torch.Tensor`` of the ``DNDarray``.

        Parameters
        ----------
        array : torch.Tensor
            The new underlying local ``torch.tensor`` of the ``DNDarray``

        Warning
        -----------
        Please use this function with care, as it might corrupt/invalidate the metadata in the ``DNDarray`` instance.
        """
        # sanitize tensor input
        sanitation.sanitize_in_tensor(array)
        # verify consistency of tensor shape with global DNDarray
        sanitation.sanitize_lshape(self, array)
        # set balanced status
        split = self.split
        if split is not None and array.shape[split] != self.lshape[split]:
            self.__balanced = None
        self.__array = array

    @property
    def nbytes(self) -> int:
        """
        Returns the number of bytes consumed by the global tensor. Equivalent to property gnbytes.

        Note
        ------------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.
        """
        return self.__array.element_size() * self.size

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the ``DNDarray``
        """
        return len(self.__gshape)

    @property
    def __partitioned__(self) -> dict:
        """
        This will return a dictionary containing information useful for working with the partitioned
        data. These items include the shape of the data on each process, the starting index of the data
        that a process has, the datatype of the data, the local devices, as well as the global
        partitioning scheme.

        An example of the output and shape is shown in :func:`ht.core.DNDarray.create_partition_interface <ht.core.DNDarray.create_partition_interface>`.

        Returns
        -------
        dictionary with the partition interface
        """
        if self.__partitions_dict__ is None:
            self.__partitions_dict__ = self.create_partition_interface()
        return self.__partitions_dict__

    @property
    def size(self) -> int:
        """
        Number of total elements of the ``DNDarray``
        """
        return (
            torch.prod(
                torch.tensor(self.gshape, dtype=torch.float64, device=self.device.torch_device)
            )
            .long()
            .item()
        )

    @property
    def gnbytes(self) -> int:
        """
        Returns the number of bytes consumed by the global ``DNDarray``

        Note
        -----------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.
        """
        return self.nbytes

    @property
    def gnumel(self) -> int:
        """
        Returns the number of total elements of the ``DNDarray``
        """
        return self.size

    @property
    def imag(self) -> DNDarray:
        """
        Return the imaginary part of the ``DNDarray``.
        """
        return complex_math.imag(self)

    @property
    def lnbytes(self) -> int:
        """
        Returns the number of bytes consumed by the local ``torch.Tensor``

        Note
        -------------------
            Does not include memory consumed by non-element attributes of the ``DNDarray`` object.
        """
        return self.__array.element_size() * self.__array.nelement()

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
        DNDarray([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]], dtype=ht.float32, device=cpu:0, split=0)
        >>> a.lloc[1, 0:4]
        (1/2) tensor([0., 0., 0., 0.])
        (2/2) tensor([0., 0., 0., 0.])
        >>> a.lloc[1, 0:4] = torch.arange(1, 5)
        >>> a
        DNDarray([[0., 0., 0., 0., 0.],
                  [1., 2., 3., 4., 0.],
                  [0., 0., 0., 0., 0.],
                  [1., 2., 3., 4., 0.]], dtype=ht.float32, device=cpu:0, split=0)
        """
        return LocalIndex(self.__array)

    @property
    def lshape(self) -> Tuple[int]:
        """
        Returns the shape of the ``DNDarray`` on each node
        """
        return tuple(self.__array.shape)

    @property
    def lshape_map(self) -> torch.Tensor:
        """
        Returns the lshape map. If it hasn't been previously created then it will be created here.
        """
        return self.create_lshape_map()

    @property
    def real(self) -> DNDarray:
        """
        Return the real part of the ``DNDarray``.
        """
        return complex_math.real(self)

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
        steps = list(self.larray.stride())
        itemsize = self.larray.storage().element_size()
        strides = tuple(step * itemsize for step in steps)
        return strides

    @property
    def T(self):
        """
        Reverse the dimensions of a DNDarray.
        """
        # specialty docs for this version of transpose. The transpose function is in heat/core/linalg/basics
        return linalg.transpose(self, axes=None)

    @property
    def array_with_halos(self) -> torch.Tensor:
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

    def get_halo(self, halo_size: int) -> torch.Tensor:
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next/self.halo_prev``.

        Parameters
        ----------
        halo_size : int
            Size of the halo.
        """
        if not isinstance(halo_size, int):
            raise TypeError(
                f"halo_size needs to be of Python type integer, {type(halo_size)} given"
            )
        if halo_size < 0:
            raise ValueError(
                f"halo_size needs to be a positive Python integer, {type(halo_size)} given"
            )

        if self.is_distributed() and halo_size > 0:
            # gather lshapes
            lshape_map = self.lshape_map
            rank = self.comm.rank

            populated_ranks = torch.nonzero(lshape_map[:, self.split]).squeeze().tolist()
            if rank in populated_ranks:
                first_rank = populated_ranks[0]
                last_rank = populated_ranks[-1]
                if rank != last_rank:
                    next_rank = populated_ranks[populated_ranks.index(rank) + 1]
                if rank != first_rank:
                    prev_rank = populated_ranks[populated_ranks.index(rank) - 1]
            else:
                # if process has no data we ignore it
                return

            if (halo_size > self.lshape_map[:, self.split][populated_ranks]).any():
                # halo_size is larger than the local size on at least one process
                raise ValueError(
                    f"halo_size {halo_size} needs to be smaller than chunk-size {self.lshape[self.split]} )"
                )

            a_prev = self.__prephalo(0, halo_size)
            a_next = self.__prephalo(-halo_size, None)
            res_prev = None
            res_next = None

            req_list = []

            # exchange data with next populated process
            if rank != last_rank:
                self.comm.Isend(a_next, next_rank)
                res_prev = torch.zeros(
                    a_prev.size(), dtype=a_prev.dtype, device=self.device.torch_device
                )
                req_list.append(self.comm.Irecv(res_prev, source=next_rank))

            if rank != first_rank:
                self.comm.Isend(a_prev, prev_rank)
                res_next = torch.zeros(
                    a_next.size(), dtype=a_next.dtype, device=self.device.torch_device
                )
                req_list.append(self.comm.Irecv(res_next, source=prev_rank))

            for req in req_list:
                req.Wait()

            self.__halo_next = res_prev
            self.__halo_prev = res_next
            self.__ishalo = True

    def __cat_halo(self) -> torch.Tensor:
        """
        Return local array concatenated to halos if they are available.
        """
        if not self.is_distributed():
            return self.__array
        return torch.cat(
            [_ for _ in (self.__halo_prev, self.__array, self.__halo_next) if _ is not None],
            dim=self.split,
        )

    def __array__(self) -> np.ndarray:
        """
        Returns a view of the process-local slice of the :class:`DNDarray` as a numpy ndarray, if the ``DNDarray`` resides on CPU. Otherwise, it returns a copy, on CPU, of the process-local slice of ``DNDarray`` as numpy ndarray.
        """
        return self.larray.cpu().__array__()

    def astype(self, dtype, copy=True) -> DNDarray:
        """
        Returns a casted version of this array.
        Casted array is a new array of the same shape but with given type of this array. If copy is ``True``, the
        same array is returned instead.

        Parameters
        ----------
        dtype : datatype
            Heat type to which the array is cast
        copy : bool, optional
            By default the operation returns a copy of this array. If copy is set to ``False`` the cast is performed
            in-place and this array is returned

        """
        dtype = canonical_heat_type(dtype)
        casted_array = self.__array.type(dtype.torch_type())
        if copy:
            return DNDarray(
                casted_array, self.shape, dtype, self.split, self.device, self.comm, self.balanced
            )

        self.__array = casted_array
        self.__dtype = dtype

        return self

    def balance_(self) -> DNDarray:
        """
        Function for balancing a :class:`DNDarray` between all nodes. To determine if this is needed use the :func:`is_balanced()` function.
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
        if self.is_balanced(force_check=True):
            return
        self.redistribute_()

    def __bool__(self) -> bool:
        """
        Boolean scalar casting.
        """
        return self.__cast(bool)

    def __cast(self, cast_function) -> Union[float, int]:
        """
        Implements a generic cast function for ``DNDarray`` objects.

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

    def __complex__(self) -> DNDarray:
        """
        Complex scalar casting.
        """
        return self.__cast(complex)

    def counts_displs(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Returns actual counts (number of items per process) and displacements (offsets) of the DNDarray.
        Does not assume load balance.
        """
        if self.split is not None:
            counts = self.lshape_map[:, self.split]
            displs = [0] + torch.cumsum(counts, dim=0)[:-1].tolist()
            return tuple(counts.tolist()), tuple(displs)
        else:
            raise ValueError("Non-distributed DNDarray. Cannot calculate counts and displacements.")

    def cpu(self) -> DNDarray:
        """
        Returns a copy of this object in main memory. If this object is already in main memory, then no copy is
        performed and the original object is returned.
        """
        self.__array = self.__array.cpu()
        self.__device = devices.cpu
        return self

    def create_lshape_map(self, force_check: bool = False) -> torch.Tensor:
        """
        Generate a 'map' of the lshapes of the data on all processes.
        Units are ``(process rank, lshape)``

        Parameters
        ----------
        force_check : bool, optional
            if False (default) and the lshape map has already been created, use the previous
            result. Otherwise, create the lshape_map
        """
        if not force_check and self.__lshape_map is not None:
            return self.__lshape_map.clone()

        lshape_map = torch.zeros(
            (self.comm.size, self.ndim), dtype=torch.int64, device=self.device.torch_device
        )
        if not self.is_distributed:
            lshape_map[:] = torch.tensor(self.gshape, device=self.device.torch_device)
            return lshape_map
        if self.is_balanced(force_check=True):
            for i in range(self.comm.size):
                _, lshape, _ = self.comm.chunk(self.gshape, self.split, rank=i)
                lshape_map[i, :] = torch.tensor(lshape, device=self.device.torch_device)
        else:
            lshape_map[self.comm.rank, :] = torch.tensor(
                self.lshape, device=self.device.torch_device
            )
            self.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        self.__lshape_map = lshape_map
        return lshape_map.clone()

    def create_partition_interface(self):
        """
        Create a partition interface in line with the DPPY proposal. This is subject to change.
        The intention of this to facilitate the usage of a general format for the referencing of
        distributed datasets.

        An example of the output and shape is shown below.

        __partitioned__ = {
            'shape': (27, 3, 2),
            'partition_tiling': (4, 1, 1),
            'partitions': {
                (0, 0, 0): {
                    'start': (0, 0, 0),
                    'shape': (7, 3, 2),
                    'data': tensor([...], dtype=torch.int32),
                    'location': [0],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (1, 0, 0): {
                    'start': (7, 0, 0),
                    'shape': (7, 3, 2),
                    'data': None,
                    'location': [1],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (2, 0, 0): {
                    'start': (14,  0,  0),
                    'shape': (7, 3, 2),
                    'data': None,
                    'location': [2],
                    'dtype': torch.int32,
                    'device': 'cpu'
                },
                (3, 0, 0): {
                    'start': (21,  0,  0),
                    'shape': (6, 3, 2),
                    'data': None,
                    'location': [3],
                    'dtype': torch.int32,
                    'device': 'cpu'
                }
            },
            'locals': [(rank, 0, 0)],
            'get': lambda x: x,
        }

        Returns
        -------
        dictionary containing the partition interface as shown above.
        """
        lshape_map = self.create_lshape_map()
        start_idx_map = torch.zeros_like(lshape_map)

        part_tiling = [1] * self.ndim
        lcls = [0] * self.ndim

        z = torch.tensor([0], device=self.device.torch_device, dtype=self.dtype.torch_type())
        if self.split is not None:
            starts = torch.cat((z, torch.cumsum(lshape_map[:, self.split], dim=0)[:-1]), dim=0)
            lcls[self.split] = self.comm.rank
            part_tiling[self.split] = self.comm.size
        else:
            starts = torch.zeros(self.ndim, dtype=torch.int, device=self.device.torch_device)

        start_idx_map[:, self.split] = starts

        partitions = {}
        base_key = [0] * self.ndim
        for r in range(self.comm.size):
            if self.split is not None:
                base_key[self.split] = r
                dat = None if r != self.comm.rank else self.larray
            else:
                dat = self.larray
            partitions[tuple(base_key)] = {
                "start": tuple(start_idx_map[r].tolist()),
                "shape": tuple(lshape_map[r].tolist()),
                "data": dat,
                "location": [r],
                "dtype": self.dtype.torch_type(),
                "device": self.device.torch_device,
            }

        partition_dict = {
            "shape": self.gshape,
            "partition_tiling": tuple(part_tiling),
            "partitions": partitions,
            "locals": [tuple(lcls)],
            "get": lambda x: x,
        }

        self.__partitions_dict__ = partition_dict

        return partition_dict

    def __float__(self) -> DNDarray:
        """
        Float scalar casting.

        See Also
        ---------
        :func:`~heat.core.manipulations.flatten`
        """
        return self.__cast(float)

    def fill_diagonal(self, value: float) -> DNDarray:
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
                    self.larray[:, indices[0] : indices[1]] = self.larray[
                        :, indices[0] : indices[1]
                    ].fill_diagonal_(value)
                elif self.split == 1:
                    self.larray[indices[0] : indices[1], :] = self.larray[
                        indices[0] : indices[1], :
                    ].fill_diagonal_(value)

        else:
            self.larray = self.larray.fill_diagonal_(value)

        return self

    def __getitem__(self, key: Union[int, Tuple[int, ...], List[int, ...]]) -> DNDarray:
        """
        Global getter function for DNDarrays.
        Returns a new DNDarray composed of the elements of the original tensor selected by the indices
        given. This does *NOT* redistribute or rebalance the resulting tensor. If the selection of values is
        unbalanced then the resultant tensor is also unbalanced!
        To redistributed the ``DNDarray`` use :func:`balance()` (issue #187)

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
        key = getattr(key, "copy()", key)
        l_dtype = self.dtype.torch_type()
        advanced_ind = False
        if isinstance(key, DNDarray) and key.ndim == self.ndim:
            """if the key is a DNDarray and it has as many dimensions as self, then each of the
            entries in the 0th dim refer to a single element. To handle this, the key is split
            into the torch tensors for each dimension. This signals that advanced indexing is
            to be used."""
            # NOTE: this gathers the entire key on every process!!
            # TODO: remove this resplit!!
            key = manipulations.resplit(key)
            if key.larray.dtype in [torch.bool, torch.uint8]:
                key = indexing.nonzero(key)

            if key.ndim > 1:
                key = list(key.larray.split(1, dim=1))
                # key is now a list of tensors with dimensions (key.ndim, 1)
                # squeeze singleton dimension:
                key = [key[i].squeeze_(1) for i in range(len(key))]
            else:
                key = [key]
            advanced_ind = True
        elif not isinstance(key, tuple):
            """this loop handles all other cases. DNDarrays which make it to here refer to
            advanced indexing slices, as do the torch tensors. Both DNDaarrys and torch.Tensors
            are cast into lists here by PyTorch. lists mean advanced indexing will be used"""
            h = [slice(None, None, None)] * max(self.ndim, 1)
            if isinstance(key, DNDarray):
                key = manipulations.resplit(key)
                if key.larray.dtype in [torch.bool, torch.uint8]:
                    h[0] = torch.nonzero(key.larray).flatten()  # .tolist()
                else:
                    h[0] = key.larray.tolist()
            elif isinstance(key, torch.Tensor):
                if key.dtype in [torch.bool, torch.uint8]:
                    # (coquelin77) i am not certain why this works without being a list. but it works...for now
                    h[0] = torch.nonzero(key).flatten()  # .tolist()
                else:
                    h[0] = key.tolist()
            else:
                h[0] = key

            key = list(h)

        if isinstance(key, (list, tuple)):
            key = list(key)
            for i, k in enumerate(key):
                # this might be a good place to check if the dtype is there
                try:
                    k = manipulations.resplit(k)
                    key[i] = k.larray
                except AttributeError:
                    pass

        # ellipsis
        key = list(key)
        key_classes = [type(n) for n in key]
        # if any(isinstance(n, ellipsis) for n in key):
        n_elips = key_classes.count(type(...))
        if n_elips > 1:
            raise ValueError("key can only contain 1 ellipsis")
        elif n_elips == 1:
            # get which item is the ellipsis
            ell_ind = key_classes.index(type(...))
            kst = key[:ell_ind]
            kend = key[ell_ind + 1 :]
            slices = [slice(None)] * (self.ndim - (len(kst) + len(kend)))
            key = kst + slices + kend
        else:
            key = key + [slice(None)] * (self.ndim - len(key))

        self_proxy = self.__torch_proxy__()
        for i in range(len(key)):
            if self.__key_adds_dimension(key, i, self_proxy):
                key[i] = slice(None)
                return self.expand_dims(i)[tuple(key)]

        key = tuple(key)
        # assess final global shape
        gout_full = list(self_proxy[key].shape)

        # calculate new split axis
        new_split = self.split
        # when slicing, squeezed singleton dimensions may affect new split axis
        if self.split is not None and len(gout_full) < self.ndim:
            if advanced_ind:
                new_split = 0
            else:
                for i in range(len(key[: self.split + 1])):
                    if self.__key_is_singular(key, i, self_proxy):
                        new_split = None if i == self.split else new_split - 1

        key = tuple(key)
        if not self.is_distributed():
            arr = self.__array[key].reshape(gout_full)
            return DNDarray(
                arr, tuple(gout_full), self.dtype, new_split, self.device, self.comm, self.balanced
            )

        # else: (DNDarray is distributed)
        arr = torch.tensor([], dtype=self.__array.dtype, device=self.__array.device)
        rank = self.comm.rank
        counts, chunk_starts = self.counts_displs()
        counts, chunk_starts = torch.tensor(counts), torch.tensor(chunk_starts)
        chunk_ends = chunk_starts + counts
        chunk_start = chunk_starts[rank]
        chunk_end = chunk_ends[rank]

        if len(key) == 0:  # handle empty list
            # this will return an array of shape (0, ...)
            arr = self.__array[key]

        """ At the end of the following if/elif/elif block the output array will be set.
            each block handles the case where the element of the key along the split axis
            is a different type and converts the key from global indices to local indices. """
        lout = gout_full.copy()

        if (
            isinstance(key[self.split], (list, torch.Tensor, DNDarray, np.ndarray))
            and len(key[self.split]) > 1
        ):
            # advanced indexing, elements in the split dimension are adjusted to the local indices
            lkey = list(key)
            if isinstance(key[self.split], DNDarray):
                lkey[self.split] = key[self.split].larray

            if not isinstance(lkey[self.split], torch.Tensor):
                inds = torch.tensor(
                    lkey[self.split], dtype=torch.long, device=self.device.torch_device
                )
            elif lkey[self.split].dtype in [torch.bool, torch.uint8]:  # or torch.byte?
                # need to convert the bools to indices
                inds = torch.nonzero(lkey[self.split])
            else:
                inds = lkey[self.split]
            # todo: remove where in favor of nonzero? might be a speed upgrade. testing required
            loc_inds = torch.where((inds >= chunk_start) & (inds < chunk_end))
            # if there are no local indices on a process, then `arr` is empty
            # if local indices exist:
            if len(loc_inds[0]) != 0:
                # select same local indices for other (non-split) dimensions if necessary
                for i, k in enumerate(lkey):
                    if isinstance(k, (list, torch.Tensor, DNDarray)) and i != self.split:
                        lkey[i] = k[loc_inds]
                # correct local indices for offset
                inds = inds[loc_inds] - chunk_start
                lkey[self.split] = inds
                lout[new_split] = len(inds)
                arr = self.__array[tuple(lkey)].reshape(tuple(lout))
            elif len(loc_inds[0]) == 0:
                if new_split is not None:
                    lout[new_split] = len(loc_inds[0])
                else:
                    lout = [0] * len(gout_full)
                arr = torch.tensor([], dtype=self.larray.dtype, device=self.larray.device).reshape(
                    tuple(lout)
                )

        elif isinstance(key[self.split], slice):
            # standard slicing along the split axis,
            # adjust the slice start, stop, and step, then run it on the processes which have the requested data
            key = list(key)
            key[self.split] = stride_tricks.sanitize_slice(key[self.split], self.gshape[self.split])
            key_start, key_stop, key_step = (
                key[self.split].start,
                key[self.split].stop,
                key[self.split].step,
            )
            og_key_start = key_start
            st_pr = torch.where(key_start < chunk_ends)[0]
            st_pr = st_pr[0] if len(st_pr) > 0 else self.comm.size
            sp_pr = torch.where(key_stop >= chunk_starts)[0]
            sp_pr = sp_pr[-1] if len(sp_pr) > 0 else 0
            actives = list(range(st_pr, sp_pr + 1))
            if rank in actives:
                key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
                key_stop = counts[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
                key_start, key_stop = self.__xitem_get_key_start_stop(
                    rank, actives, key_start, key_stop, key_step, chunk_ends, og_key_start
                )
                key[self.split] = slice(key_start, key_stop, key_step)
                lout[new_split] = (
                    math.ceil((key_stop - key_start) / key_step)
                    if key_step is not None
                    else key_stop - key_start
                )
                arr = self.__array[tuple(key)].reshape(lout)
            else:
                lout[new_split] = 0
                arr = torch.empty(lout, dtype=self.__array.dtype, device=self.__array.device)

        elif self.__key_is_singular(key, self.split, self_proxy):
            # getting one item along split axis:
            key = list(key)
            if isinstance(key[self.split], list):
                key[self.split] = key[self.split].pop()
            elif isinstance(key[self.split], (torch.Tensor, DNDarray, np.ndarray)):
                key[self.split] = key[self.split].item()
            # translate negative index
            if key[self.split] < 0:
                key[self.split] += self.gshape[self.split]

            active_rank = torch.where(key[self.split] >= chunk_starts)[0][-1].item()
            # slice `self` on `active_rank`, allocate `arr` on all other ranks in preparation for Bcast
            if rank == active_rank:
                key[self.split] -= chunk_start.item()
                arr = self.__array[tuple(key)].reshape(tuple(lout))
            else:
                arr = torch.empty(tuple(lout), dtype=self.larray.dtype, device=self.larray.device)
            # broadcast result
            # TODO: Replace with `self.comm.Bcast(arr, root=active_rank)` after fixing #784
            arr = self.comm.bcast(arr, root=active_rank)
            if arr.device != self.larray.device:
                # todo: remove when unnecessary (also after #784)
                arr = arr.to(device=self.larray.device)

        return DNDarray(
            arr.type(l_dtype),
            gout_full if isinstance(gout_full, tuple) else tuple(gout_full),
            self.dtype,
            new_split,
            self.device,
            self.comm,
            balanced=True if new_split is None else None,
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

    def __int__(self) -> DNDarray:
        """
        Integer scalar casting.
        """
        return self.__cast(int)

    def is_balanced(self, force_check: bool = False) -> bool:
        """
        Determine if ``self`` is balanced evenly (or as evenly as possible) across all nodes
        distributed evenly (or as evenly as possible) across all processes.
        This is equivalent to returning ``self.balanced``. If no information
        is available (``self.balanced = None``), the balanced status will be
        assessed via collective communication.

        Parameters
        force_check : bool, optional
            If True, the balanced status of the ``DNDarray`` will be assessed via
            collective communication in any case.
        """
        if not force_check and self.balanced is not None:
            return self.balanced

        _, _, chk = self.comm.chunk(self.shape, self.split)
        test_lshape = tuple([x.stop - x.start for x in chk])
        balanced = 1 if test_lshape == self.lshape else 0

        out = self.comm.allreduce(balanced, MPI.SUM)
        balanced = True if out == self.comm.size else False
        return balanced

    def is_distributed(self) -> bool:
        """
        Determines whether the data of this ``DNDarray`` is distributed across multiple processes.
        """
        return self.split is not None and self.comm.is_distributed()

    @staticmethod
    def __key_is_singular(key: any, axis: int, self_proxy: torch.Tensor) -> bool:
        # determine if the key gets a singular item
        zeros = (0,) * (self_proxy.ndim - 1)
        return self_proxy[(*zeros[:axis], key[axis], *zeros[axis:])].ndim == 0

    @staticmethod
    def __key_adds_dimension(key: any, axis: int, self_proxy: torch.Tensor) -> bool:
        # determine if the key adds a new dimension
        zeros = (0,) * (self_proxy.ndim - 1)
        return self_proxy[(*zeros[:axis], key[axis], *zeros[axis:])].ndim == 2

    def item(self):
        """
        Returns the only element of a 1-element :class:`DNDarray`.
        Mirror of the pytorch command by the same name. If size of ``DNDarray`` is >1 element, then a ``ValueError`` is
        raised (by pytorch)

        Examples
        -------
        >>> import heat as ht
        >>> x = ht.zeros((1))
        >>> x.item()
        0.0
        """
        if self.size > 1:
            raise ValueError("only one-element DNDarrays can be converted to Python scalars")
        # make sure the element is on every process
        self.resplit_(None)
        return self.__array.item()

    def __len__(self) -> int:
        """
        The length of the ``DNDarray``, i.e. the number of items in the first dimension.
        """
        return self.shape[0]

    def numpy(self) -> np.array:
        """
        Returns a copy of the :class:`DNDarray` as numpy ndarray. If the ``DNDarray`` resides on the GPU, the underlying data will be copied to the CPU first.

        If the ``DNDarray`` is distributed, an MPI Allgather operation will be performed before converting to np.ndarray, i.e. each MPI process will end up holding a copy of the entire array in memory.  Make sure process memory is sufficient!

        Examples
        --------
        >>> import heat as ht
        T1 = ht.random.randn((10,8))
        T1.numpy()
        """
        dist = self.copy().resplit_(axis=None)
        return dist.larray.cpu().numpy()

    def __repr__(self) -> str:
        """
        Computes a printable representation of the passed DNDarray.
        """
        return printing.__str__(self)

    def ravel(self):
        """
        Flattens the ``DNDarray``.

        See Also
        --------
        :func:`~heat.core.manipulations.ravel`

        Examples
        --------
        >>> a = ht.ones((2,3), split=0)
        >>> b = a.ravel()
        >>> a[0,0] = 4
        >>> b
        DNDarray([4., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)
        """
        return manipulations.ravel(self)

    def redistribute_(
        self, lshape_map: Optional[torch.Tensor] = None, target_map: Optional[torch.Tensor] = None
    ):
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
        >>> target_map = torch.zeros((st.comm.size, 3), dtype=torch.int64)
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
            lshape_map = self.create_lshape_map(force_check=True)
        else:
            if not isinstance(lshape_map, torch.Tensor):
                raise TypeError(f"lshape_map must be a torch.Tensor, currently {type(lshape_map)}")
            if lshape_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    f"lshape_map must have the shape ({self.comm.size}, {len(self.gshape)}), currently {lshape_map.shape}"
                )
        if target_map is None:  # if no target map is given then it will balance the tensor
            _, _, chk = self.comm.chunk(self.shape, self.split)
            target_map = lshape_map.clone()
            target_map[..., self.split] = 0
            for pr in range(self.comm.size):
                target_map[pr, self.split] = self.comm.chunk(self.shape, self.split, rank=pr)[1][
                    self.split
                ]
            self.__balanced = True
        else:
            sanitation.sanitize_in_tensor(target_map)
            if target_map[..., self.split].sum() != self.shape[self.split]:
                raise ValueError(
                    f"Sum along the split axis of the target map must be equal to the shape in that dimension, currently {target_map[..., self.split]}"
                )
            if target_map.shape != (self.comm.size, len(self.gshape)):
                raise ValueError(
                    f"target_map must have the shape {(self.comm.size, len(self.gshape))}, currently {target_map.shape}"
                )
            # no info on balanced status
            self.__balanced = False
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

        self.__lshape_map = target_map

    def __redistribute_shuffle(
        self,
        snd_pr: Union[int, torch.Tensor],
        send_amt: Union[int, torch.Tensor],
        rcv_pr: Union[int, torch.Tensor],
        snd_dtype: torch.dtype,
    ):
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
            Receiving process
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

    def resplit_(self, axis: int = None):
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
        if self.comm.size == 1:
            self.__split = axis
            if axis is None:
                self.__partitions_dict__ = None
        if axis == self.split:
            return self

        self.__partitions_dict__ = None

        if axis is None:
            gathered = torch.empty(
                self.shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )
            counts, displs = self.counts_displs()
            self.comm.Allgatherv(self.__array, (gathered, counts, displs), recv_axis=self.split)
            self.__array = gathered
            self.__split = axis
            self.__lshape_map = None
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
            self.__lshape_map = None
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
                elif spr == rpr == rank:
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
                    rcv[k][0].Wait()
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
        self.__lshape_map = None
        return self

    def __setitem__(
        self,
        key: Union[int, Tuple[int, ...], List[int, ...]],
        value: Union[float, DNDarray, torch.Tensor],
    ):
        """
        Global item setter

        Parameters
        ----------
        key : Union[int, Tuple[int,...], List[int,...]]
            Index/indices to be set
        value: Union[float, DNDarray,torch.Tensor]
            Value to be set to the specified positions in the DNDarray (self)

        Notes
        -----
        If a ``DNDarray`` is given as the value to be set then the split axes are assumed to be equal.
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
        key = getattr(key, "copy()", key)
        try:
            if value.split != self.split:
                val_split = int(value.split)
                sp = self.split
                warnings.warn(
                    f"\nvalue.split {val_split} not equal to this DNDarray's split:"
                    f" {sp}. this may cause errors or unwanted behavior",
                    category=RuntimeWarning,
                )
        except (AttributeError, TypeError):
            pass

        # NOTE: for whatever reason, there is an inplace op which interferes with the abstraction
        # of this next block of code. this is shared with __getitem__. I attempted to abstract it
        # in a standard way, but it was causing errors in the test suite. If someone else is
        # motived to do this they are welcome to, but i have no time right now
        # print(key)
        if isinstance(key, DNDarray) and key.ndim == self.ndim:
            """if the key is a DNDarray and it has as many dimensions as self, then each of the
            entries in the 0th dim refer to a single element. To handle this, the key is split
            into the torch tensors for each dimension. This signals that advanced indexing is
            to be used."""
            key = manipulations.resplit(key)
            if key.larray.dtype in [torch.bool, torch.uint8]:
                key = indexing.nonzero(key)

            if key.ndim > 1:
                key = list(key.larray.split(1, dim=1))
                # key is now a list of tensors with dimensions (key.ndim, 1)
                # squeeze singleton dimension:
                key = [key[i].squeeze_(1) for i in range(len(key))]
            else:
                key = [key]
        elif not isinstance(key, tuple):
            """this loop handles all other cases. DNDarrays which make it to here refer to
            advanced indexing slices, as do the torch tensors. Both DNDaarrys and torch.Tensors
            are cast into lists here by PyTorch. lists mean advanced indexing will be used"""
            h = [slice(None, None, None)] * self.ndim
            if isinstance(key, DNDarray):
                key = manipulations.resplit(key)
                if key.larray.dtype in [torch.bool, torch.uint8]:
                    h[0] = torch.nonzero(key.larray).flatten()  # .tolist()
                else:
                    h[0] = key.larray.tolist()
            elif isinstance(key, torch.Tensor):
                if key.dtype in [torch.bool, torch.uint8]:
                    # (coquelin77) im not sure why this works without being a list...but it does...for now
                    h[0] = torch.nonzero(key).flatten()  # .tolist()
                else:
                    h[0] = key.tolist()
            else:
                h[0] = key
            key = list(h)

        # key must be torch-proof
        if isinstance(key, (list, tuple)):
            key = list(key)
            for i, k in enumerate(key):
                try:  # extract torch tensor
                    k = manipulations.resplit(k)
                    key[i] = k.larray
                except AttributeError:
                    pass
                # remove bools from a torch tensor in favor of indexes
                try:
                    if key[i].dtype in [torch.bool, torch.uint8]:
                        key[i] = torch.nonzero(key[i]).flatten()
                except (AttributeError, TypeError):
                    pass

        key = list(key)

        # ellipsis stuff
        key_classes = [type(n) for n in key]
        # if any(isinstance(n, ellipsis) for n in key):
        n_elips = key_classes.count(type(...))
        if n_elips > 1:
            raise ValueError("key can only contain 1 ellipsis")
        elif n_elips == 1:
            # get which item is the ellipsis
            ell_ind = key_classes.index(type(...))
            kst = key[:ell_ind]
            kend = key[ell_ind + 1 :]
            slices = [slice(None)] * (self.ndim - (len(kst) + len(kend)))
            key = kst + slices + kend
        # ---------- end ellipsis stuff -------------

        for c, k in enumerate(key):
            try:
                key[c] = k.item()
            except (AttributeError, ValueError, RuntimeError):
                pass

        rank = self.comm.rank
        if self.split is not None:
            counts, chunk_starts = self.counts_displs()
        else:
            counts, chunk_starts = 0, [0] * self.comm.size
        counts = torch.tensor(counts, device=self.device.torch_device)
        chunk_starts = torch.tensor(chunk_starts, device=self.device.torch_device)
        chunk_ends = chunk_starts + counts
        chunk_start = chunk_starts[rank]
        chunk_end = chunk_ends[rank]
        # determine which elements are on the local process (if the key is a torch tensor)
        try:
            # if isinstance(key[self.split], torch.Tensor):
            filter_key = torch.nonzero(
                (chunk_start <= key[self.split]) & (key[self.split] < chunk_end)
            )
            for k in range(len(key)):
                try:
                    key[k] = key[k][filter_key].flatten()
                except TypeError:
                    pass
        except TypeError:  # this will happen if the key doesnt have that many
            pass

        key = tuple(key)

        if not self.is_distributed():
            return self.__setter(key, value)  # returns None

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

        self_proxy = self.__torch_proxy__()

        # if the value is a DNDarray, the divisions need to be balanced:
        #   this means that we need to know how much data is where for both DNDarrays
        #   if the value data is not in the right place, then it will need to be moved

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

            if (
                isinstance(value, type(self))
                and value.split is not None
                and value.shape[self.split] != self.shape[self.split]
            ):
                # setting elements in self with a DNDarray which is not the same size in the
                # split dimension
                local_keys = []
                # below is used if the target needs to be reshaped
                target_reshape_map = torch.zeros(
                    (self.comm.size, self.ndim), dtype=torch.int64, device=self.device.torch_device
                )
                for r in range(self.comm.size):
                    if r not in actives:
                        loc_key = key.copy()
                        loc_key[self.split] = slice(0, 0, 0)
                    else:
                        key_start_l = 0 if r != actives[0] else key_start - chunk_starts[r]
                        key_stop_l = ends[r] if r != actives[-1] else key_stop - chunk_starts[r]
                        key_start_l, key_stop_l = self.__xitem_get_key_start_stop(
                            r, actives, key_start_l, key_stop_l, key_step, chunk_ends, og_key_start
                        )
                        loc_key = key.copy()
                        loc_key[self.split] = slice(key_start_l, key_stop_l, key_step)

                        gout_full = torch.tensor(
                            self_proxy[loc_key].shape, device=self.device.torch_device
                        )
                        target_reshape_map[r] = gout_full
                    local_keys.append(loc_key)

                key = local_keys[rank]
                value = value.redistribute(target_map=target_reshape_map)

                if rank not in actives:
                    return  # non-active ranks can exit here

                chunk_starts_v = target_reshape_map[:, self.split]
                value_slice = [slice(None, None, None)] * value.ndim
                step2 = key_step if key_step is not None else 1
                key_start = (chunk_starts_v[rank] - og_key_start).item()

                key_start = max(key_start, 0)
                key_stop = key_start + key_stop
                slice_loc = min(self.split, value.ndim - 1)
                value_slice[slice_loc] = slice(
                    key_start, math.ceil(torch.true_divide(key_stop, step2)), 1
                )

                self.__setter(tuple(key), value.larray)
                return

            # if rank in actives:
            if rank not in actives:
                return  # non-active ranks can exit here
            key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
            key_stop = ends[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
            key_start, key_stop = self.__xitem_get_key_start_stop(
                rank, actives, key_start, key_stop, key_step, chunk_ends, og_key_start
            )
            key[self.split] = slice(key_start, key_stop, key_step)

            # todo: need to slice the values to be the right size...
            if isinstance(value, (torch.Tensor, type(self))):
                # if its a torch tensor, it is assumed to exist on all processes
                value_slice = [slice(None, None, None)] * value.ndim
                step2 = key_step if key_step is not None else 1
                key_start = (chunk_starts[rank] - og_key_start).item()
                key_start = max(key_start, 0)
                key_stop = key_start + key_stop
                slice_loc = min(self.split, value.ndim - 1)
                value_slice[slice_loc] = slice(
                    key_start, math.ceil(torch.true_divide(key_stop, step2)), 1
                )
                self.__setter(tuple(key), value[tuple(value_slice)])
            else:
                self.__setter(tuple(key), value)
        elif isinstance(key[self.split], (torch.Tensor, list)):
            key = list(key)
            key[self.split] -= chunk_start
            if len(key[self.split]) != 0:
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

    def __setter(
        self,
        key: Union[int, Tuple[int, ...], List[int, ...]],
        value: Union[float, DNDarray, torch.Tensor],
    ):
        """
        Utility function for checking ``value`` and forwarding to :func:``__setitem__``

        Raises
        -------------
        NotImplementedError
            If the type of ``value`` ist not supported
        """
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
            raise NotImplementedError(f"Not implemented for {value.__class__.__name__}")

    def __str__(self) -> str:
        """
        Computes a string representation of the passed ``DNDarray``.
        """
        return printing.__str__(self)

    def tolist(self, keepsplit: bool = False) -> List:
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
            return self.resplit(axis=None).__array.tolist()

        return self.__array.tolist()

    def __torch_proxy__(self) -> torch.Tensor:
        """
        Return a 1-element `torch.Tensor` strided as the global `self` shape.
        Used internally for sanitation purposes.
        """
        return torch.ones((1,), dtype=torch.int8, device=self.larray.device).as_strided(
            self.gshape, [0] * self.ndim
        )

    @staticmethod
    def __xitem_get_key_start_stop(
        rank: int,
        actives: list,
        key_st: int,
        key_sp: int,
        step: int,
        ends: torch.Tensor,
        og_key_st: int,
    ) -> Tuple[int, int]:
        # this does some basic logic for adjusting the starting and stoping of the a key for
        #   setitem and getitem
        if step is not None and rank > actives[0]:
            offset = (ends[rank - 1] - og_key_st) % step
            if step > 2 and offset > 0:
                key_st += step - offset
            elif step == 2 and offset > 0:
                key_st += (ends[rank - 1] - og_key_st) % step
        if isinstance(key_st, torch.Tensor):
            key_st = key_st.item()
        if isinstance(key_sp, torch.Tensor):
            key_sp = key_sp.item()
        return key_st, key_sp


# HeAT imports at the end to break cyclic dependencies
from . import complex_math
from . import devices
from . import factories
from . import indexing
from . import linalg
from . import manipulations
from . import printing
from . import rounding
from . import sanitation
from . import statistics
from . import stride_tricks
from . import tiling

from .devices import Device
from .stride_tricks import sanitize_axis
from .types import datatype, canonical_heat_type
