"""Provides HeAT's core data structure, the DNDarray, a distributed n-dimensional array"""
from __future__ import annotations

import math
import numpy as np
import torch
import warnings

from inspect import stack
from mpi4py import MPI
from pathlib import Path
from typing import List, Union, Tuple, TypeVar, Optional, Iterable

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
        print("DEBUGGING: larray setter")
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
        try:
            itemsize = self.larray.untyped_storage().element_size()
        except AttributeError:
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

    def collect_(self, target_rank: Optional[int] = 0) -> None:
        """
        A method collecting a distributed DNDarray to one MPI rank, chosen by the `target_rank` variable.
        It is a specific case of the ``redistribute_`` method.

        Parameters
        ----------
        target_rank : int, optional
            The rank to which the DNDarray will be collected. Default: 0.

        Raises
        ------
        TypeError
            If the target rank is not an integer.
        ValueError
            If the target rank is out of bounds.

        Examples
        --------
        >>> st = ht.ones((50, 81, 67), split=2)
        >>> print(st.lshape)
        [0/2] (50, 81, 23)
        [1/2] (50, 81, 22)
        [2/2] (50, 81, 22)
        >>> st.collect_()
        >>> print(st.lshape)
        [0/2] (50, 81, 67)
        [1/2] (50, 81, 0)
        [2/2] (50, 81, 0)
        >>> st.collect_(1)
        >>> print(st.lshape)
        [0/2] (50, 81, 0)
        [1/2] (50, 81, 67)
        [2/2] (50, 81, 0)
        """
        if not isinstance(target_rank, int):
            raise TypeError(f"target rank must be of type int , but was {type(target_rank)}")
        if target_rank >= self.comm.size:
            raise ValueError("target rank is out of bounds")
        if not self.is_distributed():
            return

        target_map = self.lshape_map.clone()
        target_map[:, self.split] = 0
        target_map[target_rank, self.split] = self.gshape[self.split]
        self.redistribute_(target_map=target_map)

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

    def __process_key(
        arr: DNDarray,
        key: Union[Tuple[int, ...], List[int, ...]],
        return_local_indices: Optional[bool] = False,
        op: Optional[str] = None,
    ) -> Tuple:
        """
        Private method to process the key used for indexing a ``DNDarray`` so that it can be applied to the process-local data, i.e. `key` must be "torch-proof".
        In a processed key:
        - any ellipses or newaxis have been replaced with the appropriate number of slice objects
        - ndarrays and DNDarrays have been converted to torch tensors
        - the dimensionality is the same as the ``DNDarray`` it indexes
        This function also manipulates `arr` if necessary, inserting and/or transposing dimensions as indicated by `key`. It calculates the output shape, split axis and balanced status of the indexed array.

        Parameters
        ----------
        arr : DNDarray
            The ``DNDarray`` to be indexed
        key : int, Tuple[int, ...], List[int, ...]
            The key used for indexing
        return_local_indices : bool, optional
            Whether to return the process-local indices of the key in the split dimension. This is only possible when the indexing key in the split dimension is ordered e.g. `split_key_is_ordered == 1`. Default: False
        op : str, optional
            The indexing operation that the key is being processed for. Get be "get" for `__getitem__` or "set" for `__setitem__`. Default: "get".

        Returns
        -------
        arr : DNDarray
            The ``DNDarray`` to be indexed. Its dimensions might have been modified if advanced, dimensional, broadcasted indexing is used.
        key : Union(Tuple[Any, ...], DNDarray, np.ndarray, torch.Tensor, slice, int, List[int, ...])
            The processed key ready for indexing ``arr``. Its dimensions match the (potentially modified) dimensions of ``arr``.
            Note: the key indices along the split axis are LOCAL indices, i.e. refer to the process-local data, if ordered indexing is used. Otherwise, they are GLOBAL indices, referring to the global memory-distributed DNDarray. Communication to extract the non-ordered elements of the input ``DNDarray`` is handled by the ``__getitem__`` function.
        output_shape : Tuple[int, ...]
            The shape of the output ``DNDarray``
        new_split : int
            The new split axis
        split_key_is_ordered : int
            Whether the split key is sorted or ordered. Can be 1: ascending, 0: not ordered, -1: descending order.
        out_is_balanced : bool
            Whether the output ``DNDarray`` is balanced
        root : int
            The root process for the ``MPI.Bcast`` call when single-element indexing along the split axis is used
        backwards_transpose_axes : Tuple[int, ...]
            The axes to transpose the input ``DNDarray`` back to its original shape if it has been transposed for advanced indexing
        """
        output_shape = list(arr.gshape)
        split_bookkeeping = [None] * arr.ndim
        new_split = arr.split
        arr_is_distributed = False
        if arr.split is not None:
            split_bookkeeping[arr.split] = "split"
            if arr.is_distributed():
                counts, displs = arr.counts_displs()
                arr_is_distributed = True

        advanced_indexing = False
        split_key_is_ordered = 1
        out_is_balanced = False
        root = None
        backwards_transpose_axes = tuple(range(arr.ndim))

        if isinstance(key, list):
            try:
                key = torch.tensor(key, device=arr.larray.device)
            except RuntimeError:
                raise IndexError("Invalid indices: expected a list of integers, got {}".format(key))
        if isinstance(key, (DNDarray, torch.Tensor, np.ndarray)):
            if key.dtype in (bool, uint8, torch.bool, torch.uint8, np.bool_, np.uint8):
                # boolean indexing: shape must match arr.shape
                if not tuple(key.shape) == arr.shape:
                    raise IndexError(
                        "Boolean index of shape {} does not match indexed array of shape {}".format(
                            tuple(key.shape), arr.shape
                        )
                    )
                try:
                    # key is DNDarray or ndarray
                    key = key.copy()
                except AttributeError:
                    # key is torch tensor
                    key = key.clone()
                if not arr_is_distributed:
                    try:
                        # key is DNDarray, extract torch tensor
                        key = key.larray
                    except AttributeError:
                        pass
                    try:
                        # key is torch tensor
                        key = key.nonzero(as_tuple=True)
                    except TypeError:
                        # key is np.ndarray
                        key = key.nonzero()
                        # convert to torch tensor
                        key = tuple(torch.tensor(k, device=arr.larray.device) for k in key)
                    output_shape = tuple(key[0].shape)
                    new_split = None if arr.split is None else 0
                    out_is_balanced = True
                    split_key_is_ordered = 1
                    return (
                        arr,
                        key,
                        output_shape,
                        new_split,
                        split_key_is_ordered,
                        out_is_balanced,
                        root,
                        backwards_transpose_axes,
                    )

                # arr is distributed
                if not isinstance(key, DNDarray) or not key.is_distributed():
                    key = factories.array(key, split=arr.split, device=arr.device)
                else:
                    if key.split != arr.split:
                        raise IndexError(
                            "Boolean index does not match distribution scheme of indexed array. index.split is {}, array.split is {}".format(
                                key.split, arr.split
                            )
                        )
                if arr.split == 0:
                    # ensure arr and key are aligned
                    key.redistribute_(target_map=arr.lshape_map)
                    # transform key to sequence of indexing (1-D) arrays
                    key = list(key.nonzero())
                    output_shape = key[0].shape
                    new_split = 0
                    split_key_is_ordered = 1
                    out_is_balanced = False
                    for i, k in enumerate(key):
                        key[i] = k.larray
                    if return_local_indices:
                        key[arr.split] -= displs[arr.comm.rank]
                    key = tuple(key)
                else:
                    key = key.larray.nonzero(as_tuple=False)
                    # construct global key array
                    nz_size = torch.tensor(key.shape[0], device=key.device, dtype=key.dtype)
                    arr.comm.Allreduce(MPI.IN_PLACE, nz_size, MPI.SUM)
                    key_gshape = (nz_size.item(), arr.ndim)
                    key[:, arr.split] += displs[arr.comm.rank]
                    key_split = 0
                    key = DNDarray(
                        key,
                        gshape=key_gshape,
                        dtype=canonical_heat_type(key.dtype),
                        split=key_split,
                        device=arr.device,
                        comm=arr.comm,
                        balanced=False,
                    )
                    # vectorized sorting along axis 0
                    key.balance_()
                    key = manipulations.unique(key, axis=0, return_inverse=False)
                    # return tuple key
                    key = list(key.larray.split(1, dim=1))
                    for i, k in enumerate(key):
                        key[i] = k.squeeze(1)
                    key = tuple(key)

                    output_shape = (key[0].shape[0],)
                    new_split = 0
                    split_key_is_ordered = 0
                    out_is_balanced = True
                return (
                    arr,
                    key,
                    output_shape,
                    new_split,
                    split_key_is_ordered,
                    out_is_balanced,
                    root,
                    backwards_transpose_axes,
                )

            # advanced indexing on first dimension: first dim will expand to shape of key
            output_shape = tuple(list(key.shape) + output_shape[1:])
            print("DEBUGGING ADV IND: output_shape = ", output_shape)
            # adjust split axis accordingly
            if arr_is_distributed:
                if arr.split != 0:
                    # split axis is not affected
                    split_bookkeeping = [None] * key.ndim + split_bookkeeping[1:]
                    new_split = (
                        split_bookkeeping.index("split") if "split" in split_bookkeeping else None
                    )
                    out_is_balanced = arr.balanced
                else:
                    # split axis is affected
                    if key.ndim > 1:
                        try:
                            key_numel = key.numel()
                        except AttributeError:
                            key_numel = key.size
                        if key_numel == arr.shape[0]:
                            new_split = tuple(key.shape).index(arr.shape[0])
                        else:
                            new_split = key.ndim - 1
                        try:
                            key_split = key[new_split].larray
                            sorted, _ = key_split.sort(stable=True)
                        except AttributeError:
                            key_split = key[new_split]
                            sorted = key_split.sort()
                    else:
                        new_split = 0
                        # assess if key is sorted along split axis
                        try:
                            # DNDarray key
                            sorted, _ = torch.sort(key.larray, stable=True)
                            split_key_is_ordered = torch.tensor(
                                (key.larray == sorted).all(),
                                dtype=torch.uint8,
                                device=key.larray.device,
                            )
                            if key.split is not None:
                                out_is_balanced = key.balanced
                                split_key_is_ordered = factories.array(
                                    [split_key_is_ordered],
                                    is_split=0,
                                    device=arr.device,
                                    copy=False,
                                ).all()
                            key = key.larray
                        except AttributeError:
                            # torch or ndarray key
                            try:
                                sorted, _ = torch.sort(key, stable=True)
                            except TypeError:
                                # ndarray key
                                sorted = torch.tensor(np.sort(key), device=arr.larray.device)
                            split_key_is_ordered = torch.tensor(
                                key == sorted, dtype=torch.uint8
                            ).item()
                            if not split_key_is_ordered:
                                # prepare for distributed non-ordered indexing: distribute torch/numpy key
                                key = factories.array(key, split=0, device=arr.device).larray
                                out_is_balanced = True
                        if split_key_is_ordered:
                            # extract local key
                            cond1 = key >= displs[arr.comm.rank]
                            cond2 = key < displs[arr.comm.rank] + counts[arr.comm.rank]
                            key = key[cond1 & cond2]
                            if return_local_indices:
                                key -= displs[arr.comm.rank]
                            out_is_balanced = False
            else:
                try:
                    out_is_balanced = key.balanced
                    new_split = key.split
                    key = key.larray
                except AttributeError:
                    # torch or numpy key, non-distributed indexed array
                    out_is_balanced = True
                    new_split = None
            return (
                arr,
                key,
                output_shape,
                new_split,
                split_key_is_ordered,
                out_is_balanced,
                root,
                backwards_transpose_axes,
            )

        key = list(key) if isinstance(key, Iterable) else [key]

        # check for ellipsis, newaxis. NB: (np.newaxis is None)==True
        add_dims = sum(k is None for k in key)
        ellipsis = sum(isinstance(k, type(...)) for k in key)
        if ellipsis > 1:
            raise ValueError("indexing key can only contain 1 Ellipsis (...)")
        if ellipsis:
            # key contains exactly 1 ellipsis
            # replace with explicit `slice(None)` for affected dimensions
            # output_shape, split_bookkeeping not affected
            expand_key = [slice(None)] * (arr.ndim + add_dims)
            ellipsis_index = key.index(...)
            ellipsis_dims = arr.ndim - (len(key) - ellipsis - add_dims)
            expand_key[:ellipsis_index] = key[:ellipsis_index]
            expand_key[ellipsis_index + ellipsis_dims :] = key[ellipsis_index + 1 :]
            key = expand_key
            print("DEBUGGING: ELLIPSIS: ", key)
        while add_dims > 0:
            # expand array dims: output_shape, split_bookkeeping to reflect newaxis
            # replace newaxis with slice(None) in key
            for i, k in reversed(list(enumerate(key))):
                if k is None:
                    key[i] = slice(None)
                    arr = arr.expand_dims(i - add_dims + 1)
                    output_shape = (
                        output_shape[: i - add_dims + 1] + [1] + output_shape[i - add_dims + 1 :]
                    )
                    split_bookkeeping = (
                        split_bookkeeping[: i - add_dims + 1]
                        + [None]
                        + split_bookkeeping[i - add_dims + 1 :]
                    )
                    add_dims -= 1

        # recalculate new_split, transpose_axes after dimensions manipulation
        new_split = split_bookkeeping.index("split") if "split" in split_bookkeeping else None
        transpose_axes, backwards_transpose_axes = tuple(range(arr.ndim)), tuple(range(arr.ndim))
        # check for advanced indexing and slices
        print("DEBUGGING: key = ", key)
        advanced_indexing_dims = []
        advanced_indexing_shapes = []
        lose_dims = 0
        for i, k in enumerate(key):
            if np.isscalar(k) or getattr(k, "ndim", 1) == 0:
                # single-element indexing along axis i
                try:
                    output_shape[i], split_bookkeeping[i] = None, None
                except IndexError:
                    raise IndexError(
                        f"Too many indices for DNDarray: DNDarray is {arr.ndim}-dimensional, but {len(key)} dimensions were indexed"
                    )
                lose_dims += 1
                if i == arr.split:
                    key[i], root = arr.__process_scalar_key(
                        k, indexed_axis=i, return_local_indices=return_local_indices
                    )
                else:
                    key[i], _ = arr.__process_scalar_key(
                        k, indexed_axis=i, return_local_indices=False
                    )
            elif isinstance(k, Iterable) or isinstance(k, DNDarray):
                advanced_indexing = True
                advanced_indexing_dims.append(i)
                if isinstance(k, DNDarray):
                    advanced_indexing_shapes.append(k.gshape)
                    if arr_is_distributed and i == arr.split:
                        # we have no info on order of indices
                        split_key_is_ordered = 0
                        # redistribute key along last axis to match split axis of indexed array
                        k = k.resplit(-1)
                        out_is_balanced = True
                    key[i] = k.larray
                elif not isinstance(k, torch.Tensor):
                    key[i] = torch.tensor(k, dtype=torch.int64, device=arr.larray.device)
                    advanced_indexing_shapes.append(tuple(key[i].shape))
                    # IMPORTANT: here we assume that torch or ndarray key is THE SAME SET OF GLOBAL INDICES on every rank
                    if arr_is_distributed and i == arr.split:
                        # make no assumption on data locality wrt key
                        out_is_balanced = None
                        # assess if indices are in ascending order
                        if (
                            key[i].ndim == 1
                            and (key[i] == torch.sort(key[i], stable=True)[0]).all()
                        ):
                            split_key_is_ordered = 1
                            # extract local key
                            cond1 = key[i] >= displs[arr.comm.rank]
                            cond2 = key[i] < displs[arr.comm.rank] + counts[arr.comm.rank]
                            key[i] = key[i][cond1 & cond2]
                            if return_local_indices:
                                key[i] -= displs[arr.comm.rank]
                        else:
                            split_key_is_ordered = 0

            elif isinstance(k, slice) and k != slice(None):
                start, stop, step = k.start, k.stop, k.step
                if start is None:
                    start = 0
                elif start < 0:
                    start += arr.gshape[i]
                if stop is None:
                    stop = arr.gshape[i]
                elif stop < 0:
                    stop += arr.gshape[i]
                if step is None:
                    step = 1
                if step < 0 and start > stop:
                    print("TEST LOCAL SLICE: ", arr.__get_local_slice(k))
                    # PyTorch doesn't support negative step as of 1.13
                    # Lazy solution, potentially large memory footprint
                    # TODO: implement ht.fromiter (implemented in ASSET_ht)
                    key[i] = torch.tensor(list(range(start, stop, step)), device=arr.larray.device)
                    output_shape[i] = len(key[i])
                    split_key_is_ordered = -1
                    if arr_is_distributed and new_split == i:
                        if op == "set":
                            # setitem: flip key and keep process-local indices
                            key[i] = key[i].flip(0)
                            cond1 = key[i] >= displs[arr.comm.rank]
                            cond2 = key[i] < displs[arr.comm.rank] + counts[arr.comm.rank]
                            key[i] = key[i][cond1 & cond2]
                            if return_local_indices:
                                key[i] -= displs[arr.comm.rank]
                        else:
                            # getitem: distribute key and proceed with non-ordered indexing
                            key[i] = factories.array(
                                key[i], split=0, device=arr.device, copy=False
                            ).larray
                            print("DEBUGGING: key[i] = ", key[i])
                            out_is_balanced = True
                elif step > 0 and start < stop:
                    output_shape[i] = int(torch.tensor((stop - start) / step).ceil().item())
                    if arr_is_distributed and new_split == i:
                        split_key_is_ordered = 1
                        out_is_balanced = False
                        local_arr_end = displs[arr.comm.rank] + counts[arr.comm.rank]
                        if stop > displs[arr.comm.rank] and start < local_arr_end:
                            print(
                                "stop, start, displs[arr.comm.rank], displs[arr.comm.rank] + counts[arr.comm.rank] = ",
                                stop,
                                start,
                                displs[arr.comm.rank],
                                displs[arr.comm.rank] + counts[arr.comm.rank],
                            )
                            index_in_cycle = (displs[arr.comm.rank] - start) % step
                            if start >= displs[arr.comm.rank]:
                                # slice begins on current rank
                                local_start = start - displs[arr.comm.rank]
                            else:
                                local_start = 0 if index_in_cycle == 0 else step - index_in_cycle
                            if stop <= local_arr_end:
                                # slice ends on current rank
                                local_stop = stop - displs[arr.comm.rank]
                            else:
                                local_stop = local_arr_end
                            key[i] = slice(local_start, local_stop, step)
                        else:
                            key[i] = slice(0, 0)
                elif step == 0:
                    raise ValueError("Slice step cannot be zero")
                else:
                    key[i] = slice(0, 0)
                    output_shape[i] = 0

        if advanced_indexing:
            print("ADV IND KEY = ", key)
            print("DEBUGGING: advanced_indexing_shapes = ", advanced_indexing_shapes)
            # shapes of indexing arrays must be broadcastable
            try:
                broadcasted_shape = torch.broadcast_shapes(*advanced_indexing_shapes)
            except RuntimeError:
                raise IndexError(
                    "Shape mismatch: indexing arrays could not be broadcast together with shapes: {}".format(
                        advanced_indexing_shapes
                    )
                )
            add_dims = len(broadcasted_shape) - len(advanced_indexing_dims)
            if (
                len(advanced_indexing_dims) == 1
                or list(range(advanced_indexing_dims[0], advanced_indexing_dims[-1] + 1))
                == advanced_indexing_dims
            ):
                # dimensions affected by advanced indexing are consecutive:
                output_shape[
                    advanced_indexing_dims[0] : advanced_indexing_dims[0]
                    + len(advanced_indexing_dims)
                ] = broadcasted_shape
                split_bookkeeping = (
                    split_bookkeeping[: advanced_indexing_dims[0]]
                    + [None] * add_dims
                    + split_bookkeeping[advanced_indexing_dims[0] :]
                )
                print("ADV IND output_shape = ", output_shape)
            else:
                # advanced-indexing dimensions are not consecutive:
                # transpose array to make the advanced-indexing dimensions consecutive as the first dimensions
                non_adv_ind_dims = list(
                    i for i in range(arr.ndim) if i not in advanced_indexing_dims
                )
                # keep track of transpose axes order, to be able to transpose back later
                transpose_axes = tuple(advanced_indexing_dims + non_adv_ind_dims)
                arr = arr.transpose(transpose_axes)
                backwards_transpose_axes = tuple(
                    torch.tensor(transpose_axes, device=arr.larray.device)
                    .argsort(stable=True)
                    .tolist()
                )
                # output shape and split bookkeeping
                output_shape = list(output_shape[i] for i in transpose_axes)
                output_shape[: len(advanced_indexing_dims)] = broadcasted_shape
                split_bookkeeping = list(split_bookkeeping[i] for i in transpose_axes)
                split_bookkeeping = [None] * add_dims + split_bookkeeping
                # modify key to match the new dimension order
                key = [key[i] for i in advanced_indexing_dims] + [key[i] for i in non_adv_ind_dims]
                # update advanced-indexing dims
                advanced_indexing_dims = list(range(len(advanced_indexing_dims)))

        # expand key to match the number of dimensions of the DNDarray
        if arr.ndim > len(key):
            key += [slice(None)] * (arr.ndim - len(key))

        key = tuple(key)
        for i in range(output_shape.count(None)):
            lost_dim = output_shape.index(None)
            output_shape.remove(None)
            split_bookkeeping = split_bookkeeping[:lost_dim] + split_bookkeeping[lost_dim + 1 :]
        output_shape = tuple(output_shape)
        new_split = split_bookkeeping.index("split") if "split" in split_bookkeeping else None
        print(
            "key, output_shape, new_split, split_key_is_ordered, out_is_balanced = ",
            key,
            output_shape,
            new_split,
            split_key_is_ordered,
            out_is_balanced,
        )
        return (
            arr,
            key,
            output_shape,
            new_split,
            split_key_is_ordered,
            out_is_balanced,
            root,
            backwards_transpose_axes,
        )

    def __process_scalar_key(
        arr: DNDarray,
        key: Union[int, DNDarray, torch.Tensor, np.ndarray],
        indexed_axis: int,
        return_local_indices: Optional[bool] = False,
    ) -> Tuple(int, int):
        """
        Private method to process a single-item scalar key used for indexing a ``DNDarray``.

        """
        device = arr.larray.device
        try:
            # is key an ndarray or DNDarray or torch.Tensor?
            key = key.item()
        except AttributeError:
            # key is already an integer, do nothing
            pass
        if not arr.is_distributed():
            root = None
            return key, root
        if arr.split == indexed_axis:
            # adjust negative key
            if key < 0:
                key += arr.shape[0]
            # work out active process
            _, displs = arr.counts_displs()
            if key in displs:
                root = displs.index(key)
            else:
                displs = torch.cat(
                    (
                        torch.tensor(displs, device=device),
                        torch.tensor(key, device=device).reshape(-1),
                    ),
                    dim=0,
                )
                _, sorted_indices = displs.unique(sorted=True, return_inverse=True)
                root = sorted_indices[-1].item() - 1
                displs = displs.tolist()
            # correct key for rank-specific displacement
            if return_local_indices:
                if arr.comm.rank == root:
                    key -= displs[root]
        else:
            root = None
        return key, root

    def __get_local_slice(self, key: slice):
        split = self.split
        if split is None:
            return key
        key = stride_tricks.sanitize_slice(key, self.shape[split])
        start, stop, step = key.start, key.stop, key.step
        if step < 0:  # NOT supported by torch, should be filtered by torch_proxy
            key = self.__get_local_slice(slice(stop + 1, start + 1, abs(step)))
            if key is None:
                return None
            start, stop, step = key.start, key.stop, key.step
            return slice(key.stop - 1, key.start - 1, -1 * key.step)

        _, offsets = self.counts_displs()
        offset = offsets[self.comm.rank]
        range_proxy = range(self.lshape[split])
        local_inds = range_proxy[start - offset : stop - offset]  # only works if stop - offset > 0
        local_inds = local_inds[max(offset - start, 0) % step :: step]
        if len(local_inds) and stop > offset:
            # otherwise if (stop-offset) > -self.lshape[split] this can index into the local chunk despite ending before it
            return slice(local_inds.start, local_inds.stop, local_inds.step)
        return None

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
        # key can be: int, tuple, list, slice, DNDarray, torch tensor, numpy array, or sequence thereof
        # Trivial cases
        print("DEBUGGING: RAW KEY = ", key, type(key))

        if key is None:
            return self.expand_dims(0)
        if (
            key is ... or isinstance(key, slice) and key == slice(None)
        ):  # latter doesnt work with torch for 0-dim tensors
            return self

        original_split = self.split

        # Single-element indexing
        scalar = np.isscalar(key) or getattr(key, "ndim", 1) == 0
        if scalar:
            # single-element indexing on axis 0
            if self.ndim == 0:
                raise IndexError(
                    "Too many indices for DNDarray: DNDarray is 0-dimensional, but 1 were indexed"
                )
            output_shape = self.gshape[1:]
            if original_split is None or original_split == 0:
                output_split = None
            else:
                output_split = original_split - 1
            split_key_is_ordered = 1
            out_is_balanced = True
            backwards_transpose_axes = tuple(range(self.ndim))
            key, root = self.__process_scalar_key(key, indexed_axis=0, return_local_indices=True)
            if root is None:
                # early out for single-element indexing not affecting split axis
                indexed_arr = self.larray[key]
                indexed_arr = DNDarray(
                    indexed_arr,
                    gshape=output_shape,
                    dtype=self.dtype,
                    split=output_split,
                    device=self.device,
                    comm=self.comm,
                    balanced=out_is_balanced,
                )
                return indexed_arr
        else:
            # process multi-element key
            (
                self,
                key,
                output_shape,
                output_split,
                split_key_is_ordered,
                out_is_balanced,
                root,
                backwards_transpose_axes,
            ) = self.__process_key(key, return_local_indices=True)

        if not self.is_distributed():
            # key is torch-proof, index underlying torch tensor
            print("DEBUGGING: key = ", key)
            indexed_arr = self.larray[key]
            # transpose array back if needed
            self = self.transpose(backwards_transpose_axes)
            return DNDarray(
                indexed_arr,
                gshape=output_shape,
                dtype=self.dtype,
                split=output_split,
                device=self.device,
                comm=self.comm,
                balanced=out_is_balanced,
            )

        if split_key_is_ordered == 1:
            if root is not None:
                # single-element indexing along split axis
                # prepare for Bcast: allocate buffer on all processes
                if self.comm.rank == root:
                    indexed_arr = self.larray[key]
                else:
                    indexed_arr = torch.zeros(
                        output_shape, dtype=self.larray.dtype, device=self.larray.device
                    )
                # broadcast result to all processes
                self.comm.Bcast(indexed_arr, root=root)
                indexed_arr = DNDarray(
                    indexed_arr,
                    gshape=output_shape,
                    dtype=self.dtype,
                    split=output_split,
                    device=self.device,
                    comm=self.comm,
                    balanced=out_is_balanced,
                )
                # transpose array back if needed
                self = self.transpose(backwards_transpose_axes)
                return indexed_arr

            # root is None, i.e. indexing does not affect split axis, apply as is
            indexed_arr = self.larray[key]
            # transpose array back if needed
            self = self.transpose(backwards_transpose_axes)
            return DNDarray(
                indexed_arr,
                gshape=output_shape,
                dtype=self.dtype,
                split=output_split,
                device=self.device,
                balanced=out_is_balanced,
                comm=self.comm,
            )

        # key is not ordered along self.split
        # key is tuple of torch.Tensor or mix of torch.Tensors and slices
        _, displs = self.counts_displs()

        # determine whether indexed array will be 1D or nD
        try:
            return_1d = getattr(key, "ndim") == self.ndim
            send_axis = 0
        except AttributeError:
            # key is tuple of torch tensors
            key_shapes = []
            for k in key:
                key_shapes.append(getattr(k, "shape", None))
            print("KEY SHAPES = ", key_shapes)
            return_1d = key_shapes.count(key_shapes[original_split]) == self.ndim
            # check for broadcasted indexing: key along split axis is not 1D
            broadcasted_indexing = (
                key_shapes[original_split] is not None and len(key_shapes[original_split]) > 1
            )
            if broadcasted_indexing:
                broadcast_shape = key_shapes[original_split]
                key = list(key)
                key[original_split] = key[original_split].flatten()
                key = tuple(key)
                send_axis = original_split
            else:
                send_axis = output_split
            # print("RANK, RETURN_1D, broadcasted_indexing = ", self.comm.rank, return_1d, broadcasted_indexing)

        # send and receive "request key" info on what data element to ship where
        recv_counts = torch.zeros((self.comm.size, 1), dtype=torch.int64, device=self.larray.device)

        # construct empty tensor that we'll append to later
        if return_1d:
            request_key_shape = (0, self.ndim)
        else:
            request_key_shape = (0, 1)

        outgoing_request_key = torch.empty(
            tuple(request_key_shape), dtype=torch.int64, device=self.larray.device
        )
        outgoing_request_key_counts = torch.zeros(
            (self.comm.size,), dtype=torch.int64, device=self.larray.device
        )

        # process-local: calculate which/how many elements will be received from what process
        if split_key_is_ordered == -1:
            # key is sorted in descending order (i.e. slicing w/ negative step):
            # shrink selection of active processes
            if key[original_split].numel() > 0:
                key_edges = torch.cat(
                    (key[original_split][-1].reshape(-1), key[original_split][0].reshape(-1)), dim=0
                ).unique()
                displs = torch.tensor(displs, device=self.larray.device)
                _, inverse, counts = torch.cat((displs, key_edges), dim=0).unique(
                    sorted=True, return_inverse=True, return_counts=True
                )
                if key_edges.numel() == 2:
                    correction = counts[inverse[-2]] % 2
                    start_rank = inverse[-2] - correction
                    correction += counts[inverse[-1]] % 2
                    end_rank = inverse[-1] - correction + 1
                elif key_edges.numel() == 1:
                    correction = counts[inverse[-1]] % 2
                    start_rank = inverse[-1] - correction
                    end_rank = start_rank + 1
            else:
                start_rank = 0
                end_rank = 0
        else:
            start_rank = 0
            end_rank = self.comm.size
        all_local_indexing = torch.ones(
            (self.comm.size,), dtype=torch.bool, device=self.larray.device
        )
        all_local_indexing[start_rank:end_rank] = False
        for i in range(start_rank, end_rank):
            try:
                cond1 = key >= displs[i]
                if i != self.comm.size - 1:
                    cond2 = key < displs[i + 1]
                else:
                    # cond2 is always true
                    cond2 = torch.ones((key.shape[0],), dtype=torch.bool, device=self.larray.device)
            except TypeError:
                cond1 = key[original_split] >= displs[i]
                if i != self.comm.size - 1:
                    cond2 = key[original_split] < displs[i + 1]
                else:
                    # cond2 is always true
                    cond2 = torch.ones(
                        (key[original_split].shape[0],), dtype=torch.bool, device=self.larray.device
                    )
            if return_1d:
                # advanced indexing returning 1D array
                if isinstance(key, torch.Tensor):
                    selection = key[cond1 & cond2]
                    recv_counts[i, :] = selection.shape[0]
                    if i == self.comm.rank:
                        all_local_indexing[i] = selection.shape[0] == key.shape[0]
                    selection.unsqueeze_(dim=1)
                else:
                    # key is tuple of torch tensors
                    selection = list(k[cond1 & cond2] for k in key)
                    recv_counts[i, :] = selection[0].shape[0]
                    if i == self.comm.rank:
                        all_local_indexing[i] = selection[0].shape[0] == key[0].shape[0]
                    selection = torch.stack(selection, dim=1)
            else:
                selection = key[original_split][cond1 & cond2]
                recv_counts[i, :] = selection.shape[0]
                if i == self.comm.rank:
                    all_local_indexing[i] = selection.shape[0] == key[original_split].shape[0]
                selection.unsqueeze_(dim=1)
            outgoing_request_key = torch.cat((outgoing_request_key, selection), dim=0)
        all_local_indexing = factories.array(
            all_local_indexing, is_split=0, device=self.device, copy=False
        )
        if all_local_indexing.all().item():
            # TODO: if advanced indexing, indexed array must be a copy. Probably addressed by torch
            if broadcasted_indexing:
                key[original_split] = key[original_split].reshape(broadcast_shape)
            indexed_arr = self.larray[key]
            # transpose array back if needed
            self = self.transpose(backwards_transpose_axes)
            return factories.array(
                indexed_arr, is_split=output_split, device=self.device, copy=False
            )

        print("RECV_COUNTS = ", recv_counts)
        # share recv_counts among all processes
        comm_matrix = torch.empty(
            (self.comm.size, self.comm.size), dtype=recv_counts.dtype, device=recv_counts.device
        )
        self.comm.Allgather(recv_counts, comm_matrix)
        print("DEBUGGING: comm_matrix = ", comm_matrix, comm_matrix.shape)

        outgoing_request_key_counts = comm_matrix[self.comm.rank]
        outgoing_request_key_displs = torch.cat(
            (
                torch.zeros(
                    (1,),
                    dtype=outgoing_request_key_counts.dtype,
                    device=outgoing_request_key_counts.device,
                ),
                outgoing_request_key_counts,
            ),
            dim=0,
        ).cumsum(dim=0)[:-1]
        incoming_request_key_counts = comm_matrix[:, self.comm.rank]
        incoming_request_key_displs = torch.cat(
            (
                torch.zeros(
                    (1,),
                    dtype=outgoing_request_key_counts.dtype,
                    device=outgoing_request_key_counts.device,
                ),
                incoming_request_key_counts,
            ),
            dim=0,
        ).cumsum(dim=0)[:-1]

        if return_1d:
            incoming_request_key = torch.empty(
                (incoming_request_key_counts.sum(), self.ndim),
                dtype=outgoing_request_key_counts.dtype,
                device=outgoing_request_key_counts.device,
            )
        else:
            incoming_request_key = torch.empty(
                (incoming_request_key_counts.sum(), 1),
                dtype=outgoing_request_key_counts.dtype,
                device=outgoing_request_key_counts.device,
            )
        # send and receive request keys
        self.comm.Alltoallv(
            (
                outgoing_request_key,
                outgoing_request_key_counts.tolist(),
                outgoing_request_key_displs.tolist(),
            ),
            (
                incoming_request_key,
                incoming_request_key_counts.tolist(),
                incoming_request_key_displs.tolist(),
            ),
        )
        print("DEBUGGING:incoming_request_key = ", incoming_request_key)
        if return_1d:
            incoming_request_key = list(incoming_request_key[:, d] for d in range(self.ndim))
            incoming_request_key[original_split] -= displs[self.comm.rank]
        else:
            incoming_request_key -= displs[self.comm.rank]
            incoming_request_key = (
                key[:original_split]
                + (incoming_request_key.squeeze_(1),)
                + key[original_split + 1 :]
            )

        print("AFTER: incoming_request_key = ", incoming_request_key)
        print("original_split = ", original_split)
        # calculate shape of local recv buffer
        output_lshape = list(output_shape)
        if getattr(key, "ndim", 0) == 1:
            output_lshape[output_split] = key.shape[0]
        else:
            if broadcasted_indexing:
                output_lshape = (
                    output_lshape[:original_split]
                    + [torch.prod(torch.tensor(broadcast_shape, device=self.larray.device)).item()]
                    + output_lshape[output_split + 1 :]
                )
            else:
                output_lshape[output_split] = key[original_split].shape[0]
        # allocate recv buffer
        recv_buf = torch.empty(
            tuple(output_lshape), dtype=self.larray.dtype, device=self.larray.device
        )

        # index local data into send_buf.
        send_empty = sum(
            list(isinstance(k, torch.Tensor) and k.numel() == 0 for k in incoming_request_key)
        )  # incoming_request_key.count([])
        if send_empty:
            # Edge case 1. empty slice along split axis: send_buf is 0-element tensor
            empty_shape = list(output_shape)
            empty_shape[output_split] = 0
            send_buf = torch.empty(empty_shape, dtype=self.larray.dtype, device=self.larray.device)
        else:
            send_buf = self.larray[incoming_request_key]
            # Edge case 2. local single-element indexing results into local loss of split axis
            if send_buf.ndim < len(output_lshape):
                all_keys_scalar = sum(
                    list(
                        np.isscalar(k) or k.numel() == 1 and getattr(k, "ndim", 2) < 2
                        for k in incoming_request_key
                    )
                ) == len(incoming_request_key)
                if not all_keys_scalar:
                    send_buf = send_buf.unsqueeze_(dim=output_split)

        print("OUTPUT_SHAPE = ", output_shape)
        print("OUTPUT_SPLIT = ", output_split)
        print("SEND_BUF SHAPE = ", send_buf.shape)

        # output_lshape = list(output_shape)
        # if getattr(key, "ndim", 0) == 1:
        #     output_lshape[output_split] = key.shape[0]
        # else:
        #     if broadcasted_indexing:
        #         output_lshape = (
        #             output_lshape[:original_split]
        #             + [torch.prod(torch.tensor(broadcast_shape, device=send_buf.device)).item()]
        #             + output_lshape[output_split + 1 :]
        #         )
        #     else:
        #         output_lshape[output_split] = key[original_split].shape[0]
        # recv_buf = torch.empty(
        #     tuple(output_lshape), dtype=self.larray.dtype, device=self.larray.device
        # )
        recv_counts = torch.squeeze(recv_counts, dim=1).tolist()
        recv_displs = outgoing_request_key_displs.tolist()
        send_counts = incoming_request_key_counts.tolist()
        send_displs = incoming_request_key_displs.tolist()
        print("DEBUGGING: send_buf recv_buf shape= ", send_buf.shape, recv_buf.shape)
        print("DEBUGGING: send_counts recv_counts = ", send_counts, recv_counts)
        print("DEBUGGING: send_displs recv_displs = ", send_displs, recv_displs)
        print("DEBUGGING: output_split = ", output_split)
        self.comm.Alltoallv(
            (send_buf, send_counts, send_displs),
            (recv_buf, recv_counts, recv_displs),
            send_axis=send_axis,
        )
        # transpose original array back if needed, all further indexing on recv_buf
        self = self.transpose(backwards_transpose_axes)

        # reorganize incoming counts according to original key order along split axis
        if return_1d:
            if isinstance(key, tuple):
                key = torch.stack(key, dim=1)
            _, key_inverse = key.unique(dim=0, sorted=True, return_inverse=True)
            # if _.shape == key.shape:
            _, ork_inverse = outgoing_request_key.unique(dim=0, sorted=True, return_inverse=True)
            map = ork_inverse.argsort(stable=True)[
                key_inverse.argsort(stable=True).argsort(stable=True)
            ]
            indexed_arr = recv_buf[map]
            return factories.array(indexed_arr, is_split=output_split, copy=False)

        outgoing_request_key = outgoing_request_key.squeeze_(1)
        # incoming elements likely already stacked in ascending or descending order
        # TODO: is this check really worth it? blanket argsort solution below might be ok
        if (key[original_split] == outgoing_request_key).all():
            return factories.array(recv_buf, is_split=output_split, copy=False)
        if (key[original_split] == outgoing_request_key.flip(dims=(0,))).all():
            return factories.array(
                recv_buf.flip(dims=(output_split,)), is_split=output_split, copy=False
            )

        map = [slice(None)] * recv_buf.ndim
        print("DEBUGGING: outgoing_request_key = ", outgoing_request_key)
        print("DEBUGGING: key[original_split] = ", key[original_split])
        if broadcasted_indexing:
            map[original_split] = outgoing_request_key.argsort(stable=True)[
                key[original_split].argsort(stable=True).argsort(stable=True)
            ]
            map[original_split] = map[original_split].reshape(broadcast_shape)
        else:
            map[output_split] = outgoing_request_key.argsort(stable=True)[
                key[original_split].argsort(stable=True).argsort(stable=True)
            ]
        indexed_arr = recv_buf[map]
        return factories.array(indexed_arr, is_split=output_split, copy=False)

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
        try:
            len = self.shape[0]
            return len
        except IndexError:
            raise TypeError("len() of unsized DNDarray")

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

        def __broadcast_value(
            arr: DNDarray,
            key: Union[int, Tuple[int, ...], slice],
            value: DNDarray,
            **kwargs,
        ):
            """
            Broadcasts the assignment DNDarray `value` to the shape of the indexed array `arr[key]` if necessary.
            """
            is_scalar = (
                np.isscalar(value)
                or getattr(value, "ndim", 1) == 0
                or (value.shape == (1,) and value.split is None)
            )
            if is_scalar:
                # no need to broadcast
                return value, is_scalar
            # need information on indexed array
            output_shape = kwargs.get("output_shape", None)
            if output_shape is not None:
                indexed_dims = len(output_shape)
            else:
                if isinstance(key, (int, tuple)):
                    # direct indexing, output_shape has not been calculated
                    # use proxy to avoid MPI communication and limit memory usage
                    indexed_proxy = arr.__torch_proxy__()[key]
                    indexed_dims = indexed_proxy.ndim
                    output_shape = tuple(indexed_proxy.shape)
                else:
                    raise RuntimeError(
                        "Not enough information to broadcast value to indexed array, please provide `output_shape`"
                    )
            value_shape = value.shape
            # check if value needs to be broadcasted
            if value_shape != output_shape:
                # assess whether the shapes are compatible, starting from the trailing dimension
                for i in range(1, min(len(value_shape), len(output_shape)) + 1):
                    if i == 1:
                        if value_shape[-i] != output_shape[-i]:
                            # shapes are not compatible, raise error
                            raise ValueError(
                                f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                            )
                    else:
                        if value_shape[-i] != output_shape[-i] and (
                            not value_shape[-i] == 1 or not output_shape[-i] == 1
                        ):
                            # shapes are not compatible, raise error
                            raise ValueError(
                                f"could not broadcast input from shape {value_shape} into shape {output_shape}"
                            )
                # value has more dimensions than indexed array
                if value.ndim > indexed_dims:
                    # check if all dimensions except the indexed ones are singletons
                    all_singletons = value.shape[: value.ndim - indexed_dims] == (1,) * (
                        value.ndim - indexed_dims
                    )
                    if not all_singletons:
                        raise ValueError(
                            f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                        )
                    # squeeze out singleton dimensions
                    value = value.squeeze(tuple(range(value.ndim - indexed_dims)))
                else:
                    while value.ndim < indexed_dims:
                        # broadcasting
                        # expand missing dimensions to align split axis
                        value = value.expand_dims(0)
                        try:
                            value_shape = tuple(torch.broadcast_shapes(value.shape, output_shape))
                        except RuntimeError:
                            raise ValueError(
                                f"could not broadcast input array from shape {value_shape} into shape {output_shape}"
                            )
            return value, is_scalar

        def __set(
            arr: DNDarray,
            key: Union[int, Tuple[int, ...], List[int, ...]],
            value: Union[DNDarray, torch.Tensor, np.ndarray, float, int, list, tuple],
        ):
            """
            Setter for not advanced indexing, i.e. when arr[key] is an in-place view of arr.
            """
            # # need information on indexed array, use proxy to limit memory usage
            # subarray = arr.__torch_proxy__()[key]
            # subarray_shape, subarray_ndim = tuple(subarray.shape), subarray.ndim
            # while value.ndim < subarray_ndim:  # broadcasting
            #     value = value.expand_dims(0)
            #     try:
            #         value_shape = tuple(torch.broadcast_shapes(value_shape, subarray_shape))
            #     except RuntimeError:
            #         raise ValueError(
            #             f"could not broadcast input array from shape {value.shape} into shape {arr.shape}"
            #         )
            # # TODO: take this out of this function
            # sanitation.sanitize_out(subarray, value_shape, value.split, value.device, value.comm)
            #            arr.larray[None] = value.larray

            # make sure value is same datatype as arr
            process_is_inactive = arr.larray[key].numel() == 0
            if not process_is_inactive:
                # only assign values if key does not contain empty slices
                arr.larray[key] = value.larray.type(arr.dtype.torch_type())
            return

        # make sure `value` is a DNDarray
        if not isinstance(value, DNDarray):
            try:
                value = factories.array(
                    value, dtype=self.dtype, split=None, device=self.device, comm=self.comm
                )
            except TypeError:
                raise TypeError(f"Cannot assign object of type {type(value)} to DNDarray.")

        # workaround for Heat issue #1292. TODO: remove when issue is fixed
        if not isinstance(key, DNDarray):
            if key is None or key is ... or key is slice(None):
                # match dimensions
                value, _ = __broadcast_value(self, key, value)
                # make sure `self` and `value` distribution are aligned
                value = sanitation.sanitize_distribution(value, target=self)
                return __set(self, key, value)

        # single-element key
        scalar = np.isscalar(key) or getattr(key, "ndim", 1) == 0
        if scalar:
            key, root = self.__process_scalar_key(key, indexed_axis=0, return_local_indices=True)
            # match dimensions
            value, _ = __broadcast_value(self, key, value)
            # `root` will be None when the indexed axis is not the split axis, or when the
            # indexed axis is the split axis but the indexed element is not local
            if root is not None:
                if self.comm.rank == root:
                    # verify that `self[key]` and `value` distribution are aligned
                    # do not index `self` with `key` directly here, as this would MPI-broadcast to all ranks
                    indexed_proxy = self.__torch_proxy__()[key]
                    if indexed_proxy.names.count("split") != 0:
                        # distribution map of indexed subarray is the same as the lshape_map of the original array after losing the first dimension
                        indexed_lshape_map = self.lshape_map[:, 1:]
                        if value.lshape_map != indexed_lshape_map:
                            try:
                                value.redistribute_(target_map=indexed_lshape_map)
                            except ValueError:
                                raise ValueError(
                                    f"cannot assign value to indexed DNDarray because distribution schemes do not match: {value.lshape_map} vs. {indexed_lshape_map}"
                                )
                    __set(self, key, value)
            else:
                # `root` is None, i.e. the indexed element is local on each process
                # verify that `self[key]` and `value` distribution are aligned
                value = sanitation.sanitize_distribution(value, target=self[key])
                __set(self, key, value)
            return

        # multi-element key, incl. slicing and striding, ordered and non-ordered advanced indexing
        # # store original key for later use
        # try:
        #     original_key = key.copy()
        # except AttributeError:
        #     try:
        #         original_key = key.clone()
        #     except AttributeError:
        #         original_key = key

        (
            self,
            key,
            output_shape,
            output_split,
            split_key_is_ordered,
            out_is_balanced,
            root,
            backwards_transpose_axes,
        ) = self.__process_key(key, return_local_indices=True, op="set")

        # match dimensions
        value, value_is_scalar = __broadcast_value(self, key, value, output_shape=output_shape)

        if split_key_is_ordered == 1:
            # key all local
            if root is not None:
                # single-element assignment along split axis, only one active process
                if self.comm.rank == root:
                    self.larray[key] = value.larray.type(self.dtype.torch_type())
            else:
                # indexed elements are process-local
                if self.is_distributed() and not value_is_scalar and not value.is_distributed():
                    # work with distributed `value`
                    value = factories.array(
                        value.larray,
                        dtype=value.dtype,
                        split=output_split,
                        device=self.device,
                        comm=self.comm,
                    )
                    # verify that `self[key]` and `value` distribution are aligned
                    target_shape = torch.tensor(
                        tuple(self.larray[key].shape), device=self.device.torch_device
                    )
                    target_map = torch.zeros(
                        (self.comm.size, len(target_shape)),
                        dtype=torch.int64,
                        device=self.device.torch_device,
                    )
                    self.comm.Allgather(target_shape, target_map)
                    value.redistribute_(target_map=target_map)
                process_is_inactive = sum(
                    list(isinstance(k, torch.Tensor) and k.numel() == 0 for k in key)
                )
                if not process_is_inactive:
                    # only assign values if key does not contain empty slices
                    __set(self, key, value)
            self = self.transpose(backwards_transpose_axes)
            return

        if split_key_is_ordered == -1:
            # key along split axis is in descending order, i.e. slice with negative step
            if self.is_distributed():
                # flip value, match value distribution to key's
                # NB: `value.ndim` might be smaller than `self.ndim`, hence  `value.split` nominally different from `self.split`
                flipped_value = manipulations.flip(value, axis=output_split)
                split_key = factories.array(
                    key[self.split], is_split=0, device=self.device, comm=self.comm
                )
                if not flipped_value.is_distributed():
                    # work with distributed `flipped_value`
                    flipped_value = factories.array(
                        flipped_value.larray,
                        dtype=flipped_value.dtype,
                        split=output_split,
                        device=self.device,
                        comm=self.comm,
                    )
                # match `value` distribution to `self[key]` distribution
                target_map = flipped_value.lshape_map
                target_map[:, output_split] = split_key.lshape_map[:, 0]
                flipped_value.redistribute_(target_map=target_map)

                process_is_inactive = sum(
                    list(isinstance(k, torch.Tensor) and k.numel() == 0 for k in key)
                )
                if not process_is_inactive:
                    # only assign values if key does not contain empty slices
                    __set(self, key, flipped_value)
            else:
                # 1 process, no communication needed
                __set(self, key, value)
            self = self.transpose(backwards_transpose_axes)
            return

        # non-ordered key along split axis
        # indices are global

        # process-local indices

        # if advanced_indexing:
        #     raise Exception("Advanced indexing is not supported yet")

        # split = self.split
        # if not self.is_distributed() or key[split] == slice(None):
        #     return __set(self[key], value)

        # if isinstance(key[split], slice):
        #     return __set(self[key], value)

        # if np.isscalar(key[split]):
        #     key = list(key)
        #     idx = int(key[split])
        #     key[split] = slice(idx, idx + 1)
        #     return __set(self[tuple(key)], value)

        # key = getattr(key, "copy()", key)
        # try:
        #     if value.split != self.split:
        #         val_split = int(value.split)
        #         sp = self.split
        #         warnings.warn(
        #             f"\nvalue.split {val_split} not equal to this DNDarray's split:"
        #             f" {sp}. this may cause errors or unwanted behavior",
        #             category=RuntimeWarning,
        #         )
        # except (AttributeError, TypeError):
        #     pass

        # # NOTE: for whatever reason, there is an inplace op which interferes with the abstraction
        # # of this next block of code. this is shared with __getitem__. I attempted to abstract it
        # # in a standard way, but it was causing errors in the test suite. If someone else is
        # # motived to do this they are welcome to, but i have no time right now
        # # print(key)
        # if isinstance(key, DNDarray) and key.ndim == self.ndim:
        #     """if the key is a DNDarray and it has as many dimensions as self, then each of the
        #     entries in the 0th dim refer to a single element. To handle this, the key is split
        #     into the torch tensors for each dimension. This signals that advanced indexing is
        #     to be used."""
        #     key = manipulations.resplit(key)
        #     if key.larray.dtype in [torch.bool, torch.uint8]:
        #         key = indexing.nonzero(key)

        #     if key.ndim > 1:
        #         key = list(key.larray.split(1, dim=1))
        #         # key is now a list of tensors with dimensions (key.ndim, 1)
        #         # squeeze singleton dimension:
        #         key = [key[i].squeeze_(1) for i in range(len(key))]
        #     else:
        #         key = [key]
        # elif not isinstance(key, tuple):
        #     """this loop handles all other cases. DNDarrays which make it to here refer to
        #     advanced indexing slices, as do the torch tensors. Both DNDaarrys and torch.Tensors
        #     are cast into lists here by PyTorch. lists mean advanced indexing will be used"""
        #     h = [slice(None, None, None)] * self.ndim
        #     if isinstance(key, DNDarray):
        #         key = manipulations.resplit(key)
        #         if key.larray.dtype in [torch.bool, torch.uint8]:
        #             h[0] = torch.nonzero(key.larray).flatten()  # .tolist()
        #         else:
        #             h[0] = key.larray.tolist()
        #     elif isinstance(key, torch.Tensor):
        #         if key.dtype in [torch.bool, torch.uint8]:
        #             # (coquelin77) im not sure why this works without being a list...but it does...for now
        #             h[0] = torch.nonzero(key).flatten()  # .tolist()
        #         else:
        #             h[0] = key.tolist()
        #     else:
        #         h[0] = key
        #     key = list(h)

        # # key must be torch-proof
        # if isinstance(key, (list, tuple)):
        #     key = list(key)
        #     for i, k in enumerate(key):
        #         try:  # extract torch tensor
        #             k = manipulations.resplit(k)
        #             key[i] = k.larray
        #         except AttributeError:
        #             pass
        #         # remove bools from a torch tensor in favor of indexes
        #         try:
        #             if key[i].dtype in [torch.bool, torch.uint8]:
        #                 key[i] = torch.nonzero(key[i]).flatten()
        #         except (AttributeError, TypeError):
        #             pass

        # key = list(key)

        # # ellipsis stuff
        # key_classes = [type(n) for n in key]
        # # if any(isinstance(n, ellipsis) for n in key):
        # n_elips = key_classes.count(type(...))
        # if n_elips > 1:
        #     raise ValueError("key can only contain 1 ellipsis")
        # elif n_elips == 1:
        #     # get which item is the ellipsis
        #     ell_ind = key_classes.index(type(...))
        #     kst = key[:ell_ind]
        #     kend = key[ell_ind + 1 :]
        #     slices = [slice(None)] * (self.ndim - (len(kst) + len(kend)))
        #     key = kst + slices + kend
        # # ---------- end ellipsis stuff -------------

        # for c, k in enumerate(key):
        #     try:
        #         key[c] = k.item()
        #     except (AttributeError, ValueError, RuntimeError):
        #         pass

        # rank = self.comm.rank
        # if self.split is not None:
        #     counts, chunk_starts = self.counts_displs()
        # else:
        #     counts, chunk_starts = 0, [0] * self.comm.size
        # counts = torch.tensor(counts, device=self.device.torch_device)
        # chunk_starts = torch.tensor(chunk_starts, device=self.device.torch_device)
        # chunk_ends = chunk_starts + counts
        # chunk_start = chunk_starts[rank]
        # chunk_end = chunk_ends[rank]
        # # determine which elements are on the local process (if the key is a torch tensor)
        # try:
        #     # if isinstance(key[self.split], torch.Tensor):
        #     filter_key = torch.nonzero(
        #         (chunk_start <= key[self.split]) & (key[self.split] < chunk_end)
        #     )
        #     for k in range(len(key)):
        #         try:
        #             key[k] = key[k][filter_key].flatten()
        #         except TypeError:
        #             pass
        # except TypeError:  # this will happen if the key doesnt have that many
        #     pass

        # key = tuple(key)

        # if not self.is_distributed():
        #     return self.__setter(key, value)  # returns None

        # # raise RuntimeError("split axis of array and the target value are not equal") removed
        # # this will occur if the local shapes do not match
        # rank = self.comm.rank
        # ends = []
        # for pr in range(self.comm.size):
        #     _, _, e = self.comm.chunk(self.shape, self.split, rank=pr)
        #     ends.append(e[self.split].stop - e[self.split].start)
        # ends = torch.tensor(ends, device=self.device.torch_device)
        # chunk_ends = ends.cumsum(dim=0)
        # chunk_starts = torch.tensor([0] + chunk_ends.tolist(), device=self.device.torch_device)
        # _, _, chunk_slice = self.comm.chunk(self.shape, self.split)
        # chunk_start = chunk_slice[self.split].start
        # chunk_end = chunk_slice[self.split].stop

        # self_proxy = self.__torch_proxy__()

        # # if the value is a DNDarray, the divisions need to be balanced:
        # #   this means that we need to know how much data is where for both DNDarrays
        # #   if the value data is not in the right place, then it will need to be moved

        # if isinstance(key[self.split], slice):
        #     key = list(key)
        #     key_start = key[self.split].start if key[self.split].start is not None else 0
        #     key_stop = (
        #         key[self.split].stop
        #         if key[self.split].stop is not None
        #         else self.gshape[self.split]
        #     )
        #     if key_stop < 0:
        #         key_stop = self.gshape[self.split] + key[self.split].stop
        #     key_step = key[self.split].step
        #     og_key_start = key_start
        #     st_pr = torch.where(key_start < chunk_ends)[0]
        #     st_pr = st_pr[0] if len(st_pr) > 0 else self.comm.size
        #     sp_pr = torch.where(key_stop >= chunk_starts)[0]
        #     sp_pr = sp_pr[-1] if len(sp_pr) > 0 else 0
        #     actives = list(range(st_pr, sp_pr + 1))

        #     if (
        #         isinstance(value, type(self))
        #         and value.split is not None
        #         and value.shape[self.split] != self.shape[self.split]
        #     ):
        #         # setting elements in self with a DNDarray which is not the same size in the
        #         # split dimension
        #         local_keys = []
        #         # below is used if the target needs to be reshaped
        #         target_reshape_map = torch.zeros(
        #             (self.comm.size, self.ndim), dtype=torch.int64, device=self.device.torch_device
        #         )
        #         for r in range(self.comm.size):
        #             if r not in actives:
        #                 loc_key = key.copy()
        #                 loc_key[self.split] = slice(0, 0, 0)
        #             else:
        #                 key_start_l = 0 if r != actives[0] else key_start - chunk_starts[r]
        #                 key_stop_l = ends[r] if r != actives[-1] else key_stop - chunk_starts[r]
        #                 key_start_l, key_stop_l = self.__xitem_get_key_start_stop(
        #                     r, actives, key_start_l, key_stop_l, key_step, chunk_ends, og_key_start
        #                 )
        #                 loc_key = key.copy()
        #                 loc_key[self.split] = slice(key_start_l, key_stop_l, key_step)

        #                 gout_full = torch.tensor(
        #                     self_proxy[loc_key].shape, device=self.device.torch_device
        #                 )
        #                 target_reshape_map[r] = gout_full
        #             local_keys.append(loc_key)

        #         key = local_keys[rank]
        #         value = value.redistribute(target_map=target_reshape_map)

        #         if rank not in actives:
        #             return  # non-active ranks can exit here

        #         chunk_starts_v = target_reshape_map[:, self.split]
        #         value_slice = [slice(None, None, None)] * value.ndim
        #         step2 = key_step if key_step is not None else 1
        #         key_start = (chunk_starts_v[rank] - og_key_start).item()

        #         key_start = max(key_start, 0)
        #         key_stop = key_start + key_stop
        #         slice_loc = min(self.split, value.ndim - 1)
        #         value_slice[slice_loc] = slice(
        #             key_start, math.ceil(torch.true_divide(key_stop, step2)), 1
        #         )

        #         self.__setter(tuple(key), value.larray)
        #         return

        #     # if rank in actives:
        #     if rank not in actives:
        #         return  # non-active ranks can exit here
        #     key_start = 0 if rank != actives[0] else key_start - chunk_starts[rank]
        #     key_stop = ends[rank] if rank != actives[-1] else key_stop - chunk_starts[rank]
        #     key_start, key_stop = self.__xitem_get_key_start_stop(
        #         rank, actives, key_start, key_stop, key_step, chunk_ends, og_key_start
        #     )
        #     key[self.split] = slice(key_start, key_stop, key_step)

        #     # todo: need to slice the values to be the right size...
        #     if isinstance(value, (torch.Tensor, type(self))):
        #         # if its a torch tensor, it is assumed to exist on all processes
        #         value_slice = [slice(None, None, None)] * value.ndim
        #         step2 = key_step if key_step is not None else 1
        #         key_start = (chunk_starts[rank] - og_key_start).item()
        #         key_start = max(key_start, 0)
        #         key_stop = key_start + key_stop
        #         slice_loc = min(self.split, value.ndim - 1)
        #         value_slice[slice_loc] = slice(
        #             key_start, math.ceil(torch.true_divide(key_stop, step2)), 1
        #         )
        #         self.__setter(tuple(key), value[tuple(value_slice)])
        #     else:
        #         self.__setter(tuple(key), value)
        # elif isinstance(key[self.split], (torch.Tensor, list)):
        #     key = list(key)
        #     key[self.split] -= chunk_start
        #     if len(key[self.split]) != 0:
        #         self.__setter(tuple(key), value)

        # elif key[self.split] in range(chunk_start, chunk_end):
        #     key = list(key)
        #     key[self.split] = key[self.split] - chunk_start
        #     self.__setter(tuple(key), value)

        # elif key[self.split] < 0:
        #     key = list(key)
        #     if self.gshape[self.split] + key[self.split] in range(chunk_start, chunk_end):
        #         key[self.split] = key[self.split] + self.shape[self.split] - chunk_start
        #         self.__setter(tuple(key), value)

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
        Return a 1-element `torch.Tensor` strided as the global `self` shape. The split axis of the initial DNDarray is stored in the `names` attribute of the returned tensor.
        Used internally to lower memory footprint of sanitation.
        """
        names = [None] * self.ndim
        if self.split is not None:
            names[self.split] = "split"
        return (
            torch.ones((1,), dtype=torch.int8, device=self.larray.device)
            .as_strided(self.gshape, [0] * self.ndim)
            .refine_names(*names)
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
import types
from .types import datatype, canonical_heat_type, bool, uint8
