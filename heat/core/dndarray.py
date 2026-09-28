"""Provides Heat's core data structure, the DNDarray, a distributed n-dimensional array"""

from __future__ import annotations
import math

import numpy as np
import torch
import warnings

from mpi4py import MPI
from pathlib import Path
from enum import Enum
from typing import Any, Union, TypeVar

warnings.simplefilter("always", ResourceWarning)

# NOTE: heat module imports need to be placed at the very end of the file to avoid cyclic dependencies
__all__ = ["DNDarray"]

Communication = TypeVar("Communication")

# Type aliases
Index = Union[int, slice, type(...), None, torch.Tensor, np.ndarray, "DNDarray"]
Key = Union[Index, tuple[Index, ...], list[Index]]


class DNDarray:
    """
    Distributed N-Dimensional array. The core element of Heat. It is composed of
    PyTorch tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor
        Local array elements
    gshape : tuple[int,...]
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
        gshape: tuple[int, ...],
        dtype: datatype,
        split: int | None,
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
        self.__balanced: bool = balanced
        self.__ishalo = False
        self.__halo_next: torch.Tensor | None = None
        self.__halo_prev: torch.Tensor | None = None
        self.__partitions_dict__ = None
        self.__lshape_map = None
        self.__counts_displs = None

        # check for inconsistencies between torch and heat devices
        assert str(array.device) == device.torch_device

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def array_with_halos(self) -> torch.Tensor:
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next``/``self.halo_prev``
        in case they are not already stored. If ``halo_size`` differs from the size of already stored halos,
        the are overwritten.
        """
        return self.__cat_halo()

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
    def gshape(self) -> tuple:
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
    def imag(self) -> DNDarray:
        """
        Return the imaginary part of the ``DNDarray``.
        """
        return complex_math.imag(self)

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
            self.__lshape_map = None
            self.__counts_displs = None
        self.__array = array

    @property
    def lloc(self):
        """Deprecated function for local indexing. Use `DNDarray.larray` for local indexing instead"""
        # TODO: Remove this entirely by heat v2.5
        raise Exception(
            "`DNDarray.lloc` is deprecated. Use `DNDarray.larray` for local indexing instead."
        )

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
    def lshape(self) -> tuple[int]:
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
        Return a dictionary containing information useful for working with the partitioned
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
    def real(self) -> DNDarray:
        """
        Return the real part of the ``DNDarray``.
        """
        return complex_math.real(self)

    @property
    def shape(self) -> tuple[int]:
        """
        Returns the shape of the ``DNDarray`` as a whole
        """
        return self.__gshape

    @property
    def size(self) -> int:
        """
        Number of total elements of the ``DNDarray``
        """
        if self.larray.is_mps:
            # MPS does not support double precision
            size = torch.prod(
                torch.tensor(self.gshape, dtype=torch.float32, device=self.device.torch_device)
            )
        else:
            size = torch.prod(
                torch.tensor(self.gshape, dtype=torch.float64, device=self.device.torch_device)
            )
        return size.long().item()

    @property
    def split(self) -> int | None:
        """
        Returns the axis on which the ``DNDarray`` is split
        """
        return self.__split

    @property
    def stride(self) -> tuple[int]:
        """
        Returns the steps in each dimension when traversing a ``DNDarray``. torch-like usage: ``self.stride()``
        """
        return self.__array.stride

    @property
    def strides(self) -> tuple[int]:
        """
        Returns bytes to step in each dimension when traversing a ``DNDarray``. numpy-like usage: ``self.strides()``
        """
        steps = list(self.__array.stride())
        try:
            itemsize = self.__array.untyped_storage().element_size()
        except AttributeError:
            itemsize = self.__array.storage().element_size()
        strides = tuple(step * itemsize for step in steps)
        return strides

    # ------------------------------------------------------------------
    # Public methods / protocols
    # ------------------------------------------------------------------

    def __array__(self) -> np.ndarray:
        """
        Returns a view of the process-local slice of the :class:`DNDarray` as a numpy ndarray, if the ``DNDarray`` resides on CPU. Otherwise, it returns a copy, on CPU, of the process-local slice of ``DNDarray`` as numpy ndarray.
        """
        return self.larray.cpu().__array__()

    def __array_function__(self, func, types, args, kwargs):
        """
        Augments NumPy's functions.
        """
        import heat

        try:
            ht_func = getattr(heat, func.__name__)
        except AttributeError:
            return NotImplemented
        return ht_func(*args, **kwargs)

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        """
        Returns an object that has all the array API functions on it.

        Parameters
        ----------
        api_version : Optional[str]
            string representing the version of the array API specification to
            be returned, in ``'YYYY.MM'`` form. If it is ``None`` (default), it
            returns the namespace corresponding to latest version of the
            array API specification.
        """
        if api_version is not None and api_version != "2025.12":
            raise ValueError(f"Unrecognized array API version: {api_version}")
        import heat

        return heat

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override NumPy's universal functions.
        """
        import heat

        # TODO support ufunc method variants
        if method == "__call__":
            try:
                func = getattr(heat, ufunc.__name__)
            except AttributeError:
                return NotImplemented
            return func(*inputs, **kwargs)
        else:
            return NotImplemented

    def astype(self, dtype, copy=True, device: Device = None) -> DNDarray:
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
        device: ht.Device, optional
            The device on which to place the array. If ``None``, keep device. Default: None.
        """
        dtype = canonical_heat_type(dtype)
        device = self.__device if device is None else devices.sanitize_device(device)
        if self.__array.is_mps:
            if dtype == types.float64:
                # print warning
                warnings.warn(
                    "MPS does not support float64. Casting to float32 instead.",
                    ResourceWarning,
                )
                dtype = types.float32
            elif dtype == types.complex128:
                # print warning
                warnings.warn(
                    "MPS does not support complex128. Casting to complex64 instead.",
                    ResourceWarning,
                )
                dtype = types.complex64
        casted_array = self.__array.to(
            device=device.torch_device, dtype=dtype.torch_type(), copy=copy
        )
        if copy:
            return DNDarray(
                casted_array,
                gshape=self.shape,
                dtype=dtype,
                split=self.split,
                device=device,
                comm=self.comm,
                balanced=self.balanced,
            )

        self.__array = casted_array
        self.__dtype = dtype
        self.__device = device

        return self

    def balance_(self) -> None:
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
        if not self.is_distributed():
            self.__balanced = True
        if self.is_balanced(force_check=True):
            return
        self.redistribute_()

    def __bool__(self) -> bool:
        """
        Boolean scalar casting.
        """
        return self.__cast(bool)

    def collect_(self, target_rank: int = 0) -> None:
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

    def counts_displs(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Returns actual counts (number of items per process) and displacements (offsets) of the DNDarray.
        Does not assume load balance.
        """
        if self.split is not None:
            if self.__counts_displs is not None:
                return self.__counts_displs

            if self.__lshape_map is None:
                self.create_lshape_map()

            counts = self.__lshape_map[:, self.split]
            displs = [0] + torch.cumsum(counts, dim=0)[:-1].tolist()
            res = (tuple(counts.tolist()), tuple(displs))
            self.__counts_displs = res
            return res

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
        if not self.is_distributed():
            lshape_map[:] = torch.tensor(self.gshape, device=self.device.torch_device)
            self.__lshape_map = lshape_map
            return lshape_map.clone()
        elif self.is_balanced(force_check=True):
            for i in range(self.comm.size):
                _, lshape, _ = self.comm.chunk(self.gshape, self.split, rank=i)
                lshape_map[i, :] = torch.tensor(lshape, device=self.device.torch_device)
        else:
            lshape_map[self.comm.rank, :] = torch.tensor(
                self.lshape, device=self.device.torch_device
            )
            self.comm.Allreduce(MPI.IN_PLACE, lshape_map, MPI.SUM)

        self.__lshape_map = lshape_map
        self.__counts_displs = None
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

        z = torch.tensor([0], device=self.device.torch_device, dtype=torch.int64)

        if self.split is not None:
            starts = torch.cat((z, torch.cumsum(lshape_map[:, self.split], dim=0)[:-1]), dim=0)
            lcls[self.split] = self.comm.rank
            part_tiling[self.split] = self.comm.size
            start_idx_map[:, self.split] = starts
        else:
            start_idx_map[:] = 0

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

    def __dlpack__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Exports the undistributed array for consumption by ``from_dlpack()`` as a DLPack capsule.
        Any positional arguments ``*args`` and keyword arguments ``**kwargs`` are directly forwarded to torch ``__dlpack__``.

        Note
        ----
        See `Array API <https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.array.__dlpack__.html>`_ for details and the function signature as implemented by torch.

        Raises
        ------
        BufferError
            if the DNDarray is distributed, as this is not supported by DLPack.
        """
        if self.is_distributed():
            raise BufferError("DLPack export works for undistributed arrays only.")

        return self.larray.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> tuple[Enum, int]:
        """
        Returns device type and device ID in DLPack format. Meant for use
        within ``from_dlpack()``.
        """
        return self.larray.__dlpack_device__()

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

        if self.is_distributed():
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

    def __float__(self) -> float:
        """
        Float scalar casting.

        See Also
        --------
        :func:`~heat.core.manipulations.flatten`
        """
        return self.__cast(float)

    def get_halo(self, halo_size: int, prev: bool = True, next: bool = True):
        """
        Fetch halos of size ``halo_size`` from neighboring ranks and save them in ``self.halo_next/self.halo_prev``.

        Parameters
        ----------
        halo_size : int
            Size of the halo.
        prev : bool, optional
            If True, fetch the halo from the previous rank. Default: True.
        next : bool, optional
            If True, fetch the halo from the next rank. Default: True.
        """
        if not isinstance(halo_size, int):
            raise TypeError(
                f"halo_size needs to be of Python type integer, {type(halo_size)} given"
            )
        if halo_size < 0:
            raise ValueError(
                f"halo_size needs to be a non-negative Python integer, {halo_size} given"
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
            if prev:
                if rank != last_rank:
                    req_list.append(self.comm.Isend(a_next, next_rank))
                if rank != first_rank:
                    res_prev = torch.empty(
                        a_prev.size(), dtype=a_prev.dtype, device=self.device.torch_device
                    )
                    req_list.append(self.comm.Irecv(res_prev, source=prev_rank))

            if next:
                if rank != first_rank:
                    req_list.append(self.comm.Isend(a_prev, prev_rank))
                if rank != last_rank:
                    res_next = torch.empty(
                        a_next.size(), dtype=a_next.dtype, device=self.device.torch_device
                    )
                    req_list.append(self.comm.Irecv(res_next, source=next_rank))

            for req in req_list:
                req.Wait()

            self.__halo_next = res_next
            self.__halo_prev = res_prev
            self.__ishalo = True

    def __getitem__(self, key: Key) -> DNDarray:
        """
        Global getter function for DNDarrays.

        Returns a new DNDarray corresponding to the selection of values from the original DNDarray
        as specified by `key`. The `key` can be a variety of indexers, including integers, slices,
        lists, boolean masks, DNDarrays, ndarrays, torch tensors, and a combination thereof.

        The function determines the appropriate method to retrieve the requested data based on the
        type and structure of `key`, executing MPI communication if the indexing pattern requires
        data from multiple processes.

        Notes
        -----
        The returned DNDarray will have its shape, split, and balanced status determined according
        to the indexing operation performed. For more details on supported indexing behaviors, see
        the :doc:`indexing documentation <INDEXING>`.

        Parameters
        ----------
        key : array-like indexer
            Indices to get from the ``DNDarray``.

        Examples
        --------
        >>> a = ht.arange(10, split=0)
        (1/2) >>> tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        >>> a[1:6]
        (1/2) >>> tensor([1, 2, 3, 4], dtype=torch.int32)
        (2/2) >>> tensor([5], dtype=torch.int32)
        >>> a = ht.zeros((4, 5), split=0)
        (1/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        (2/2) >>> tensor([[0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
        >>> a[1:4, 1]
        (1/2) >>> tensor([0.])
        (2/2) >>> tensor([0., 0.])
        """
        if key is None:
            return self.expand_dims(0)
        if (
            key is ...
            or (isinstance(key, slice) and key == slice(None))
            or (isinstance(key, tuple) and key == ())
        ):
            return self

        # attempt early out for non-distributed arrays
        if not self.is_distributed():
            try:
                res_tensor = self.larray[_unwrap_local_key(key, device=self.device.torch_device)]
                return DNDarray(
                    res_tensor,
                    gshape=tuple(res_tensor.shape),
                    dtype=self.dtype,
                    split=None,
                    device=self.device,
                    comm=self.comm,
                    balanced=True,
                )
            except Exception:
                pass

        # key processing returns a ProcessedKey namedtuple
        self, processed_key = _resolve_indexing_state(
            self, key, return_local_indices=True, op="get"
        )

        # dispatch to appropriate getitem method
        op = processed_key.op_type

        if op == "scalar":
            return self.__getitem_scalar(processed_key)
        elif op == "distr_mask":
            return self.__getitem_mask(processed_key)
        elif op == "distributed":
            return self.__getitem_advanced_distributed(processed_key)
        elif op == "descending_slice":
            return self.__getitem_descending_slice_distributed(processed_key)
        elif op in ("local_mask", "local"):
            return self.__getitem_local(processed_key)

    if torch.cuda.device_count() > 0:

        def gpu(self) -> DNDarray:
            """
            Returns a copy of this object in GPU memory. If this object is already in GPU memory, then no copy is
            performed and the original object is returned.

            """
            self.__array = self.__array.cuda(devices.gpu.torch_device)
            self.__device = devices.gpu
            return self

    def __index__(self) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.
        """
        if not issubclass(self.dtype, integer):
            raise TypeError("only integer scalar arrays can be converted to a scalar index")
        return self.__cast(int)

    def __int__(self) -> int:
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
        ----------
        force_check : bool, optional
            If True, the balanced status of the ``DNDarray`` will be assessed via
            collective communication in any case.
        """
        if not self.is_distributed():
            self.__balanced = True
            return self.balanced

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

    def item(self):
        """
        Returns the only element of a 1-element :class:`DNDarray`.
        Mirror of the pytorch command by the same name. If size of ``DNDarray`` is >1 element, then a ``ValueError`` is
        raised (by pytorch)

        Examples
        --------
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

    def numpy(self) -> np.typing.NDArray[Any]:
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

    def ravel(self) -> DNDarray:
        """
        Flattens the ``DNDarray``.

        See Also
        --------
        :func:`~heat.core.manipulations.ravel`

        Examples
        --------
        >>> a = ht.ones((2, 3), split=0)
        >>> b = a.ravel()
        >>> a[0, 0] = 4
        >>> b
        DNDarray([4., 1., 1., 1., 1., 1.], dtype=ht.float32, device=cpu:0, split=0)
        """
        return manipulations.ravel(self)

    def redistribute_(
        self, lshape_map: torch.Tensor | None = None, target_map: torch.Tensor | None = None
    ) -> None:
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
        self.__counts_displs = None

    def __repr__(self) -> str:
        """
        Returns a printable representation of the passed DNDarray, targeting developers.
        """
        return printing.__repr__(self)

    def _repr_pretty_(self, p, cycle):
        """
        Pretty print for IPython.
        """
        if cycle:
            p.text(printing.__str__(self))
        else:
            p.text(printing.__str__(self))

    def resplit_(self, axis: int = None):
        """
        In-place option for resplitting a :class:`DNDarray`.

        Parameters
        ----------
        axis : int
            The new split axis, ``None`` denotes gathering, an int will set the new split axis

        Examples
        --------
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
        >>> a.lshape
        (0/2) (2, 5)
        (1/2) (2, 5)
        >>> ht.resplit_(a, None)
        >>> a.split
        None
        >>> a.lshape
        (0/2) (4, 5)
        (1/2) (4, 5)
        >>> a = ht.zeros(
        ...     (
        ...         4,
        ...         5,
        ...     ),
        ...     split=0,
        ... )
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

        self.__partitions_dict__ = None

        # early out for unchanged content
        if self.comm.size == 1:
            self.__split = axis
        if axis == self.split:
            return self

        if axis is None:
            gathered = torch.empty(
                self.shape, dtype=self.dtype.torch_type(), device=self.device.torch_device
            )
            counts, displs = self.counts_displs()
            self.comm.Allgatherv(self.__array, (gathered, counts, displs), recv_axis=self.split)
            self.__array = gathered
            self.__split = axis
            self.__lshape_map = None
            self.__counts_displs = None
            return self
        # tensor needs be split/sliced locally
        if self.split is None:
            _, _, slices = self.comm.chunk(self.shape, axis)
            temp = self.__array[slices]
            self.__array = torch.empty((1,), device=self.device.torch_device)
            # necessary to clear storage of local __array
            self.__array = temp.clone().detach()
            self.__split = axis
            self.__lshape_map = None
            self.__counts_displs = None
            return self

        arr_tiles = tiling.SplitTiles(self)
        new_tiles = tiling.SplitTiles(self)

        gshape = self.shape
        new_lshape = list(gshape)
        new_lshape[axis] = int(arr_tiles.tile_dimensions[axis][self.comm.rank].item())

        recv_buffer = torch.empty(
            tuple(new_lshape), dtype=self.dtype.torch_type(), device=self.device.torch_device
        )

        self._axis2axisResplit(
            self.larray, self.split, arr_tiles, recv_buffer, axis, new_tiles, self.comm
        )

        self.__array = recv_buffer
        self.__split = axis
        self.__lshape_map = None
        self.__counts_displs = None

        return self

    def __setitem__(
        self,
        key: Key,
        value: float | "DNDarray" | torch.Tensor,
    ):
        """
        Global item setter for DNDarrays.

        Assigns values to the specified positions in the ``DNDarray``. The `key` can be a variety
        of indexers, including integers, slices, lists, boolean masks, DNDarrays, ndarrays,
        torch tensors, or a combination thereof.

        If a distributed ``DNDarray`` is given as the `value` to be set, this function will
        automatically attempt to align its distribution scheme (split axis and local shapes)
        with the target indexed array via MPI communication. If the distributions cannot be
        safely aligned, a ``ValueError`` or ``RuntimeError`` is raised.

        Parameters
        ----------
        key : array-like indexer
            Index/indices to be set
        value: float | "DNDarray" | torch.Tensor
            Value to be set to the specified positions in the DNDarray (self)

        Notes
        -----
        For more details on supported indexing behaviors, see the :doc:`indexing documentation <INDEXING>`.

        Examples
        --------
        >>> a = ht.zeros((4, 5), split=0)
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
        if not self.is_distributed() and not (
            isinstance(value, DNDarray) and value.is_distributed()
        ):
            try:
                torch_key = _unwrap_local_key(key, device=self.device.torch_device)
                if isinstance(value, DNDarray):
                    rhs = value.larray.to(self.larray.dtype)
                elif isinstance(value, torch.Tensor):
                    rhs = value.to(self.larray.dtype)
                else:
                    rhs = value  # Python scalar / float / int

                if self.larray.is_cuda and torch.is_tensor(rhs):
                    torch_key, rhs = _resolve_duplicate_indices(torch_key, rhs, self.larray.shape)

                self.larray[torch_key] = rhs
                return
            except Exception:
                pass

        # bypass factories.array() for primitive types and single-element tensors to avoid unnecessary overhead
        value_is_primitive = isinstance(value, (int, float, complex, bool)) or (
            isinstance(value, torch.Tensor) and value.numel() == 1 and value.ndim == 0
        )
        if not value_is_primitive and not isinstance(value, DNDarray):
            value = factories.array(value)

        original_key = key
        original_split = self.split

        self, processed_key = _resolve_indexing_state(
            self, key, return_local_indices=True, op="set"
        )

        op = processed_key.op_type

        # match dimensions (except for distr_mask as it perfectly aligns)
        if op == "distr_mask":
            value_is_scalar = (
                np.isscalar(value)
                or getattr(value, "ndim", 1) == 0
                or (getattr(value, "shape", None) == (1,) and getattr(value, "split", 0) is None)
            )
        else:
            value, value_is_scalar = _broadcast_value(value, processed_key.output_shape)

        # dispatch to the appropriate setter
        if op == "distr_mask":
            self.__setitem_mask(processed_key, value, value_is_scalar)
        elif op == "scalar":
            self.__setitem_scalar(processed_key, value, value_is_scalar)
        elif op == "distributed":
            self.__setitem_advanced_distributed(
                processed_key, original_key, value, value_is_scalar, original_split=original_split
            )
        elif op == "descending_slice":
            self.__setitem_descending_slice_distributed(processed_key, value, value_is_scalar)
        elif op in ("local_mask", "local"):
            self.__setitem_local(processed_key, value, value_is_scalar)

    def __str__(self) -> str:
        """
        Computes a string representation of the passed ``DNDarray``.
        """
        return printing.__str__(self)

    def to_device(self, device: Device, /, *, stream: int | Any | None = None) -> DNDarray:
        """
        Copy the array from the device on which it currently resides to the specified ``device``.

        Parameters
        ----------
        device : Device
            A ``Device`` object.
        stream : Int or Any, optional
            Stream object to use during copy.
        """
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device.device_type == "cpu":
            return self.cpu()
        elif device.device_type == "gpu":
            return self.gpu()
        raise ValueError(f"Unsupported device {device!r}")

    def tolist(self, keepsplit: bool = False) -> list[int | float]:
        """
        Return a copy of the local array data as a (nested) Python list. For scalars, a standard Python number is returned.

        Parameters
        ----------
        keepsplit: bool
            Whether the list should be returned locally or globally.

        Examples
        --------
        >>> a = ht.array([[0, 1], [2, 3]])
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0, 1], [2, 3]], split=0)
        >>> a.tolist()
        [[0, 1], [2, 3]]

        >>> a = ht.array([[0, 1], [2, 3]], split=1)
        >>> a.tolist(keepsplit=True)
        (1/2) [[0], [2]]
        (2/2) [[1], [3]]
        """
        if not keepsplit:
            return self.resplit(axis=None).__array.tolist()

        return self.__array.tolist()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Supports PyTorch's dispatch mechanism.
        """
        import heat

        if kwargs is None:
            kwargs = {}
        try:
            ht_func = getattr(heat, func.__name__)
        except AttributeError:
            return NotImplemented
        return ht_func(*args, **kwargs)

    def __torch_proxy__(self) -> torch.Tensor:
        """
        Return a 1-element `torch.Tensor` strided as the global `self` shape.
        Used internally for sanitation purposes.
        """
        return torch.empty(self.gshape, device="meta")

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def __cast(self, cast_function) -> float | int:
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
            if not self.is_distributed():
                return cast_function(self.__array)

            is_empty = np.prod(self.__array.shape) == 0
            root = self.comm.allreduce(0 if is_empty else self.comm.rank, op=MPI.SUM)

            return self.comm.bcast(None if is_empty else cast_function(self.__array), root=root)

        raise TypeError("only size-1 arrays can be converted to Python scalars")

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

        return self.__array[tuple(ix)].clone()

    def __redistribute_shuffle(
        self,
        snd_pr: int | torch.Tensor,
        send_amt: int | torch.Tensor,
        rcv_pr: int | torch.Tensor,
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
            data = self.__array[tuple(send_slice)].clone()
            self.comm.Send(data, dest=rcv_pr, tag=685)
            self.__array = self.__array[tuple(keep_slice)]
        if rank == rcv_pr:
            shp = list(self.gshape)
            shp[self.split] = send_amt
            data = torch.zeros(shp, dtype=snd_dtype, device=self.device.torch_device)
            self.comm.Recv(data, source=snd_pr, tag=685)
            if snd_pr < rcv_pr:  # data passed from a lower rank (append to top)
                self.__array = torch.cat((data, self.__array), dim=self.split)
            if snd_pr > rcv_pr:  # data passed from a higher rank (append to bottom)
                self.__array = torch.cat((self.__array, data), dim=self.split)

    def __set(
        self,
        key: int | tuple[int, ...] | list[int],
        value: float | "DNDarray" | torch.Tensor,
    ):
        """
        Setter for not advanced indexing, i.e. when arr[key] is an in-place view of arr.
        """
        # only assign values if key does not contain empty slices
        if self.larray.numel() == 0:
            return
        if torch.is_tensor(key) and key.numel() == 0:
            return
        if key == slice(0, 0):
            return
        if isinstance(key, tuple) and any(
            (torch.is_tensor(k) and k.numel() == 0) or k == slice(0, 0) for k in key
        ):
            return

        rhs = value.larray.type(self.dtype.torch_type()) if hasattr(value, "larray") else value
        key_to_use = key

        # CUDA: make advanced indexing assignment deterministic for duplicate indices
        if self.larray.is_cuda:
            key_to_use, rhs = _resolve_duplicate_indices(key_to_use, rhs, self.larray.shape)

        self.larray[key_to_use] = rhs
        return

    def __prepare_unordered_comm(self, split_key_flat: torch.Tensor, displs: tuple) -> tuple:
        """
        Helper function for distributed unordered indexing.
        Determines destination ranks, sorts the key, and computes Alltoallv parameters.
        """
        displs_t = torch.tensor(displs, device=self.device.torch_device)

        # map global indices to destination ranks
        dest_ranks = torch.searchsorted(displs_t[1:], split_key_flat, right=True).to(torch.int64)

        # sort by destination rank to pack memory contiguously
        sort_idx = torch.argsort(dest_ranks)
        dest_ranks_sorted = dest_ranks[sort_idx]

        # calculate send_counts and send_displs
        send_counts = torch.bincount(dest_ranks_sorted, minlength=self.comm.size).to(torch.int64)
        send_displs = torch.zeros_like(send_counts)
        send_displs[1:] = torch.cumsum(send_counts, dim=0)[:-1]

        # collect and calculate recv_counts and recv_displs
        recv_counts = torch.empty_like(send_counts)
        self.comm.Alltoall(send_counts, recv_counts)
        recv_displs = torch.zeros_like(recv_counts)
        recv_displs[1:] = recv_counts.cumsum(0)[:-1]

        return (
            sort_idx,
            send_counts,
            send_displs,
            recv_counts,
            recv_displs,
        )

    def __getitem_scalar(self, p: ProcessedKey) -> DNDarray:
        """
        Handles single-element extraction. If the scalar index falls on the
        split axis, the extracted value is broadcasted from the
        root process to all others.
        """
        if p.root is not None:
            # Single-element indexing along split axis
            if self.comm.rank == p.root:
                indexed_arr = self.larray[p.key]
            else:
                indexed_arr = torch.zeros(
                    p.output_shape, dtype=self.larray.dtype, device=self.device.torch_device
                )
            self.comm.Bcast(indexed_arr, root=p.root)
        else:
            indexed_arr = self.larray[p.key]

        return DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=p.out_is_balanced,
        )

    def __getitem_local(self, p: ProcessedKey) -> "DNDarray":
        """
        Handles process-local indexing (including standard slices and local advanced indices) directly on local array partitions
        without MPI communication.
        """
        indexed_arr = self.larray[p.key]

        return DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=p.out_is_balanced,
        )

    def __getitem_descending_slice_distributed(self, p: ProcessedKey) -> DNDarray:
        """
        Handles negative step slicing along the split axis. This is a workaround as torch does not support negative-step slicing.
        """
        from .manipulations import flip

        # local indexing
        indexed_arr = self.larray[p.key]

        # wrap the reversed local chunks into an unbalanced DNDarray
        intermediate = DNDarray(
            indexed_arr,
            gshape=p.output_shape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=False,
        )

        # global flip to reflect the descending slice
        return flip(intermediate, axis=p.output_split)

    def __getitem_mask(self, p: ProcessedKey) -> "DNDarray":
        """
        Handles fast-path boolean masking. Applies the mask locally without
        requiring MPI communication during extraction, returning a flattened array
        distributed along the specified split axis.
        """
        # local masking, then wrap into DNDarray
        local_mask = p.key
        local_result = self.larray[local_mask]

        # calculate gshape
        local_count = local_result.shape[0]
        total_count = self.comm.allreduce(local_count, op=MPI.SUM)
        gshape = (total_count,) + local_result.shape[1:]

        return DNDarray(
            local_result,
            gshape=gshape,
            dtype=self.dtype,
            split=p.output_split,
            device=self.device,
            comm=self.comm,
            balanced=False,
        )

    def __getitem_advanced_distributed(self, p: ProcessedKey) -> "DNDarray":
        """
        Handles advanced indexing with unordered global indices. Defers to
        ``__getitem_unordered`` to resolve data dependencies via an ``Alltoallv`` exchange.
        """
        key = p.key

        # If key was not distributed, partition it so each rank requests its share of output
        if p.output_split is not None:
            if isinstance(key, torch.Tensor) and key.ndim > 0:
                key_split = p.output_split
                if key_split < key.ndim and key.shape[key_split] == p.output_shape[p.output_split]:
                    k_dnd = factories.array(
                        key, split=key_split, comm=self.comm, device=self.device
                    )
                    key = k_dnd.larray
            elif isinstance(key, tuple):
                split_k = key[self.split]
                if isinstance(split_k, torch.Tensor) and split_k.ndim > 0:
                    key_split = 0 if p.key_is_mask_like else split_k.ndim - 1
                    if split_k.shape[key_split] == p.output_shape[p.output_split]:
                        key_list = list(key)
                        if p.key_is_mask_like:
                            for idx in range(len(key_list)):
                                if isinstance(key_list[idx], torch.Tensor):
                                    kd = factories.array(
                                        key_list[idx],
                                        split=key_split,
                                        comm=self.comm,
                                        device=self.device,
                                    )
                                    key_list[idx] = kd.larray
                        else:
                            kd = factories.array(
                                split_k,
                                split=key_split,
                                comm=self.comm,
                                device=self.device,
                            )
                            key_list[self.split] = kd.larray
                        key = tuple(key_list)

        self, indexed_arr = self.__getitem_unordered(
            key=key,
            output_shape=p.output_shape,
            output_split=p.output_split,
            out_is_balanced=p.out_is_balanced,
            key_is_mask_like=p.key_is_mask_like,
        )
        return indexed_arr

    def __getitem_unordered(
        self,
        key: tuple,
        output_shape: tuple,
        output_split: int,
        out_is_balanced: bool,
        key_is_mask_like: bool,
    ) -> DNDarray:
        """
        Handles the MPI communication (Alltoallv) when the key along the
        split axis is unordered and indices are global.
        """
        _, displs = self.counts_displs()
        rank = self.comm.rank

        key_is_single_tensor = isinstance(key, torch.Tensor)
        split_key = key if key_is_single_tensor else key[self.split]
        split_key_flat = split_key.reshape(-1)

        # Calculate communication split axis for transposing later
        if key_is_single_tensor or key_is_mask_like:
            communication_split = 0
        else:
            communication_split = (
                output_split - (split_key.ndim - 1) if split_key.ndim > 1 else output_split
            )

        # Step 1: route and send index requests

        sort_idx, send_counts_t, send_displs_t, recv_counts_t, recv_displs_t = (
            self.__prepare_unordered_comm(split_key_flat, displs)
        )

        send_counts = send_counts_t.tolist()
        send_displs = send_displs_t.tolist()
        recv_counts = recv_counts_t.tolist()
        recv_displs = recv_displs_t.tolist()

        # Expand counts for multidimensional mask coordinates
        if key_is_mask_like:
            mask_dims = len(key)
            idx_send_counts = [c * mask_dims for c in send_counts]
            idx_send_displs = [d * mask_dims for d in send_displs]
            idx_recv_counts = [c * mask_dims for c in recv_counts]
            idx_recv_displs = [d * mask_dims for d in recv_displs]

            send_indices = torch.stack([k.flatten()[sort_idx] for k in key], dim=1).reshape(-1)
            recv_indices_flat = torch.empty(
                sum(idx_recv_counts), dtype=split_key.dtype, device=self.device.torch_device
            )
        else:
            idx_send_counts, idx_send_displs = send_counts, send_displs
            idx_recv_counts, idx_recv_displs = recv_counts, recv_displs

            send_indices = split_key_flat[sort_idx]
            recv_indices_flat = torch.empty(
                sum(idx_recv_counts), dtype=split_key.dtype, device=self.device.torch_device
            )

        self.comm.Alltoallv(
            (send_indices, idx_send_counts, idx_send_displs),
            (recv_indices_flat, idx_recv_counts, idx_recv_displs),
        )

        if key_is_mask_like:
            recv_indices = recv_indices_flat.reshape(sum(recv_counts), len(key))
        else:
            recv_indices = recv_indices_flat

        # Step 2: local data lookup based on received indices

        if key_is_mask_like:
            recv_indices[:, self.split] -= displs[rank]
            lookup_key = tuple(recv_indices[:, i] for i in range(len(key)))
            local_vals = self.larray[lookup_key]
        else:
            recv_indices -= displs[rank]
            if key_is_single_tensor:
                local_vals = self.larray[recv_indices]
            else:
                lookup_key = list(key)
                lookup_key[self.split] = recv_indices
                local_vals = self.larray[tuple(lookup_key)]

        # Step 3: return data to requesting processes

        # Ensure the indexed elements are aligned along axis 0
        transpose_axes = list(range(local_vals.ndim))
        transpose_axes[0], transpose_axes[communication_split] = (
            transpose_axes[communication_split],
            transpose_axes[0],
        )
        local_vals = local_vals.permute(*transpose_axes)

        feature_shape = list(local_vals.shape[1:])
        feature_size = 1
        for dim in feature_shape:
            feature_size *= dim

        return_send_counts = [c * feature_size for c in recv_counts]
        return_send_displs = [d * feature_size for d in recv_displs]
        return_recv_counts = [c * feature_size for c in send_counts]
        return_recv_displs = [d * feature_size for d in send_displs]

        send_vals = local_vals.reshape(-1)
        recv_vals_flat = torch.empty(
            sum(return_recv_counts), dtype=self.larray.dtype, device=self.device.torch_device
        )

        self.comm.Alltoallv(
            (send_vals, return_send_counts, return_send_displs),
            (recv_vals_flat, return_recv_counts, return_recv_displs),
        )

        # Step 4: reshape received values and reorder to match original key order

        recv_vals = recv_vals_flat.reshape(-1, *feature_shape)

        # Reverse the sorting applied in Step 1
        inv_sort_idx = torch.empty_like(sort_idx)
        inv_sort_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        unsorted_vals = recv_vals[inv_sort_idx]

        # Restore original dimension order
        final_vals = unsorted_vals.permute(*transpose_axes)

        # Reshape to match the global output shape expectation
        if split_key.ndim > 1 and not key_is_mask_like:
            original_local_shape = (
                output_shape[:communication_split]
                + split_key.shape
                + output_shape[communication_split + split_key.ndim :]
            )
            final_vals = final_vals.reshape(original_local_shape)

        indexed_arr = DNDarray(
            final_vals,
            gshape=output_shape,
            dtype=self.dtype,
            split=output_split,
            device=self.device,
            comm=self.comm,
            balanced=out_is_balanced,
        )

        return self, indexed_arr

    def __setitem_scalar(self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool) -> None:
        if p.root is not None:
            if self.comm.rank == p.root:
                self.__set(p.key, value)
        else:
            if not value_is_scalar:
                value = sanitation.sanitize_distribution(value, target=self[p.key])
            self.__set(p.key, value)

    def __setitem_local(self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool) -> None:
        """
        Handles process-local item assignment (slices and local indices)  directly on local partitions. If `value` is distributed, MPI communication might be necessary to align it with the target slice before assignment.
        """
        if value_is_scalar:
            self.__set(p.key, value)
            return

        value_is_distributed = isinstance(value, DNDarray) and value.is_distributed()

        if not self.is_distributed() and not value_is_distributed:
            self.__set(p.key, value)
            return

        if self.is_distributed():
            if not value.is_distributed():
                value = factories.array(
                    value.larray,
                    dtype=value.dtype,
                    split=p.output_split,
                    device=self.device,
                    comm=self.comm,
                )
            else:
                if value.split != p.output_split:
                    raise RuntimeError(
                        f"Cannot assign distributed `value` with split axis {value.split} "
                        f"to indexed DNDarray with split axis {p.output_split}."
                    )
            target_shape = torch.tensor(
                tuple(self.larray[p.key].shape), device=self.device.torch_device
            )
            target_map = torch.zeros(
                (self.comm.size, len(target_shape)),
                dtype=torch.int64,
                device=self.device.torch_device,
            )
            self.comm.Allgather(target_shape, target_map)
            value.redistribute_(target_map=target_map)

        self.__set(p.key, value)

    def __setitem_descending_slice_distributed(
        self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool
    ) -> None:
        """
        Handles assignment via negative-step slicing. Flips the `value` array and redistributes
        it to align with the descending split key before performing the local assignment.
        """
        if value_is_scalar:
            self.__set(p.key, value)
            return

        flipped_value = manipulations.flip(value, axis=p.output_split)

        # determine local element count along split axis
        split_key = p.key[self.split]
        step = 1 if split_key.step is None else split_key.step
        local_count = len(range(split_key.start, split_key.stop, step))

        # gather local slice counts across all ranks to build the target distribution map
        counts = torch.empty(
            (self.comm.size, 1), dtype=torch.int64, device=self.device.torch_device
        )
        self.comm.Allgather(
            torch.tensor([local_count], dtype=torch.int64, device=self.device.torch_device),
            counts,
        )

        if not flipped_value.is_distributed():
            flipped_value = factories.array(
                flipped_value.larray,
                dtype=flipped_value.dtype,
                split=p.output_split,
                device=self.device,
                comm=self.comm,
            )
        target_map = flipped_value.lshape_map
        target_map[:, p.output_split] = counts[:, 0]
        flipped_value.redistribute_(target_map=target_map)
        self.__set(p.key, flipped_value)

    def __setitem_mask(self, p: ProcessedKey, value: "DNDarray", value_is_scalar: bool) -> None:
        """
        Handles assignment using boolean masks. If `value` is distributed, it will be redistributed to match the number of True elements in the local mask before assignment. If `value` is not distributed, it will be assigned directly to the masked positions on each process, with PyTorch handling any necessary broadcasting.
        """
        pytorch_key = p.key

        if isinstance(pytorch_key, tuple):
            for k in pytorch_key:
                if isinstance(k, torch.Tensor) and k.dtype in (torch.bool, torch.uint8):
                    local_mask = k
                    break
        else:
            local_mask = pytorch_key

        if value_is_scalar:
            if isinstance(value, (int, float, bool, complex)):
                self.larray[pytorch_key] = value
                return
            elif hasattr(value, "larray"):
                scalar_torch = value.larray.type(self.dtype.torch_type())
            else:
                scalar_torch = torch.as_tensor(value, device=self.device.torch_device).type(
                    self.dtype.torch_type()
                )
            self.larray[pytorch_key] = scalar_torch
        else:
            if isinstance(value, DNDarray) and value.is_distributed():
                # value should align with local mask
                value_torch = value.larray
                rhs = (
                    value_torch
                    if value_torch.dtype == self.larray.dtype
                    else value_torch.type(self.dtype.torch_type())
                )
                try:
                    self.larray[pytorch_key] = rhs
                except RuntimeError as e:
                    raise ValueError(
                        f"Shape mismatch: Cannot assign distributed array with local shape {value.lshape} "
                        f"on rank {self.comm.rank}: {e}"
                    ) from e
            else:
                # Value is a non-distributed array -> MPI prefix sum needed
                value_torch = value.larray

                # distinguish between exact-shape masks and 1D row-filtering masks
                is_row_mask = local_mask.ndim == 1 and self.ndim > 1

                if not is_row_mask and value_torch.ndim == 1:
                    # N-D mask on N-D array -> flattens into 1D sequence, requires MPI prefix sum
                    local_mask_flat = local_mask.flatten()
                    local_true = int(local_mask_flat.sum().item())

                    if self.comm.rank == 0:
                        offset = 0
                        _ = self.comm.exscan(local_true)
                    else:
                        offset = self.comm.exscan(local_true)

                    rhs_local = value_torch[offset : offset + local_true].type(
                        self.dtype.torch_type()
                    )

                    x_flat = self.larray.view(-1)
                    x_flat[local_mask_flat] = rhs_local
                else:
                    # PyTorch assigns and broadcasts natively
                    self.larray[pytorch_key] = value_torch.type(self.dtype.torch_type())

    def __setitem_advanced_distributed(
        self,
        p: ProcessedKey,
        original_key,
        value: "DNDarray",
        value_is_scalar: bool,
        original_split: int = None,
    ) -> None:
        """
        Handles advanced indexing assignments where the indexing key is distributed. This method ensures that the value array is properly aligned and redistributed if necessary before performing the local assignment on each process.
        """
        # check distribution status of the indexing key
        split_key_orig = (
            original_key[self.split] if isinstance(original_key, tuple) else original_key
        )
        key_is_distributed = (
            isinstance(split_key_orig, DNDarray) and split_key_orig.is_distributed()
        )
        value_is_distributed = isinstance(value, DNDarray) and value.is_distributed()

        # reject implicit cross-distribution assignments
        if key_is_distributed and not value_is_distributed and not value_is_scalar:
            raise ValueError(
                f"Distribution mismatch: index distributed={key_is_distributed}, value distributed={value_is_distributed}. "
                "Cannot assign a non-distributed value array using a distributed index. "
                "Please distribute the value array or use a non-distributed index."
            )

        counts, displs = self.counts_displs()

        if value_is_distributed:
            self.__setitem_unordered(
                key=p.key,
                key_is_mask_like=p.key_is_mask_like,
                value=value,
                key_is_single_tensor=isinstance(p.key, torch.Tensor),
                counts=counts,
                displs=displs,
                rank=self.comm.rank,
                key_is_distributed=key_is_distributed,
            )
            return

        rank = self.comm.rank
        key_is_single_tensor = isinstance(p.key, torch.Tensor)

        if (
            value_is_scalar
            and isinstance(original_key, tuple)
            and len(original_key) == self.ndim
            and all(
                isinstance(k, DNDarray) and k.ndim == 1 and k.dtype in (types.int32, types.int64)
                for k in original_key
            )
        ):
            global_indices = []
            for k in original_key:
                k_full = k.copy()
                k_full.resplit_(None)
                global_indices.append(k_full.larray)

            idx_split_global = global_indices[self.split]
            local_offset = displs[rank]
            local_size = counts[rank]

            mask = (idx_split_global >= local_offset) & (
                idx_split_global < local_offset + local_size
            )
            if not mask.any():
                return

            lhs_index = []
            for dim, gind in enumerate(global_indices):
                sel = gind[mask]
                if dim == self.split:
                    sel = sel - local_offset
                lhs_index.append(sel)
            lhs_index = tuple(lhs_index)

            if hasattr(value, "larray"):
                scalar_torch = value.larray
            else:
                scalar_torch = torch.as_tensor(value, device=self.device.torch_device)
            scalar_torch = scalar_torch.type(self.dtype.torch_type())

            self.larray[lhs_index] = scalar_torch
            return

        if key_is_single_tensor:
            split_key = p.key
            split_key_flat = split_key.reshape(-1)
            local_indices = torch.nonzero(
                (split_key_flat >= displs[rank]) & (split_key_flat < displs[rank] + counts[rank])
            ).flatten()

            if local_indices.numel() > 0:
                key_local = split_key_flat[local_indices] - displs[rank]

                if value_is_scalar:
                    rhs = (
                        value.larray.type(self.dtype.torch_type())
                        if hasattr(value, "larray")
                        else value
                    )
                else:
                    # flatten leading dimensions of value.larray that correspond to the multi-dimensional key
                    rhs_view = value.larray.reshape(-1, *value.larray.shape[split_key.ndim :])
                    rhs = rhs_view[local_indices].type(self.dtype.torch_type())

                if self.larray.is_cuda:
                    key_local, rhs = _resolve_duplicate_indices(key_local, rhs, self.larray.shape)

                self.larray[key_local] = rhs
            return

        raw_split_part = original_key[original_split]

        if isinstance(raw_split_part, DNDarray):
            split_key = raw_split_part.larray
        elif isinstance(raw_split_part, torch.Tensor):
            split_key = raw_split_part
        else:
            split_key = p.key[self.split]

        if isinstance(split_key, DNDarray):
            split_key = split_key.larray

        if split_key.dtype == torch.bool:
            split_key = torch.nonzero(split_key, as_tuple=False).flatten()

        local_offset = displs[rank]
        local_size = counts[rank]

        if hasattr(value, "larray"):
            value_torch = value.larray
        else:
            value_torch = torch.as_tensor(value, device=self.device.torch_device)

        feature_dims = self.larray.ndim - (self.split + 1)

        if value_is_scalar:
            value_key_start_dim = 0
        else:
            value_key_start_dim = value_torch.ndim - split_key.ndim - feature_dims
            if value_key_start_dim < 0:
                raise RuntimeError("value_key_start_dim < 0 – inconsistent shapes")

        local_split_axis = self.split

        base_index = [slice(None)] * self.larray.ndim
        if isinstance(original_key, tuple):
            for dim, k_part in enumerate(original_key):
                if dim == self.split:
                    continue
                if isinstance(k_part, DNDarray):
                    base_index[dim] = k_part.larray
                else:
                    base_index[dim] = k_part

        _setitem_advanced_unordered_local(
            x_local=self.larray,
            split_key=split_key,
            value_torch=value_torch,
            split_axis=local_split_axis,
            value_key_start_dim=value_key_start_dim,
            local_offset=local_offset,
            local_size=local_size,
            value_is_scalar=value_is_scalar,
            out_dtype=self.dtype.torch_type(),
            base_index=tuple(base_index),
        )

    def __setitem_unordered(
        self,
        key: tuple | list | torch.Tensor,
        key_is_mask_like: bool,
        value: "DNDarray",
        key_is_single_tensor: bool,
        counts: tuple,
        displs: tuple,
        rank: int,
        key_is_distributed: bool = False,
    ) -> DNDarray:
        """
        Handles the MPI communication when assigning a distributed
        value to a distributed array with unordered global indices.
        """
        # distribution of `key` and `value` must be aligned
        if key_is_mask_like:
            if key_is_distributed:
                split_key = key[self.split]
                global_split_key = factories.array(
                    split_key, is_split=0, device=self.device, comm=self.comm, copy=False
                )
                target_map = value.lshape_map
                target_map[:, value.split] = global_split_key.lshape_map[:, 0]
                value.redistribute_(target_map=target_map)
            else:
                # Key is replicated: slice locally to match value partition directly
                v_counts, v_displs = value.counts_displs()
                start = v_displs[rank]
                end = start + v_counts[rank]
                key = tuple(k[start:end] if isinstance(k, torch.Tensor) else k for k in key)
                split_key = key[self.split]
        else:
            if key_is_distributed:
                # redistribute split-axis `key` to match distribution of `value` in one pass
                if key_is_single_tensor:
                    split_key = key
                else:
                    split_key = key[self.split]
                global_split_key = factories.array(
                    split_key, is_split=0, device=self.device, comm=self.comm, copy=False
                )
                target_map = global_split_key.lshape_map
                target_map[:, 0] = value.lshape_map[:, value.split]
                global_split_key.redistribute_(target_map=target_map)
                split_key = global_split_key.larray
            else:
                # Key is replicated: slice locally to match value partition directly
                v_counts, v_displs = value.counts_displs()
                start = v_displs[rank]
                end = start + v_counts[rank]
                if key_is_single_tensor:
                    key = key[start:end]
                    split_key = key
                else:
                    key_list = list(key)
                    key_list[self.split] = key_list[self.split][start:end]
                    key = tuple(key_list)
                    split_key = key[self.split]

        # key and value are now aligned

        # prepare for `value` Alltoallv:
        # work along axis 0, transpose if necessary
        transpose_axes = list(range(value.ndim))
        transpose_axes[0], transpose_axes[value.split] = (
            transpose_axes[value.split],
            transpose_axes[0],
        )
        value = value.transpose(transpose_axes)

        split_key_flat = split_key.reshape(-1)
        sort_idx, send_counts, send_displs, recv_counts, recv_displs = (
            self.__prepare_unordered_comm(split_key_flat, displs)
        )

        send_counts_l = send_counts.tolist()
        send_displs_l = send_displs.tolist()
        recv_counts_l = recv_counts.tolist()
        recv_displs_l = recv_displs.tolist()

        # exchange indices
        if key_is_mask_like:
            mask_dims = len(key)
            idx_send_counts = [c * mask_dims for c in send_counts_l]
            idx_send_displs = [d * mask_dims for d in send_displs_l]
            idx_recv_counts = [c * mask_dims for c in recv_counts_l]
            idx_recv_displs = [d * mask_dims for d in recv_displs_l]
            send_idx = torch.stack([k.flatten()[sort_idx] for k in key], dim=1).reshape(-1)
            recv_idx_flat = torch.empty(
                sum(idx_recv_counts), dtype=split_key.dtype, device=self.device.torch_device
            )
        else:
            idx_send_counts, idx_send_displs = send_counts_l, send_displs_l
            idx_recv_counts, idx_recv_displs = recv_counts_l, recv_displs_l
            send_idx = split_key_flat[sort_idx]
            recv_idx_flat = torch.empty(
                sum(idx_recv_counts), dtype=split_key.dtype, device=self.device.torch_device
            )

        self.comm.Alltoallv(
            (send_idx, idx_send_counts, idx_send_displs),
            (recv_idx_flat, idx_recv_counts, idx_recv_displs),
        )

        # exchange value
        trailing_dims_shape = list(value.lshape[1:])
        trailing_dims_size = math.prod(trailing_dims_shape)

        val_send_counts = [c * trailing_dims_size for c in send_counts_l]
        val_send_displs = [d * trailing_dims_size for d in send_displs_l]
        val_recv_counts = [c * trailing_dims_size for c in recv_counts_l]
        val_recv_displs = [d * trailing_dims_size for d in recv_displs_l]

        send_vals = value.larray[sort_idx].contiguous().reshape(-1)
        recv_vals_flat = torch.empty(
            sum(val_recv_counts), dtype=value.larray.dtype, device=self.device.torch_device
        )
        self.comm.Alltoallv(
            (send_vals, val_send_counts, val_send_displs),
            (recv_vals_flat, val_recv_counts, val_recv_displs),
        )

        if key_is_mask_like:
            recv_indices = recv_idx_flat.reshape(sum(recv_counts_l), len(key))
            recv_indices[:, 0] -= displs[rank]
            key = tuple(recv_indices[:, i] for i in range(len(key)))
        else:
            # store incoming indices in int 1-D tensor and correct for rank offset
            recv_indices = recv_idx_flat - displs[rank]

            # replace split-axis key with incoming local indices
            if key_is_single_tensor:
                key = recv_indices
            else:
                key = list(key)
                key[self.split] = recv_indices
                key = tuple(key)

        recv_vals = recv_vals_flat.reshape(-1, *trailing_dims_shape)
        recv_buf = DNDarray(
            recv_vals.permute(*transpose_axes),
            gshape=value.gshape,
            dtype=value.dtype,
            split=value.split,
            device=value.device,
            comm=value.comm,
            balanced=value.balanced,
        )
        # set local elements of `self` to corresponding elements of `value`
        self.__set(key, recv_buf)
        return self


# Heat imports at the end to break cyclic dependencies
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
from . import types

from .devices import Device
from .stride_tricks import sanitize_axis
from .types import datatype, integer, canonical_heat_type
from .types import bool as ht_bool, uint8 as ht_uint8
from ._indexing_utils import (
    ProcessedKey,
    _resolve_duplicate_indices,
    _resolve_indexing_state,
    _unwrap_local_key,
    _setitem_advanced_unordered_local,
    _broadcast_value,
)
