"""
Implements a distributed counterpart of xarray built on top of Heats DNDarray class
"""

import torch
import heat as ht
import xarray as xr
from xarray import DataArray
from typing import Union

# imports of "dxarray_..."-dependencies at the end to avoid cyclic dependence

__all__ = ["DXarray", "from_xarray"]


# Auxiliary functions


def dim_name_to_idx(dims: list, names: Union[str, tuple, list, None]) -> Union[int, tuple, list]:
    """
    Converts a string "names" (or tuple of strings) referring to dimensions stored in "dims" to the corresponding numeric index (tuple of indices) of these dimensions.
    Inverse of :func:`dim_idx_to_name`.
    """
    if names is None:
        return None
    elif isinstance(names, str):
        return dims.index(names)
    elif isinstance(names, tuple):
        names_list = list(names)
        return tuple([dims.index(name) for name in names_list])
    elif isinstance(names, list):
        return [dims.index(name) for name in names]
    else:
        raise TypeError("Input names must be None, string, list of strings, or tuple of strings.")


def dim_idx_to_name(dims: list, idxs: Union[int, tuple, list, None]) -> Union[str, tuple, list]:
    """
    Converts an numeric index "idxs" (or tuple of such indices) referring to the dimensions stored in "dims" to the corresponding name string (or tuple of name strings).
    Inverse of :func:`dim_name_to_idx`.
    """
    if idxs is None:
        return None
    elif isinstance(idxs, int):
        return dims[idxs]
    elif isinstance(idxs, tuple):
        idxs_list = list(idxs)
        return tuple([dims[idx] for idx in idxs_list])
    elif isinstance(idxs, list):
        return [dims[idx] for idx in idxs]
    else:
        raise TypeError("Input idxs must be None, int, list of ints, or tuple of ints.")


class DXarray:
    """
    Distributed counterpart of xarray.

    Parameters
    --------------
    values: DNDarray
        data entries of the DXarray
    dims: list
        names of the dimensions of the DXarray
    coords: dictionary
        coordinates
        entries of the dictionary have the form `dim`:`coords_of_dim` for each `dim` in `dims`,
        where `coords_of_dim` can either be a list of coordinate labels ("logical coordinates") or an
        DXarray of same shape as the original one, also split along the same split axis ("physical coordinates").
    split: Union[int,None]
        dimension along which the DXarray is split (analogous to split dimension of DNDarray)

    Notes
    ---------------
    Some attributes of DNDarray are not included in DXarray, e.g., gshape, lshape, larray etc., and need to be accessed by
    DXarray.values.gshape etc.
    This is in order to avoid confusion, because a DXarray is built of possibly several DNDarrays which could cause confusion
    to which gshape etc. a global attribute DXarray.gshape could refer to.
    Currently, it is checked whether values and coords are on the same `device`; in principle, this is unnecessary.
    """

    def __init__(
        self,
        values: ht.DNDarray,
        dims: Union[list, None] = None,
        coords: Union[dict, None] = None,
        name: Union[str, None] = None,
        attrs: dict = {},
    ):
        """
        Constructor for DXarray class
        """
        # Check compatibility of the input arguments
        dxarray_sanitation.check_compatibility_values_dims_coords(values, dims, coords)
        dxarray_sanitation.check_name(name)
        dxarray_sanitation.check_attrs(attrs)

        # after the checks, set the directly given attributes...

        self.__values = values
        self.__name = name
        self.__attrs = attrs
        self.__coords = coords
        self.__device = values.device
        self.__comm = values.comm

        # ... and determine those not directly given:
        # since we are in the DXarray class, split dimension is given by a string
        self.__split = dim_idx_to_name(dims, values.split)

        # determine dimensions with and without coordinates
        if coords is not None:
            dims_with_coords = sum([list(it[0]) for it in coords.items()], [])
        else:
            dims_with_coords = []
        dims_without_coords = [dim for dim in dims if dim not in dims_with_coords]

        self.__dims_with_cooords = dims_with_coords
        self.__dims_without_coords = dims_without_coords

        # check if all appearing DNDarrays are balanced: as a result, the DXarray is balanced if and only if all DNDarrays are balanced
        self.__balanced = dxarray_sanitation.check_if_balanced(self.__values, self.__coords)

        # if no names are provided, introduce generic names "dim_N", N = 0,1,...
        if dims is None:
            self.__dims = ["dim_%d" % k for k in range(self.__values.ndim)]
        else:
            self.__dims = dims

    """
    Attribute getters and setters for the DXarray class
    """

    @property
    def values(self) -> ht.DNDarray:
        """
        Get values from DXarray
        """
        return self.__values

    @property
    def dims(self) -> list:
        """
        Get dims from DXarray
        """
        return self.__dims

    @property
    def coords(self) -> dict:
        """
        Get coords from DXarray
        """
        return self.__coords

    @property
    def split(self) -> Union[str, None]:
        """
        Get split dimension from DXarray
        """
        return self.__split

    @property
    def device(self) -> ht.Device:
        """
        Get device from DXarray
        """
        return self.__device

    @property
    def comm(self) -> ht.Communication:
        """
        Get communicator from DXarray
        """
        return self.__comm

    @property
    def name(self) -> str:
        """
        Get name from DXarray
        """
        return self.__name

    @property
    def attrs(self) -> dict:
        """
        Get attributes from DXarray
        """
        return self.__attrs

    @property
    def dims_with_coordinates(self) -> list:
        """
        Get list of dims with coordinates from DXarray
        """
        return self.__dims_with_coordinates

    @property
    def dims_without_coordinates(self) -> list:
        """
        Get list of dims without coordinates from DXarray
        """
        return self.__dims_without_coordinates

    @property
    def balanced(self) -> bool:
        """
        Get the attributed `balanced` of DXarray.
        Does not check whether the current value of this attribute is consistent!
        (This can be ensured by calling :meth:`DXarray.is_balanced(force_check=True)` first.)
        """
        return self.__balanced

    @values.setter
    def values(self, newvalues: ht.DNDarray):
        """
        Set value array of DXarray
        """
        dxarray_sanitation.check_compatibility_values_dims_coords(
            newvalues, self.__dims, self.__coords
        )
        self.__values = newvalues

    @coords.setter
    def coors(self, newcoords: Union[dict, None]):
        """
        Set coordinates of DXarray
        """
        dxarray_sanitation.check_compatibility_values_dims_coords(
            self.__values, self.__dims, newcoords
        )
        self.__coords = newcoords

    @name.setter
    def name(self, newname: Union[str, None]):
        """
        Set name of DXarray
        """
        dxarray_sanitation.check_name(newname)
        self.__name = newname

    @attrs.setter
    def attrs(self, newattrs: Union[dict, None]):
        """
        Set attributes of DXarray
        """
        dxarray_sanitation.check_attrs(newattrs)
        self.__attrs = newattrs

    """
    Private methods of DXarray class
    """

    def __dim_name_to_idx(
        self, names: Union[str, tuple, list, None]
    ) -> Union[str, tuple, list, None]:
        """
        Converts a string (or tuple of strings) referring to dimensions of the DXarray to the corresponding numeric index (tuple of indices) of these dimensions.
        Inverse of :meth:`__dim_idx_to_name`.
        """
        return dim_name_to_idx(self.__dims, names)

    def __dim_idx_to_name(
        self, idxs: Union[int, tuple, list, None]
    ) -> Union[int, tuple, list, None]:
        """
        Converts an numeric index (or tuple of such indices) referring to the dimensions of the DXarray to the corresponding name string (or tuple of name strings).
        Inverse of :meth:`__dim_name_to_idx`.
        """
        return dim_idx_to_name(self.__dims, idxs)

    def __repr__(self) -> str:
        """
        Representation of DXarray as string. Required for printing.
        """
        if self.__name is not None:
            print_name = self.__name
        else:
            print_name = "<without_name>"
        print_values = self.__values.__repr__()
        print_dimensions = ", ".join(self.__dims)
        if self.__split is not None:
            print_split = self.__split
        else:
            print_split = "None (no splitted)"
        if self.__coords is not None:
            print_coords = "\n".join(
                [it[0].__repr__() + ": \t" + it[1].__repr__() for it in self.__coords.items()]
            )
            print_coords = 'Coordinates of "' + print_name + '": ' + print_coords
        else:
            print_coords = ""
        print_attributes = "\n".join(
            ["\t" + it[0].__repr__() + ": \t" + it[1].__repr__() for it in self.__attrs.items()]
        )
        if len(self.__dims_without_coords) != 0:
            print_coordinates_without_dims = "".join(
                [
                    'The remaining coordinates of "',
                    print_name,
                    '", ',
                    ", ".join(self.__dims_without_coords),
                    ", do not have coordinates. \n",
                ]
            )
        else:
            print_coordinates_without_dims = ""
        if self.__comm.rank == 0:
            return "".join(
                [
                    'DXarray with name "',
                    print_name,
                    '"\n',
                    'Dimensions of "',
                    print_name,
                    '": ',
                    print_dimensions,
                    "\n",
                    'Split dimension of "',
                    print_name,
                    '": ',
                    print_split,
                    "\n",
                    'Values of "',
                    print_name,
                    '": ',
                    print_values,
                    "\n",
                    print_coords,
                    "\n",
                    print_coordinates_without_dims,
                    'Attributes of "',
                    print_name,
                    '":',
                    print_attributes,
                    "\n\n",
                ]
            )
        else:
            return ""

    """
    Public Methods of DXarray
    """

    def is_balanced(self, force_check: bool = False) -> bool:
        """
        Checks if DXarray is balanced. If `force_check = False` (default), the current value of the
        attribute `balanced` is returned unless this current value is None (i.e. no information on
        no information available); only in the latter case, or if `force_check = True`, the value
        of the attribute `balanced` is updated before being returned.

        """
        if self.__balanced is None or force_check:
            self.__balanced = dxarray_sanitation.check_if_balanced(self.__values, self.__coords)
        return self.__balanced

    def resplit_(self, dim: Union[str, None] = None):
        """
        In-place option for resplitting a :class:`DXarray`.
        """
        if dim is not None and dim not in self.__dims:
            raise ValueError(
                "Input `dim` in resplit_ must be either None or a dimension of the underlying DXarray."
            )
        # early out if nothing is to do
        if self.__split == dim:
            return self
        else:
            # resplit the value array accordingly
            self.__values.resplit_(self.__dim_name_to_idx(dim))
            if self.__coords is not None:
                for item in self.__coords.items():
                    if isinstance(item[0], str) and item[0] == dim:
                        item[1].resplit_(0)
                    elif isinstance(item[0], tuple) and dim in item[0]:
                        item[1].resplit_(dim)
            self.__split = dim
            return self

    def balance_(self):
        """
        In-place option for balancing a :class:`DXarray`.
        """
        if self.is_balanced(force_check=True):
            return self
        else:
            self.__values.balance_()
            if self.__coords is not None:
                for item in self.__coords.items():
                    item[1].balance_()
            self.__balanced = True
            return self

    def xarray(self):
        """
        Convert given DXarray (possibly distributed over some processes) to a non-distributed xarray (:class:`xarray.DataArray`)
        """
        non_dist_copy = self.resplit_(None)
        if non_dist_copy.coords is None:
            xarray_coords = None
        else:
            xarray_coords = {
                item[0]: item[1].cpu().numpy()
                if isinstance(item[1], ht.DNDarray)
                else item[1].xarray()
                for item in non_dist_copy.coords.items()
            }
        xarray = DataArray(
            non_dist_copy.values.cpu().numpy(),
            dims=non_dist_copy.dims,
            coords=xarray_coords,
            name=non_dist_copy.name,
            attrs=non_dist_copy.attrs,
        )
        del non_dist_copy
        return xarray


def from_xarray(
    xarray: xr.DataArray,
    split: Union[str, None] = None,
    device: ht.Device = None,
    comm: ht.Communication = None,
) -> DXarray:
    """
    Generates a DXarray from a given xarray (:class:`xarray.DataArray`)
    """
    coords_dict = {
        item[0]: ht.from_numpy(item[1].values, device=device, comm=comm)
        if len(item[0]) == 1
        else DXarray(
            ht.from_numpy(item[1].values, device=device, comm=comm),
            dims=list(item[0]),
            coords=None,
            name=item[1].name.__str__(),
            attrs=item[1].attrs,
        )
        for item in xarray.coords.items()
    }
    dxarray = DXarray(
        ht.from_numpy(xarray.values, device=device, comm=comm),
        dims=list(xarray.dims),
        coords=coords_dict,
        name=xarray.name,
        attrs=xarray.attrs,
    )
    if split is not None:
        if split not in dxarray.dims:
            raise ValueError('split dimension "', split, '" is not a dimension of input array.')
        else:
            dxarray.resplit_(split)
    return dxarray


from . import dxarray_sanitation
from . import dxarray_manipulations
