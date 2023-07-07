"""
Implements a distributed counterpart of xarray built on top of Heats DNDarray class
"""

import torch
import heat as ht
import xarray as xr
from typing import Union

# imports of "dxarray_..."-dependencies at the end to avoid cyclic dependence

__all__ = ["DXarray"]


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
        if coords is not None:
            balanced = values.balanced and all(
                [coord_item[1].balanced for coord_item in coords.items()]
            )
        else:
            balanced = values.balanced
        self.__balanced = balanced

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
    def split(self) -> Union[int, None]:
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
        Check whether all DNDarrays in DXarray are balanced
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

    def __dim_name_to_idx(self, names: Union[str, tuple, list, None]):
        """
        Converts a string (or tuple of strings) referring to dimensions of the DXarray to the corresponding numeric index (tuple of indices) of these dimensions.
        Inverse of :meth:`__dim_idx_to_name`.
        """
        return dim_name_to_idx(self.__dims, names)

    def __dim_idx_to_name(self, idxs: Union[int, tuple, list, None]):
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


from . import dxarray_sanitation
from . import dxarray_manipulations
