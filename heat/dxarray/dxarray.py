"""
Implements a distributed counterpart of xarray built on top of Heats DNDarray class
"""

import heat as ht

# import xarray as xa
from typing import Union

__all__ = ["DXarray"]


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
    """

    def __init__(
        self,
        values: ht.DNDarray,
        dims: Union[list, None],
        coords: Union[dict, None],
        name: Union[str, None] = None,
        attrs: dict = {},
    ):
        self.__values = values
        self.__name = name
        self.__attrs = attrs

        # check if names of dims are given (and whether their number fits the number of dims of the values array)
        # if no names are provided, introduce generic names "dim_N", N = 0,1,...
        if dims is not None:
            assert len(self.__dims) == self.__values.ndim
            self.__dims = dims
        else:
            self.__dims = ["dim_%d" % k for k in range(self.__values.ndim)]
        self.__dims = dims

        # set attribute split: use dimension name instead of idx since we are in class DXarray instead of DNDarray
        self.__split = self.__dim_idx_to_name(values.split)

        # check consistency of the coordinates provided
        if coords is not None:
            # go through all entries in the dictionary coords
            for coord_item in coords.items():
                coord_item_dims = coord_item[0]
                coord_item_coords = coord_item[1]
                # first case: "classical" coordinates for a single dimension, sometimes referred to "logical coordinates"
                if isinstance(coord_item_dims, str):
                    # here, the coordinates must be given by a one-dimensional DNDarray...
                    assert isinstance(coord_item_coords, ht.DNDarray)
                    assert coord_item_coords.ndim == 1
                    # ... with matching device and communicator, ...
                    assert coord_item_coords.device == self.__values.device
                    assert coord_item_coords.comm == self.__values.comm
                    # ... correct shape, and ...
                    assert (
                        coord_item_coords.gshape[0]
                        == self.__values.gshape[self.__dim_name_to_idx(coord_item_dims)]
                    )
                    # ... that is split if and only if the coordinates refer to the split dimension of the DXarray
                    if coord_item_dims == self.__split:
                        assert coord_item_coords.split == 0
                    else:
                        assert coord_item_coords.split is None
                # second case: "physical coordinates" - two or more dimensions are "merged" together and equipped with a coordinate array
                # that cannot be expressed as meshgrid of 1d coordinate arrays
                elif isinstance(coord_item_dims, tuple(int)):
                    # now, the coordinates must be given as a DXarray...
                    assert isinstance(coord_item_coords, ht.DXarray)
                    # ... with matching dimension names, ...
                    assert coord_item_coords.dims == list(coord_item_dims)
                    # ... shape, ...
                    assert (
                        coord_item_coords.values.gshape
                        == self.__values.gshape[self.__dim_name_to_idx(list(coord_item_dims))]
                    )
                    # ... device and communicator, ...
                    assert coord_item_coords.device == self.__values.device
                    assert coord_item_coords.comm == self.__values.comm
                    # ... and split dimension.
                    if self.__split in coord_item_dims:
                        assert coord_item_coords.split == self.__split
                    else:
                        assert coord_item_coords.split is None

        # after the consistency checks, set the remaining attributes of the DXarray
        self.__coords = coords
        self.__device = values.device
        self.__comm = values.comm

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

    @values.setter
    def values(self, arr: ht.DNDarray):
        """
        Set value array of DXarray
        """
        # TODO: perform some consistency checks...
        self.__values = arr

    @name.setter
    def name(self, name: str):
        """
        Set name of DXarray
        """
        self.__name = name

    @attrs.setter
    def attrs(self, attributes: dict):
        """
        Set attributes of DXarray
        """
        self.__attrs = attributes

    def __dim_name_to_idx(self, names: Union[str, tuple, list, None]):
        """
        Converts a string (or tuple of strings) referring to dimensions of the DXarray to the corresponding numeric index (tuple of indices) of these dimensions.
        Inverse of :meth:`__dim_idx_to_name`.
        """
        if names is None:
            return None
        elif isinstance(names, str):
            return self.__dims.index(names)
        elif isinstance(names, tuple):
            names_list = list(names)
            return tuple([self.__dims.index(names) for name in names_list])
        elif isinstance(names, list):
            return tuple([self.__dims.index(names) for name in names])
        else:
            raise TypeError("Input must be None, string, list of strings, or tuple of strings.")

    def __dim_idx_to_name(self, idxs: Union[int, tuple, list, None]):
        """
        Converts an numeric index (or tuple of such indices) referring to the dimensions of the DXarray to the corresponding name string (or tuple of name strings).
        Inverse of :meth:`__dim_name_to_idx`.
        """
        if idxs is None:
            return None
        elif isinstance(self, idxs):
            return self.__dims[idxs]
        elif isinstance(idxs, tuple):
            idxs_list = list(idxs)
            return tuple([self.__dims[idx] for idx in idxs_list])
        elif isinstance(idxs, list):
            return tuple([self.__dims[idx] for idx in idxs])
        else:
            raise TypeError("Input must be None, int, list of ints, or tuple of ints.")
