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
    """

    def __init__(
        self,
        values: ht.DNDarray,
        dims: Union[list, None],
        coords: dict,
        name: Union[str, None] = None,
        attrs: dict = {},
    ):
        self.__values = values
        self.__name = name
        self.__attrs = attrs

        if dims is not None:
            assert len(self.__dims) == self.__values.ndim
            self.__dims = dims
        else:
            self.__dims = ["dim_%d" % k for k in range(self.__values.ndim)]
        self.__dims = dims

        self.__coords = coords
        self.__split = values.split
        self.__device = values.device
        self.__comm = values.comm

        for coord_item in coords.items():
            if coord_item[1] is not None:
                assert (
                    isinstance(coord_item[1], ht.DNDarray)
                    and coord_item[1].device == self.__values.device
                    and coord_item[1].comm == self.__values.comm
                )
                if coord_item[1].split is not None:
                    0 == 1
                    # ensure correct split dim...
        # TODO: we need to introduce some additional consistency checks: compare communicator, devices, and split-dimension of DNDarrays contained in coords and of values...
        #     physical coordinate arrays must be split along same dim as values...

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
