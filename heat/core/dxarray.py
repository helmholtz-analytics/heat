"""
Implements a distributed counterpart of xarray built on top of Heats DNDarray class
"""

import heat as ht


class DXarray:
    """
    Distributed counterpart of xarray.

    Parameters
    --------------
    dataarray: DNDarray
        data entries of the DXarray
    coords: dictionary
        coordinates
        entries of the dictionary have the form `dim`:`coords_of_dim` for each `dim` in `dims`, where `coords_of_dim` can either be a list of coordinate labels ("logical coordinates") or an DXarray of same shape as the original one, also split along the same split axis ("physical coordinates")
    dims: List
        names of the dimensions of the DXarray
    split: Union[int,None]
        split dimension of the DXarray (analogous to split dimension of DNDarray)
    """

    # TODO: @properties ...
