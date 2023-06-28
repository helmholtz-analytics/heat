"""
Implements a distributed counterpart of xarray built on top of Heats DNDarray class
"""

import heat as ht


class Dxarray:
    """
    Distributed counterpart of xarray.

    Parameters
    --------------
    dataarray: DNDarray
        entries of the xarray
    coords: dictionary
        coordinates
        entries of the dictionary have the form `dim`:`coords_of_dim` for each `dim` in `dims`, where `coords_of_dim` can either be a list of coordinate labels ("logical coordinates") or an Dxarray of same shape as the original one, also split along the same split axis ("physical coordinates")
    dims: List
        names of the dimensions
    split: Union[int,None]
        split dimension of the Dxarray (analogous to split dimension of DNDarray)
    """

    # TODO: @properties ...
