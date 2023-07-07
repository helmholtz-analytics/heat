"""
Validation/Sanitation routines for the DXarray class
"""

import torch
import heat as ht
from typing import Any, Union

from .dxarray import DXarray, dim_name_to_idx, dim_idx_to_name


def check_compatibility_values_dims_coords(
    values: ht.DNDarray, dims: Union[list, None], coords: Union[dict, None]
):
    """
    Checks whether input values, dims, and coords are valid and compatible inputs for a DXarray
    """
    if not isinstance(values, ht.DNDarray):
        raise TypeError("Input `values` must be a DNDarray, but is ", type(values), ".")
    if not (isinstance(dims, list) or dims is None):
        raise TypeError("Input `dims` must be a list or None, but is ", type(dims), ".")
    if not (isinstance(coords, dict) or coords is None):
        raise TypeError("Input `coords` must be a dictionary or None, but is ", type(coords), ".")

    # check if names of dims are given (and whether their number fits the number of dims of the values array)
    if dims is not None:
        if len(dims) != values.ndim:
            raise ValueError(
                "Number of dimension names in `dims` (=%d) must be equal to number of dimensions of `values` array (=%d)."
                % (len(dims), values.ndim)
            )

    # check consistency of the coordinates provided
    if coords is not None:
        # go through all entries in the dictionary coords
        for coord_item in coords.items():
            coord_item_dims = coord_item[0]
            coord_item_coords = coord_item[1]
            # first case: "classical" coordinates for a single dimension, sometimes referred to "logical coordinates"
            if isinstance(coord_item_dims, str):
                # here, the coordinates must be given by a one-dimensional DNDarray...
                if not isinstance(coord_item_coords, ht.DNDarray):
                    raise TypeError(
                        "Coordinate arrays (i.e. entries of `coords`) for single dimension must be DNDarray. Here, type ",
                        type(coord_item_coords),
                        " is given for dimension ",
                        coord_item_dims,
                        ".",
                    )
                if not coord_item_coords.ndim == 1:
                    raise ValueError(
                        "Coordinate arrays for a single dimension must have dimension 1, but coordinate array for dimension ",
                        coord_item_dims,
                        " has dimension %d." % coord_item_coords.ndim,
                    )
                # ... with matching device and communicator, ...
                if not coord_item_coords.device == values.device:
                    raise RuntimeError(
                        "Device of coordinate array for dimension ",
                        coord_item_dims,
                        "does not coincide with device for `values`.",
                    )
                if not coord_item_coords.comm == values.comm:
                    raise RuntimeError(
                        "Communicator of coordinate array for dimension ",
                        coord_item_dims,
                        "does not coincide with device for `values`.",
                    )
                # ... correct shape, and ...
                if not (
                    coord_item_coords.gshape[0]
                    == values.gshape[dim_name_to_idx(dims, coord_item_dims)]
                ):
                    raise ValueError(
                        "Size of `values` in dimension ",
                        coord_item_dims,
                        " does not coincide with size of coordinate array in this dimension.",
                    )
                # ... that is split if and only if the coordinates refer to the split dimension of the DXarray
                if coord_item_dims == dim_idx_to_name(dims, values.split):
                    if coord_item_coords.split != 0:
                        raise ValueError(
                            "`values` array is split along dimension ",
                            coord_item_dims,
                            ", but cooresponding coordinate array is not split along this dimension.",
                        )
                else:
                    if coord_item_coords.split is not None:
                        raise ValueError(
                            "`values` array is not split along dimension ",
                            coord_item_dims,
                            ", but cooresponding coordinate array is split along this dimension.",
                        )
            # second case: "physical coordinates" - two or more dimensions are "merged" together and equipped with a coordinate array
            # that cannot be expressed as meshgrid of 1d coordinate arrays
            elif isinstance(coord_item_dims, tuple):
                # now, the coordinates must be given as a DXarray...
                if not isinstance(coord_item_coords, DXarray):
                    raise TypeError(
                        "Coordinate arrays (i.e. entries of `coords`) must be DXarrays. Here, type ",
                        type(coord_item_coords),
                        " is given for dimensions ",
                        coord_item_dims,
                        ".",
                    )
                # ... with matching dimension names, ...
                if coord_item_coords.dims != list(coord_item_dims):
                    raise ValueError(
                        "Dimension names of coordinate-DXarray and the corresponding dimension names in `coords` must be equal."
                    )
                # ... shape, ...
                if not (
                    torch.tensor(coord_item_coords.values.gshape)
                    == torch.tensor(values.gshape)[dim_name_to_idx(dims, list(coord_item_dims))]
                ).all():
                    raise ValueError(
                        "Size of `values` in dimensions ",
                        coord_item_dims,
                        " does not coincide with size of coordinate array in these dimensions.",
                    )
                # ... device and communicator, ...
                if not coord_item_coords.device == values.device:
                    raise RuntimeError(
                        "Device of coordinate array for dimensions ",
                        coord_item_dims,
                        "does not coincide with device for `values`.",
                    )
                if not coord_item_coords.comm == values.comm:
                    raise RuntimeError(
                        "Communicator of coordinate array for dimensions ",
                        coord_item_dims,
                        "does not coincide with device for `values`.",
                    )
                # ... and split dimension.
                if dim_idx_to_name(dims, values.split) in coord_item_dims:
                    if not coord_item_coords.split == dim_idx_to_name(dims, values.split):
                        raise ValueError(
                            "`values` array is split along dimension ",
                            coord_item_dims,
                            ", but cooresponding coordinate array is not split along ",
                            coord_item_coords.split,
                            ".",
                        )
                else:
                    if coord_item_coords.split is not None:
                        raise ValueError(
                            "`values` array is not split along dimensions ",
                            coord_item_dims,
                            ", but cooresponding coordinate array is split.",
                        )


def check_name(name: Any):
    """
    Checks whether input is appropriate for attribute `name` of `DXarray`
    """
    if not (isinstance(name, str) or name is None):
        raise TypeError("`name` must be a string or None, but is ", type(name), ".")


def check_attrs(attrs: Any):
    """
    Checks whether input is appropriate for attributed `attrs` of `DXarray`.
    """
    if not (isinstance(attrs, dict) or attrs is None):
        raise TypeError("`attrs` must be a dictionary or None, but is ", type(attrs), ".")


def check_if_balanced(values: ht.DNDarray, coords: Union[dict, None]):
    """
    Checks if a DXarray with values and coords is balanced, i.e., equally distributed on each process
    A DXarray is balanced if and only if all underlying DNDarrays are balanced.
    """
    if values.balanced is None:
        return None
    else:
        if coords is not None:
            if None in [coord_item[1].balanced for coord_item in coords.items()]:
                return None
            else:
                balanced = values.balanced and all(
                    [coord_item[1].balanced for coord_item in coords.items()]
                )
                return balanced
