"""
Affine transformations for N-dimensional Heat arrays.

This module implements backward-warping affine transformations
(translation, rotation, scaling) for 2D and 3D data stored as
Heat DNDarrays, using a PyTorch backend relying on affine_grid and grid_sample from torch.nn.functional.

Spatial axis conventions in Heat:
- 2D arrays: (H, W)  == (y, x)
- 3D arrays: (D, H, W) == (z, y, x)

Handling of Distributed arrays:
- Non-spatial splits are handled locally without communication.
- spatial splits are also hanled locally in case that the affine
  transform does not change anything along that axis
"""

import warnings
from typing import Sequence
import torch
from torch.nn.functional import affine_grid, grid_sample
import heat as ht
from heat.core import types
from heat.core.dndarray import DNDarray
from heat.core.factories import array
from heat.core.sanitation import sanitize_in_min_max_nd
from heat.core.communication import Communication

MODE_TO_PADDING = {
    # scipy mode : torch padding_mode
    "grid-constant": "zeros",  # fill with black
    "zeros": "zeros",
    "mirror": "reflection",  # reflect at the middle of the border pixel
    "reflection": "reflection",  # allow also torch identifier directly
    "nearest": "border",  # replicate the edge pixel
    "border": "border",  # allow also torch identifier directly
    # The following SciPy modes have no exact Torch counterpart.
    "constant": None,
    "wrap": None,
    "grid-wrap": None,
    "reflect": None,
    "grid-mirror": None,
}

ORDER_TO_MODE = {
    # scipy order : torch (interpolation)mode
    0: "nearest",  # nearest‑neighbour
    1: "bilinear",  # can also mean trilinear depending on image dimension count
    3: "bicubic",  # tricubic not supported by torch
    # SciPy supports orders 2,4,5 as well, but they have no direct Torch counterpart.
    2: None,
    4: None,
    5: None,
}


def _remove_slice(tensor: torch.Tensor, idx: int, dim: int) -> torch.Tensor:
    # Keep rows before and after the removed row
    if idx < 0:
        idx = tensor.shape[dim] + idx

    rows_before = torch.arange(0, idx, device=tensor.device)
    if dim < 0:
        dim = tensor.ndim + dim

    size = tensor.size(dim)

    if (idx + 1) < size:
        rows_after = torch.arange(idx + 1, size, device=tensor.device)
        return torch.cat(
            [
                tensor.index_select(dim, rows_before),
                tensor.index_select(dim, rows_after),
            ],
            dim=dim,
        )

    return tensor.index_select(dim, rows_before)


def _to_full_affine(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert reduced affine matrices to full homogeneous form.
    Treats last 2 dimensions as matrix dimensions, the others are treated as batch dimensions
        - [...]DxD   -> D+1xD+1
        - [...]DxD+1 -> D+1xD+1
    """
    target_shape = list(matrix.shape)
    if len(target_shape) < 2:
        raise ValueError(
            f"Expected affine transformation matrix to be tensor with at leas 2 dimensions, got {matrix.dim()}"
        )

    if target_shape[-1] == target_shape[-2]:  # square matrix
        target_shape[-1] += 1
        target_shape[-2] += 1
    else:
        target_shape[-2] = target_shape[-1]

    dim = target_shape[-1]

    full = torch.zeros(target_shape, dtype=matrix.dtype, device=matrix.device)
    full[..., : matrix.shape[-2], : matrix.shape[-1]] = matrix  # copy matrix into new matrix
    full[..., dim - 1, dim - 1] = 1.0  # set homogeneous coordinate
    return full


def convert_matrix_space(
    matrix: torch.Tensor,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    padding_correction: bool,
) -> torch.Tensor:
    """
    Takes a full scipy affine matrix and converts the space to normalized coordinates used by affine_grid.
    scipy uses pixel coordinate space with origin in the top left,
    while affine_grid uses -1 to 1 with origin in the center of the image.
    Dependant on applied padding and provided output_shape there are additional correction applied to invert the effects on output pixel locations

    output_shape should have the same dimensions number of dimension as sizes
    """
    input_shape = tuple(
        size - 1 for size in input_shape[1:-1]
    )  # -1: correcting for image corner in pixel center
    output_shape = tuple(size - 1 for size in output_shape[1:-1])
    # construct coord space transform
    offsets = (torch.as_tensor(input_shape, device=matrix.device, dtype=matrix.dtype)) / 2.0
    scales = torch.where(offsets == 0, 1, offsets)
    diag_scales = torch.diag(scales)

    dim = len(input_shape)
    conversion = torch.zeros(dim + 1, dim + 1, device=matrix.device, dtype=matrix.dtype)
    conversion[:dim, :dim] = diag_scales
    conversion[dim, dim] = 1
    conversion[:dim, dim] = offsets
    back_conversion = conversion.inverse()

    if output_shape != input_shape:
        # when either shape have 0 add one to both input and output value
        input_shape, output_shape = zip(
            *tuple(
                (in_dim + 1, out_dim + 1) if in_dim == 0 or out_dim == 0 else (in_dim, out_dim)
                for out_dim, in_dim in zip(output_shape, input_shape)
            )
        )

        out_shape = torch.as_tensor(output_shape + (1,), device=matrix.device, dtype=matrix.dtype)
        in_shape = torch.as_tensor(input_shape + (1,), device=matrix.device, dtype=matrix.dtype)
        shape_scaling = out_shape / in_shape
        shape_scale_matrix = torch.diag(shape_scaling)
        conversion = shape_scale_matrix @ conversion

    result = back_conversion @ matrix @ conversion

    # reversing effect the padding has on the transform because it changes aspect ratio
    if padding_correction:
        pad_factors = scales / (scales + 1)
        pad_factors = torch.cat(
            [pad_factors, torch.ones(1, device=matrix.device, dtype=matrix.dtype)]
        )
        pad_matrix = torch.diag(pad_factors)
        result = pad_matrix @ result

    return result[:, :dim, :]


def _untouched_axes(matrices: DNDarray) -> DNDarray:
    """
    Tests if the provided (affine) has influence on the possible axes of the input
    """
    identity = ht.eye((matrices.shape[-2:]), dtype=matrices.dtype, device=matrices.device)
    comparison = ht.eq(matrices, identity)
    result_row = ht.all(comparison, axis=-1)
    result_column = ht.all(comparison, axis=-2)

    if result_column.shape[-1] > result_row.shape[-1]:
        result = ht.logical_and(result_row, result_column[..., : result_row.shape[-1]])
    else:
        result = ht.logical_and(result_row, result_column)

    if result.ndim > 1:
        result = ht.all(result, 0)  # combine along bulk axis
    return result


def _sanitize_key_string(key_string: str, dictionary: dict, param_name: str):
    valid_modes = [key for key, value in dictionary.items() if value is not None]
    if key_string in dictionary:
        result = dictionary[key_string]
        if result is None:
            raise NotImplementedError(
                f"the {param_name} '{key_string}' is valid in scipy, but currently"
                f"not supported in this implementation. valid modes are {valid_modes}"
            )
    else:
        raise ValueError(
            f"given {param_name} '{key_string}' is not a valid value. valid modes are {valid_modes}"
        )
    return result


def _promote_to_type(goal_type: types.datatype, *arrays: DNDarray | None):
    """
    Promotes every array in arrays to type if it is possible, else throw an error.
    None Values in arrays do not get touched and get returned in the same order
    """
    promoted_type = goal_type
    for a in arrays:
        if a is not None:
            promoted_type = types.promote_types(promoted_type, a.dtype)
    if promoted_type is not goal_type:
        raise TypeError(
            "arrays cant be promoted to goal_type, bacause they contain"
            "a Type that is higher in promotion order than the goal type"
        )
    return tuple(
        None if a is None else (a if a.dtype == promoted_type else a.astype(promoted_type))
        for a in arrays
    )


def _prepare_matrix(
    matrix: torch.Tensor,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    is_padded: bool,
) -> torch.Tensor:

    num_transform_dims = len(input_shape) - 2  # subtracting bulk and color axis
    # remove axis that represents transforming the color dimension, because
    # torch affine_grid does not support transforming color dimension
    matrix = _remove_slice(matrix, idx=num_transform_dims, dim=-2)
    matrix = _remove_slice(matrix, idx=num_transform_dims, dim=-1)

    if matrix.shape[1:] != (num_transform_dims + 1, num_transform_dims + 1):
        matrix = _to_full_affine(matrix)

    matrix = convert_matrix_space(
        matrix,
        input_shape=input_shape,
        output_shape=output_shape,
        padding_correction=is_padded,
    )
    return matrix


def _prepare_data(
    image: DNDarray, matrix: DNDarray, offset: DNDarray, output_shape: Sequence[int] | None
) -> tuple[DNDarray, DNDarray, DNDarray, tuple[int, ...]]:
    if matrix.ndim == 1:
        matrix = ht.diag(matrix)

    sanitize_in_min_max_nd(image, 3, 5)
    if image.dtype not in [ht.float32, ht.float64]:
        raise TypeError(
            f"Argument image needs to be a DNDarray with datatype float32 or float64, but data type is {image.dtype}."
        )

    sanitize_in_min_max_nd(matrix, 2, 3)

    if isinstance(offset, (float, int)):
        if offset == 0:
            offset_array = None
        else:
            offset_array: DNDarray = ht.array((offset,), dtype=matrix.dtype, device=image.device)
            if matrix.ndim == 2:
                offset_array = ht.tile(offset_array, matrix.shape[-2])
            elif matrix.ndim == 3:
                offset_array = ht.tile(offset_array, (matrix.shape[0], matrix.shape[-2]))
    else:
        if offset is not None:
            sanitize_in_min_max_nd(offset, 1, 2)
        offset_array = offset

    if output_shape is not None:
        ht.sanitize_sequence(output_shape)
        out_shape = tuple(output_shape)
        if not (image.ndim == len(out_shape)):
            raise ValueError("outputshape must have same number of dimension as input")

        if not (image.shape[-1] == out_shape[-1]):
            raise ValueError("color dimension needs same size in input and output shape")
    else:
        out_shape = image.shape

    # type promotion: image type should not be changed, matrix and offset type can be promoted to image type
    # because image can be big and should not be copied implicetly
    try:
        matrix, offset_array = _promote_to_type(image.dtype, matrix, offset_array)
    except TypeError as exc:
        raise TypeError("not all types can be converted to the image type without losses") from exc

    # input has no bulk axis, give everything a bulk axis with length 1 to treat it as if it has a bulk axis
    if matrix.ndim == 2:
        matrix = ht.expand_dims(matrix, 0)
        image = ht.expand_dims(image, 0)
        if offset_array is not None:
            offset_array = ht.expand_dims(offset_array, 0)
        out_shape = (1,) + out_shape

    is_2d_input = image.ndim == 4 and 3 <= matrix.shape[-2] <= matrix.shape[-1] <= 4
    is_3d_input = image.ndim == 5 and 4 <= matrix.shape[-2] <= matrix.shape[-1] <= 5
    if not (is_2d_input or is_3d_input):
        raise ValueError(
            f"matrix with shape {matrix.shape} does not fit to input shape {image.shape} or not supported dimension count"
        )

    if not (image.shape[0] == out_shape[0]):
        raise ValueError("bulk dimension needs same size in input and output shape")

    if offset_array is not None:
        # offset exists and matrix is no affine matrix
        if matrix.shape[-1] >= image.ndim:
            offset_array = None
            warnings.warn(
                UserWarning(
                    "offset is not used, since matrix provides offset information in its rightmost column"
                )
            )
        else:
            if offset_array.ndim != matrix.ndim - 1:
                raise ValueError(
                    f"offset vector has wrong number of dimensions compared to the matrix."
                    f"expected {matrix.ndim - 1} dimensions but got {offset_array.ndim}"
                )
            if offset_array.shape[-1] != matrix.shape[-2]:
                raise ValueError(
                    f"offset vector has not the right length, expected {matrix.shape[-2]},"
                    f"but got {offset_array.shape[-1]}"
                )

            offset_array = ht.expand_dims(offset_array, offset_array.ndim)
            matrix = ht.concatenate([matrix, offset_array], axis=(matrix.ndim - 1))

    return image, matrix, out_shape


def _sanitize_and_update_splitting(
    image_split: int | None,
    matrix: DNDarray,
) -> tuple[DNDarray, DNDarray | None]:
    # determening the split axis
    # if axis is not the bulk axis or constant axis -> abort
    # should be fine because matrix should always be small compared to image input
    if matrix.split not in (0, None):
        matrix = ht.resplit(matrix, None)

    if image_split not in (0, None):
        # transformation in direction of split is not identity
        if not (matrix.split is None and _untouched_axes(matrix)[image_split - 1]):
            raise RuntimeError(
                "the input split axis should either be the bulk axis, or an axis left unchanged by the transform."
            )
        # input split along non-bulk axis only supported if matrix is not split
        if matrix.split is not None:
            matrix = ht.resplit(matrix, None)

    return matrix


def _handle_distribution_balance(
    image: DNDarray,
    matrix: DNDarray,
):
    if image.is_balanced() and matrix.is_balanced():
        matrix.resplit_(None)
    elif (
        image.is_balanced()
        and not matrix.is_balanced()
        or not image.is_balanced()
        and matrix.is_balanced()
    ):
        matrix.resplit_(None)
    elif not image.is_balanced() and not matrix.is_balanced():
        im_counts, _ = image.counts_displs()
        ma_counts, _ = matrix.counts_displs()
        if im_counts != ma_counts:
            matrix.resplit_(None)
    else:
        return


def _get_local_torch_views(
    image: DNDarray, matrix: DNDarray, output_shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], int, Communication, bool]:
    matrix_torch: torch.Tensor = matrix.larray
    image_torch = image.larray

    if matrix.split is None and image.split == 0:
        counts, displ = image.counts_displs()
        start_index = displ[image.comm.rank]
        end_index = start_index + counts[image.comm.rank]
        matrix_torch = matrix_torch[start_index:end_index]
        out_split = 0
        out_comm = image.comm
        balanced = image.is_balanced()
    elif matrix.split == 0 and image.split is None:
        counts, displ = matrix.counts_displs()
        start_index = displ[matrix.comm.rank]
        end_index = start_index + counts[matrix.comm.rank]
        image_torch = image_torch[start_index:end_index]
        out_split = 0
        out_comm = matrix.comm
        balanced = matrix.is_balanced()
    else:
        # split should be the same between matrix and image
        out_split = image.split
        out_comm = image.comm
        balanced = image.is_balanced()

    if matrix_torch.device != image_torch.device:
        matrix_torch = matrix_torch.to(image_torch.device)

    if out_split is not None:
        out_lshape = (
            output_shape[0:out_split]
            + (image_torch.shape[out_split],)
            + output_shape[out_split + 1 :]
        )
    else:
        out_lshape = output_shape

    return image_torch, matrix_torch, out_lshape, out_split, out_comm, balanced


def _sanitize_padding(mode: str, padding_mode: str, cval: float) -> tuple[str, bool]:
    """
    Unify redundant inputs mode and padding_mode, include behaviour of cval into padding_mode
    """
    if mode is None and padding_mode is None:
        sample_padding = "zeros"  # default value
    else:
        if mode is not None and padding_mode is not None:
            raise ValueError("mode and padding_mode can't be set at the same time")

        if mode is not None:
            padding_input = mode
        else:
            padding_input = padding_mode

        if padding_input == "constant":
            raise NotImplementedError(
                "constant mode is not implemented, use 'grid-constant' for similar result"
            )
        sample_padding = _sanitize_key_string(
            padding_input, MODE_TO_PADDING, "mode or padding_mode"
        )

    apply_cval_padding = sample_padding == "zeros" and cval != 0
    if apply_cval_padding:
        sample_padding = "border"

    return sample_padding, apply_cval_padding


# ============================================================
#  main methods
# ============================================================
def affine_transform(
    image: DNDarray,
    matrix: DNDarray,
    offset: DNDarray | float | None = 0.0,
    output_shape: Sequence[int] | None = None,
    order: int = 3,
    mode: str | None = None,
    padding_mode: str | None = None,
    cval: float = 0.0,
) -> DNDarray:
    """
    Implementation of an affine transform following the signature of the scipy version with the same name.
    Additionally the Option of an additional bulk axis is added. Because the function exclusively uses the
    torch functions affine_grid and grid_sample, several limitations apply. The limitations for each
    parameter are mentioned in their discription

    Parameters
    ----------
    image : DNDarray
        the image or data array to transform. Input is expected to have shape [B x ][ D x ]H x W x C
    matrix : DNDarray
        affine matrix used to transform input
        Should be of shape [Bx]N+1xN+1, [Bx]NxN+1 or [Bx]NxN for affine, reduced affine or linear transform
        without axis information. B is an optional bulk axis and has to match image bulk axis.
        Per scipy convention, the matrix has a row and column for the color axis (bottom row and rightmost
        column). As a Tradeoff for compatibility with scipy they are required to be there, but are **ignored**
        right now. This is based on the assumtion that transforming of the color axis is rarely desired, and
        most implementations using scipy cab be switched without differing output.
        If your input is a 2D image you can add a singleton dimension to the end of the matrix to treat the
        color axis as a spacial axis: H x W x C -> H x W x C x 1
        The Transformation is centered on the top left corner of the image and the image is considered to
        have the size [ D x ]H x W by the transform
    offset : DNDarray | float | None
        offset vector that can be used instead of adding offset into affine matrix directly. This is only used
        when the matrix given is of shape [Bx]NxN and has no column for offset information. instead of a vector,
        offset can also be a float, wich then gets applied to all axis of the input. Different to scipy, the
        color axis remains untouched.
    output_shape : tuple[int, ...]
        shape of the output the function will produce. It should have the same dimension count as the input,
        only the size of spacial dimensions can be varied, color and bulk axis need to stay the same.
    order :
        type of interpolation that is used, linear(1) or cubic(3) are valid values. Cubic interpolation does
        only work with 2d Images
    mode : str
        Determines how the input array is extended beyond its boundaries. Default value is `constant-grid`.
        `constant`, the default value used by scipy is not supported and will throw an error.
        Behavior for each valid value is as follows
        - `grid-constant`
            the pixel beyond the boundary are filled with a constant value. The value is defined by the cval
            parameter. Border pixels are also interpolated.
        - `nearest`
            the pixel beyond the boundary are filled by replicating the pixel on the nearest border
        - `mirror`
            the pixel beyond the boundary are filled by mirroring the the input around the center of the last
            pixel
        - `zeros`
            same as `grid-constant`, alias using torch semantics
        - `border`
            same as `nearest`, alias using torch semantics
        - `reflection`
            same as `mirror`, alias using torch semantics
    padding_mode :
        alias for 'mode' with same behaviour. Either 'mode' or 'padding_mode' can be provided, not both
    cval :
        value with wich the padding should be filled. This is implemented as a padding applied along all axis.

    Notes
    -----
    - image, matrix and offset cannot be broadcasted because their shapes are needed to determine if a bulk axis
      has been provided and if matrix is provided as affine, reduced affine or linear transformation matrix.
    - 'image' is required to have dtpype 'float32' or 'float64'. The types of 'matrix' and 'offset' can be of a
      type that is convertible into the type of 'image', but need to be lower in promotion order, e.g. 'image'
      of type 'float32' and 'matrix' of type 'float64' would be not allowed.
    - This function does work with distributed arrays, but it will not do any distributed computation. So a
      transformation that does not need any information from any other distributed slice would work as intended
      and in parallel.
    - The function uses a backwarts warping affine transform, wich means the matrix transforms the coordinate
      system and not the image itself. To get  the behaviour of a forward warping affine transfrom, invert
      your affine matrix.
    """
    padding_mode, apply_cval_padding = _sanitize_padding(mode, padding_mode, cval)
    sample_mode = _sanitize_key_string(order, ORDER_TO_MODE, "order")

    in_gshape = image.shape

    image, matrix, out_shape = _prepare_data(image, matrix, offset, output_shape)

    matrix = _sanitize_and_update_splitting(image.split, matrix)

    if image.split is not None and out_shape[image.split] != image.shape[image.split]:
        raise RuntimeError("the output shape cannot differ form input shape along input split axis")

    _handle_distribution_balance(image, matrix)
    linput, lmatrix, out_lshape, out_split, out_comm, out_balanced = _get_local_torch_views(
        image, matrix, out_shape
    )

    # at this point computations can all be done locally

    # I still don't understand why the permute below is necessary, but the affine matrix itself
    # does not need to be reordered.
    # This should switch wich axis in input data is influenced by wich row/column in the
    # affine matrix. But the result matches scipy so I guess it's fine
    dimension_order = (0,) + tuple(
        idx for idx in range(image.ndim - 1, 0, -1)
    )  # reversed: (0,4,3,2,1) or (0,3,2,1)

    loc_out_shape_permuted = tuple(out_lshape[i] for i in dimension_order)
    out_lshape_size = torch.Size(loc_out_shape_permuted)

    # skip computation if this rank has no data
    if linput.numel() > 0:
        lmatrix = _prepare_matrix(lmatrix, linput.shape, out_lshape, apply_cval_padding)

        linput = linput.permute(dimension_order)

        if apply_cval_padding:
            padding_size = tuple(1 for _ in range((linput.ndim - 2) * 2))
            linput = torch.nn.functional.pad(linput, padding_size, "constant", cval)

        sample_grid: torch.Tensor = affine_grid(lmatrix, out_lshape_size, align_corners=True)
        transformed: torch.Tensor = grid_sample(
            linput,
            sample_grid,
            padding_mode=padding_mode,
            mode=sample_mode,
            align_corners=True,
        )
    else:
        transformed = torch.empty(out_lshape_size, dtype=linput.dtype, device=linput.device)

    transformed = transformed.permute(dimension_order)

    if transformed.ndim != len(in_gshape):  # had no bulk axis originally
        transformed = transformed.squeeze(0)
        if out_split is not None:
            out_split -= 1
        out_shape = out_shape[1:]

    transformed_dnd: DNDarray = DNDarray(
        transformed.contiguous(),
        dtype=image.dtype,
        gshape=out_shape,
        split=out_split,
        device=image.device,
        comm=out_comm,
        balanced=out_balanced,
    )

    return transformed_dnd
