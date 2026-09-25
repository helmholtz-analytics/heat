"""
Affine transformations for N-dimensional Heat arrays.

This module implements backward-warping affine transformations
(translation, rotation, scaling) for 2D and 3D data stored as
Heat DNDarrays, using a PyTorch backend.

The affine matrix M is interpreted as a *b* transform
in affine (x, y [, z]) coordinates:

    out = A @ inp + b

where M = [A | b] has shape (ND, ND+1).

Internally, the torch functions affine_grid and grid_sample are used

Spatial axis conventions in Heat:
- 2D arrays: (H, W)  == (y, x)
- 3D arrays: (D, H, W) == (z, y, x)

Interpolation and boundary handling:
- order=0: nearest-neighbor
- order=1: bilinear (2D only; 3D falls back to nearest)

Distributed arrays:
- Non-spatial splits are handled locally without communication.
- spatial splits are also hanled locally in case that the affine
  transform does not change anything along that axis
"""

import torch
import heat as ht
from torch.nn.functional import affine_grid, grid_sample
from heat.core.dndarray import DNDarray
from heat.core.factories import array

MODE_TO_PADDING = {
    # SciPy mode               # torch padding_mode
    "grid-constant": "zeros",
    "mirror": "reflection",  # reflect at the middle of the border pixel
    "reflection": "reflection",  # reflect at the middle of the border pixel
    "nearest": "border",  # replicate the edge pixel
    "border": "border",  # replicate the edge pixel
    # The following SciPy modes have no exact Torch counterpart.
    "constant": None,
    "wrap": None,
    "grid-wrap": None,
    "reflect": None,
    "grid-mirror": None,
}

ORDER_TO_MODE = {
    0: "nearest",  # order‑0 → nearest‑neighbour
    1: "bilinear",  # order‑1 → bilinear or trilinear (linear) sampling
    3: "bicubic",  # order‑3 → bicubic sampling
    # SciPy supports orders 2,4,5 as well – they have no direct Torch counterpart.
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


def _to_full_affine(matrix: torch.Tensor):
    # TODO: rewrite this to to_reduced_affine(M) because affine_grid() expects reduced form and this is just converting it to full and back to reduced form
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
    matrix: torch.Tensor, input_shape, padding_correction: bool, output_shape: tuple[int] | None
):
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
    if output_shape is not None:
        output_shape = tuple(size - 1 for size in output_shape[1:-1])
    # construct coord space transform
    scales = (torch.as_tensor(input_shape, device=matrix.device)) / 2.0
    diag_scales = torch.diag(scales)

    dim = len(input_shape)
    conversion = torch.zeros(dim + 1, dim + 1, device=matrix.device)
    conversion[:dim, :dim] = diag_scales
    conversion[dim, dim] = 1
    conversion[:dim, dim] = scales
    back_conversion = conversion.inverse()

    if output_shape:
        out_shape = torch.as_tensor(output_shape + (1,), device=matrix.device)

        # if padding_correction:
        #     padded_sizes = tuple(size + 2 for size in input_shape)
        # else:
        #     padded_sizes = input_shape

        in_shape = torch.as_tensor(input_shape + (1,), device=matrix.device)
        shape_scaling = out_shape / in_shape
        shape_scale_matrix = torch.diag(shape_scaling)
        conversion = shape_scale_matrix @ conversion

    result = back_conversion @ matrix @ conversion

    # reversing effect the padding has on the transform because it changes aspect ratio
    if padding_correction:
        pad_factors = scales / (scales + 1)
        pad_factors = torch.cat([pad_factors, torch.tensor([1], device=matrix.device)])
        pad_matrix = torch.diag(pad_factors)
        result = pad_matrix @ result

    return result[:, :dim, :]


def _untouched_axes(matrices: DNDarray):
    """
    Tests if the provided (affine) has influence on the possible axes of the input
    """
    identity = ht.eye((matrices.shape[-2:]), dtype=matrices.dtype, device=matrices.device)
    comparison = ht.eq(matrices, identity)
    result_row = ht.all(comparison, -1)
    result_column = ht.all(comparison, -2)

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
                f"""the {param_name} '{key_string}' is valid in scipy, but currently
                not supported in this implementation. valid modes are {valid_modes}"""
            )
    else:
        raise ValueError(
            f"given {param_name} '{key_string}' is not a valid valaue. valid modes are {valid_modes}"
        )
    return result


def _prepare_matrix(
    matrix: torch.Tensor,
    offset: torch.Tensor,
    input_shape: tuple[int],
    output_shape: tuple[int],
    is_padded: bool,
) -> torch.Tensor:

    num_transform_dims = len(input_shape) - 2  # subtracting bulk and color axis
    # remove axis that represents transforming the color dimension, because
    # torch affine_grid does not support transforming color dimension
    matrix = _remove_slice(matrix, idx=num_transform_dims, dim=-2)
    matrix = _remove_slice(matrix, idx=num_transform_dims, dim=-1)

    if matrix.shape[1:] == (
        num_transform_dims,
        num_transform_dims,
    ):
        if offset is not None:
            # translation information missing, using offset value and adding it at the right to bring matrix
            # to reduced affine form
            offset_torch = _remove_slice(offset.larray.float(), -1, dim=-1)

            transformed_offset = offset_torch[..., None]
            matrix = torch.cat([matrix, transformed_offset], matrix.ndim - 1)

    if matrix.shape[1:] != (num_transform_dims + 1, num_transform_dims + 1):
        matrix = _to_full_affine(matrix)

    matrix = convert_matrix_space(
        matrix,
        input_shape=input_shape,
        padding_correction=is_padded,
        output_shape=output_shape,
    )
    return matrix


def _sanitize_data(
    input: ht.DNDarray, matrix: ht.DNDarray, offset: ht.DNDarray, output_shape: tuple
):
    ht.sanitize_in(input)
    ht.sanitize_in(matrix)
    if not isinstance(offset, float) and offset is not None:
        ht.sanitize_in(offset)

    if matrix.ndim == 1:
        matrix = ht.diag(matrix)

    if isinstance(offset, float) and offset != 0.0:
        offset_array = ht.repeat(offset, matrix.shape[-2])
    elif isinstance(offset, float) and offset == 0.0:
        offset_array = None
    else:
        offset_array = offset

    if matrix.ndim > 3:
        raise ValueError("affine matrix has too many dimensions")

    if output_shape is not None:
        if not (input.ndim == len(output_shape)):
            raise ValueError("outputshape must have same number of dimension as input")

        if not (input.shape[-1] == output_shape[-1]):
            raise ValueError("color dimension needs same size in input and output shape")
    else:
        output_shape = input.shape

    # input has no bulk axis, give everything a bulk axis with length 1 to treat it as if it has a bulk axis
    if matrix.ndim == 2:
        matrix = ht.expand_dims(matrix, 0)
        input = ht.expand_dims(input, 0)
        if offset_array is not None:
            offset_array = ht.expand_dims(offset_array, 0)
        output_shape = (1,) + output_shape

    is_2d_input = input.ndim == 4 and 3 <= matrix.shape[-2] <= matrix.shape[-1] <= 4
    is_3d_input = input.ndim == 5 and 4 <= matrix.shape[-2] <= matrix.shape[-1] <= 5
    if not (is_2d_input or is_3d_input):
        raise ValueError(
            f"matrix with shape {matrix.shape} does not fit to input shape {input.shape} or not supported dimension count"
        )

    if not (input.shape[0] == output_shape[0]):
        raise ValueError("bulk dimension needs same size in input and output shape")

    # offset exists and matrix is no affine matrix
    if offset_array is not None:
        if matrix.shape[-1] >= input.ndim:
            offset_array = None
            import warnings

            warnings.warn(
                UserWarning(
                    "offset is not used, since matrix provides offset information in its rightmost column"
                )
            )
        else:
            if offset_array.ndim != matrix.ndim - 1:
                raise ValueError(f"""offset vector has wrong number of dimensions compared to the matrix.
                expected {matrix.ndim - 1} dimensions but got {offset_array.ndim}""")
            if offset_array.shape[-1] != matrix.shape[-2]:
                raise ValueError(
                    f"offset vector has not the right length, expected {matrix.shape[-2]}, but got {offset_array.shape[-1]}"
                )

    # determening the split axis
    # if axis is not the bulk axis or constant axis -> abort
    # should be fine because matrix should always be small compared to image input
    if matrix.split not in (0, None):
        matrix = ht.resplit(matrix, None)

    if input.split not in (0, None):
        # transformation in direction of split is not identity
        if not (matrix.split is None and _untouched_axes(matrix)[input.split - 1]):
            raise RuntimeError(
                "the input split axis should either be the bulk axis, or an axis left unchanged by the transform."
            )
        # input split along non-bulk axis only supported if matrix is not split
        if matrix.split is not None:
            matrix = ht.resplit(matrix, None)

        if output_shape[input.split] != input.shape[input.split]:
            raise RuntimeError(
                "the output shape cannot differ form input shape along input split axis"
            )

    if offset_array is not None:
        if offset_array.split != matrix.split:
            offset_array = ht.resplit(matrix.split)

    return input, matrix, offset_array, output_shape


def _get_local_torch_views(input: DNDarray, matrix: DNDarray, output_shape: tuple[int]):
    matrix_torch: torch.Tensor = matrix.larray
    input_torch = input.larray

    if matrix.split is None and input.split == 0:
        _, _, corresponding_slice = matrix.comm.chunk(matrix.gshape, 0)
        matrix_torch = matrix_torch[corresponding_slice]
        out_split = 0
    elif matrix.split == 0 and input.split is None:
        _, _, corresponding_slice = input.comm.chunk(input.gshape, 0)
        input_torch = input_torch[corresponding_slice]
        out_split = 0
    else:
        out_split = input.split

    if matrix_torch.device != input_torch.device:
        matrix_torch = matrix_torch.to(input_torch.device)

    if out_split is not None:
        _, output_shape, _ = matrix.comm.chunk(output_shape, out_split)

    return input_torch, matrix_torch, output_shape, out_split


# ============================================================
#  main methods
# ============================================================
def affine_transform(
    input: DNDarray,
    matrix: DNDarray,
    offset: DNDarray | float | None = 0.0,
    output_shape: DNDarray | None = None,
    order: int = 1,
    mode: str = "grid-constant",
    cval: float = 0.0,
) -> DNDarray:
    """
    Parameters
    ----------
    input : DNDarray
        the image or data array to transform. Input is expected to have shape [B x] [D x] H x W x C
    matrix : DNDarray
        affine matrix used to transform input. can be of shape Bx3x4 (3x3, 4x4 also valid) for 2d data,
        or should be of shape Bx4x5 (4x4, 5x5 also valid) for 3d data
        B stands for the Bulk axis and can be ommited
        The row and column corresponding with Transformation of the Color-Axis (the last row and column) are required to be there, but is ignored right now!,
        This is done as tradeof between compatibility with scipy.affine_transform based on the assumtion that transforming of the color axis is rarely desired.
        If your input is a 2D image you can add a singleton dimension to the end of the matrix to treat the color axis as a spacial axis: H x W x C -> H x W x C x 1
    offset : DNDarray
        offset vector that can be used instead of adding offset into affine matrix directly. only in effect when the matrix
        given has no transform vector
    output_shape :
        shape of the given output. It should map the pattern [B x] [D x] H x W x C wich is the same as the input. D, H, W can have different values than the input
    order :
        type of interpolation that is used, linear to cubic allowed
    mode :
        The mode parameter determines how the input array is extended beyond its boundaries. Default is ‘constant-grid’. Behavior for each valid value is as follows
        `grid-constant`
            the pixel beyond the boundary are filled with a constant value. The value is defined by the cval parameter
        `nearest`
            the pixel beyond the boundary are filled by replicating the pixel on the nearest border
        `mirror`
            the pixel beyond the boundary are filled by mirroring the the input around the center of the last pixel
        `constant`, `wrap`, `grid-wrap`, `reflect`, `grid-mirror`
            Those modes are not implemented
    cval :
        value with wich the padding should be filled. This is implemented as a padding applied along all axis. This approach is not suited to provide exact results
    prefilter : bool
        if the input should be filtered before transformed, currently not because torch.sample_grid does not have this functionality
    """
    # input conversion
    if mode == "constant":
        raise NotImplementedError(
            "constant mode is not implemented, use 'grid-constant' for similar result"
        )
    apply_cval_padding = mode == "grid-constant" and cval != 0
    if apply_cval_padding:
        sample_padding = "border"
    else:
        sample_padding = _sanitize_key_string(mode, MODE_TO_PADDING, "mode")
    sample_mode = _sanitize_key_string(order, ORDER_TO_MODE, "order")

    original_shape = input.shape

    input, matrix, offset, output_shape = _sanitize_data(input, matrix, offset, output_shape)

    linput, lmatrix, out_lshape, out_split = _get_local_torch_views(input, matrix, output_shape)
    # at this point computations can all be done locally

    # I still don't understand why the permute below is necessary, but the affine matrix itself
    # does not need to be reordered.
    # This should switche wich axis in input data is influenced by wich row/column in the
    # affine matrix. But the result matches scipy so thats so I guess it's fine
    dimension_order = (0,) + tuple(
        idx for idx in range(input.ndim - 1, 0, -1)
    )  # reversed: (0,4,3,2,1) or (0,3,2,1)

    loc_out_shape_permuted = tuple(out_lshape[i] for i in dimension_order)
    out_shape = torch.Size(loc_out_shape_permuted)

    # skip computation if this rank has no data
    if linput.numel() > 0:
        lmatrix = _prepare_matrix(lmatrix, offset, linput.shape, out_lshape, apply_cval_padding)

        linput = linput.permute(dimension_order)

        # TODO exclude padding at rank boundaries, should never be neccessary
        if apply_cval_padding:
            padding_size = tuple(1 for _ in range((linput.ndim - 2) * 2))
            linput = torch.nn.functional.pad(linput, padding_size, "constant", cval)

        sample_grid: torch.Tensor = affine_grid(lmatrix, out_shape, align_corners=True)

        transformed: torch.Tensor = grid_sample(
            linput,
            sample_grid,
            padding_mode=sample_padding,
            mode=sample_mode,
            align_corners=True,
        )
    else:
        transformed = torch.empty(
            out_shape, dtype=linput.dtype, device=linput.device
        )  # device=input.larray.device, dtype=input.larray.dtype)

    transformed = transformed.permute(dimension_order)

    if transformed.ndim != len(original_shape):  # had no bulk axis originally
        transformed = transformed.squeeze(0)
        if out_split is not None:
            out_split -= 1

    transformed_dnd: DNDarray = array(
        transformed.contiguous(),
        dtype=input.dtype,
        is_split=out_split,
        device=input.device,
        comm=input.comm,
        copy=False,
    )

    return transformed_dnd
