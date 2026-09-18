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

filtering_map = {}


# ============================================================
# Helper utilities
# ============================================================


def _remove_slice(tensor: torch.Tensor, idx: int, dim: int) -> torch.Tensor:
    # Keep rows before and after the removed row
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

    Works with single matrices or batches:
        - (D, D+1)           → (D+1, D+1)    # single
        - (N, D, D+1)        → (N, D+1, D+1) # batch
    """
    # Detect if batched by checking number of dimensions
    if matrix.dim() == 2:
        # Single matrix case: (D, D+1)
        D = matrix.shape[0]  # spatial dimension
        full = torch.zeros(D + 1, D + 1, dtype=matrix.dtype, device=matrix.device)
        full[:D, :] = matrix  # copy top D rows
        full[D, D] = 1.0  # set homogeneous coordinate
        return full

    if matrix.dim() == 3:
        # Batched case: (N, D, D+1)
        N, D, _ = matrix.shape
        full = torch.zeros(N, D + 1, D + 1, dtype=matrix.dtype, device=matrix.device)
        full[:, :D, :_] = matrix  # copy top D rows for each batch
        full[:, D, D] = 1.0  # set homogeneous coordinate for each batch
        return full

    raise ValueError(
        f"Expected affine transformation matrix to be 2D or 3D tensor, got {matrix.dim()}D"
    )


def convert_matrix_space(
    matrix: torch.Tensor, input_shape, padding_correction: bool, output_shape: tuple[int] | None
):
    """
    Convert scipy affine matrix to normalized coordinates used by affine_grid.
    scipy uses pixel coordinate space with origin in the top left,
    while affine_grid uses -1 to 1 with origin in the center of the image. dependant on applied padding
    and provided output_shape there are additional correction applied to invert the effects on output pixel locations

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


# ============================================================
#  main methods
# ============================================================
def affine_transform(
    input: DNDarray,
    matrix: DNDarray,
    offset=None,
    output_shape=None,
    output=None,
    order=1,
    mode="grid-constant",
    cval=0.0,
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
        The row and column corresponding with Transformation of the Color-Axis is ignored right now,
        because it is not supported by the torch.affine_grid() function.
    offset : DNDarray
        offset vector that can be used instead of adding offset into affine matrix directly. only in effect when the matrix
        given has no transform vector
    output_shape :
        shape of the given output. It should map the pattern [B x] [D x] H x W x C wich is the same as the input. D, H, W can have different values than the input
    output : DNDarray
        optional parameter to specify array in wich the output should be placed. currently not implemented yet
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
    # TODO Support both 'padding' parameter from the torch functions
    # validation

    ht.sanitize_in(input)
    ht.sanitize_in(matrix)
    # TODO move this to offset logic, does not need to happen when affine_matrix contains offset information
    if offset is not None:
        ht.sanitize_in(offset)

    # input conversion
    if mode == "constant":
        raise NotImplementedError(
            "constant mode is not implemented, use 'grid-constant' for similar result"
        )
    elif mode == "wrap" or mode == "grid-wrap":
        raise NotImplementedError(f"{mode} is not implemented")

    apply_cval_padding = mode == "grid-constant" and cval != 0

    if apply_cval_padding:
        sample_padding = "border"
    else:
        sample_padding = MODE_TO_PADDING[mode]

    sample_mode = ORDER_TO_MODE[order]

    if matrix.ndim > 3:
        raise ValueError("afine matrix has too many dimensions")

    if output_shape is not None:
        if not (input.ndim == len(output_shape)):
            raise ValueError("outputshape must have same number of dimension as input")

        if not (input.shape[-1] == output_shape[-1]):
            raise ValueError("color dimension needs same size in input and output shape")
    else:
        output_shape = input.shape
    original_shape = input.shape

    # input has no bulk axis, give everything a bulk axis with length 1 to treat it as if it has a bulk axis
    if matrix.ndim == 2:
        matrix = ht.expand_dims(matrix, 0)
        input = ht.expand_dims(input, 0)
        if offset is not None:
            offset = ht.expand_dims(offset, 0)
        if output_shape is not None:
            output_shape = (1,) + output_shape

    if (output_shape is not None) and (not (input.shape[0] == output_shape[0])):
        raise ValueError("bulk dimension needs same size in input and output shape")

    is_2d_input = input.ndim == 4 and 3 <= matrix.shape[1] <= matrix.shape[2] <= 4
    is_3d_input = input.ndim == 5 and 4 <= matrix.shape[1] <= matrix.shape[2] <= 5
    if not (is_2d_input or is_3d_input):
        raise ValueError(
            f"matrix with shape {matrix.shape} does not fit to input shape {input.shape} or not supported dimension count"
        )

    # determening the split axis
    # if axis is not the bulk axis or constant axis -> abort
    if (matrix.split not in (0, None)) or (input.split not in (0, None)):
        if input.split > 0:
            if not (
                matrix.split is None and _untouched_axes(matrix)[input.split - 1]
            ):  # transformation in direction of split is not identity
                raise RuntimeError(
                    "the input split axis should either be the bulk axis, or an axis left unchanged by the transform."
                )
        else:
            raise RuntimeError("matrix split axis should only be 0 if input split axis is also 0")

    # only bulk axis or distributed axis is fixed, computations can all be done locally
    matrix_torch: torch.Tensor = matrix.larray
    input_torch = input.larray

    if matrix.split is None and input.split == 0:
        _, _, corresponding_slice = matrix.comm.chunk(matrix.gshape, 0)
        matrix_torch = matrix_torch[corresponding_slice]
    elif matrix.split == 0 and input.split is None:
        _, _, corresponding_slice = input.comm.chunk(input.gshape, 0)
        input_torch = input_torch[corresponding_slice]

    if matrix_torch.device != input_torch.device:
        matrix_torch = matrix_torch.to(input_torch.device)

    color_dim = 2 if is_2d_input else 3  # TODO this does not look robust, fix it please
    homogenous_size = color_dim + 2

    # remove axis that represents transforming the color dimension, because
    # torch affine_grid does not support transforming color dimension
    matrix_torch = _remove_slice(matrix_torch, color_dim, dim=-2)
    matrix_torch = _remove_slice(matrix_torch, color_dim, dim=-1)

    if (
        matrix_torch.shape[1:]
        == (
            color_dim,
            color_dim,
        )
        and offset is not None
    ):
        # translation information missing, using offset value and adding it at the right to bring matrix
        # to reduced affine form
        offset_torch = _remove_slice(offset.larray.float(), color_dim, dim=-1)

        transformed_offset = offset_torch[..., None]
        matrix_torch = torch.cat([matrix_torch, transformed_offset], matrix_torch.ndim - 1)

    if matrix.shape != (homogenous_size, homogenous_size):
        matrix_torch = _to_full_affine(matrix_torch)

    # I still don't understand why the permute below is necessary, but the affine matrix itself
    # does not need to be reordered.
    # This should switche wich axis in input data is influenced by wich row/column in the
    # affine matrix. But the result matches scipy so thats so I guess it's fine
    dimension_order = (0,) + tuple(
        idx for idx in range(input.ndim - 1, 0, -1)
    )  # reversed: (0,4,3,2,1) or (0,3,2,1)

    if input.split is not None:
        _, local_out_shape, _ = matrix.comm.chunk(output_shape, input.split)
    else:
        local_out_shape = output_shape

    matrix_torch = convert_matrix_space(
        matrix_torch,
        input_torch.shape,
        padding_correction=apply_cval_padding,
        output_shape=local_out_shape,
    )

    local_out_shape = tuple(local_out_shape[i] for i in dimension_order)
    input_torch = input_torch.permute(dimension_order)
    out_shape = torch.Size(local_out_shape)

    # skip computation if this rank has no data
    if input_torch.numel() > 0:
        # TODO exclude padding at rank boundaries, should never be neccessary
        if apply_cval_padding:
            padding_size = tuple(1 for _ in range((input_torch.ndim - 2) * 2))
            input_torch = torch.nn.functional.pad(input_torch, padding_size, "constant", cval)

        sample_grid: torch.Tensor = affine_grid(matrix_torch, out_shape, align_corners=True)

        transformed = grid_sample(
            input_torch,
            sample_grid,
            padding_mode=sample_padding,
            mode=sample_mode,
            align_corners=True,
        )
    else:
        transformed = torch.zeros(out_shape)

    transformed = transformed.permute(dimension_order)

    if matrix_torch.size(2) == len(original_shape):  # had no bulk axis originally
        transformed = transformed.squeeze()

    transformed_dnd: DNDarray = array(
        transformed,
        dtype=input.dtype,
        is_split=input.split,
        device=input.device,
        comm=input.comm,
        copy=False,
    )

    return transformed_dnd
