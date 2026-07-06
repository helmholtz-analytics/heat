"""
Affine transformations for N-dimensional Heat arrays.

This module implements backward-warping affine transformations
(translation, rotation, scaling) for 2D and 3D data stored as
Heat DNDarrays, using a PyTorch backend.

The affine matrix M is interpreted as a *b* transform
in affine (x, y [, z]) coordinates:

    out = A @ inp + b

where M = [A | b] has shape (ND, ND+1).

Internally, backward warping is used for resampling:

    inp = A^{-1} @ (out - b)

Spatial axis conventions in Heat:
- 2D arrays: (H, W)  == (y, x)
- 3D arrays: (D, H, W) == (z, y, x)

Interpolation and boundary handling:
- order=0: nearest-neighbor
- order=1: bilinear (2D only; 3D falls back to nearest)
- padding modes: 'nearest', 'wrap', 'reflect', 'constant'

Distributed arrays:
- Non-spatial splits are handled locally without communication.
- Spatial splits support only simple, axis-aligned transforms
  (e.g. translation or diagonal scaling).
- More general affine transforms (rotation/shear) would require
  halo exchange and are intentionally not supported yet.

The public entry point is `affine_transform`.
"""

import torch
from torch.nn.functional import affine_grid, grid_sample
from heat.core.communication import MPI
from heat.core.dndarray import DNDarray
from heat.core.linalg.basics import transpose
from heat.core import arange

# from heat.core.manipulations import hstack
from heat.core.factories import array

MODE_TO_PADDING = {
    # SciPy mode               # torch padding_mode
    "constant": "zeros",  # fill with ``cval`` (default 0)
    "reflect": "reflection",  # reflect at the border
    "mirror": "reflection",  # also mapped to reflection
    "nearest": "border",  # replicate the edge pixel
    # The following SciPy modes have no exact Torch counterpart.
    # We keep them as ``None`` and raise an error if they are used.
    "wrap": None,
    "grid-wrap": None,
    "grid-constant": None,
}

ORDER_TO_MODE = {
    0: "nearest",  # order‑0 → nearest‑neighbour
    1: "bilinear",  # order‑1 → bilinear (linear) sampling
    3: "bicubic",  # order‑3 → bicubic sampling
    # SciPy supports orders 2,4,5 as well – they have no direct Torch counterpart.
    # We fall back to the closest supported mode.
    2: "bilinear",  # closest supported mode
    4: "bicubic",
    5: "bicubic",
    # any other order (should never happen) → default to bilinear
}

filtering_map = {}


# ============================================================
# Helper utilities
# ============================================================
def _remove_slice(A: torch.Tensor, idx: int, dim: int) -> torch.Tensor:
    # Keep rows before and after the removed row
    rows_before = torch.arange(0, idx)
    if (idx + 1) < A.size(dim):
        rows_after = torch.arange(idx + 1, A.size(dim))
        return torch.cat(
            [A.index_select(dim, rows_before), A.index_select(dim, rows_after)], dim=dim
        )

    return A.index_select(dim, rows_before)


def _to_full_affine(M):
    # TODO: make distributed, only really benefitial in the bulk axis, since the matrix probably always small
    """
    Convert reduced affine matrices to full homogeneous form.

    Works with single matrices or batches:
        - (D, D+1)           → (D+1, D+1)    # single
        - (N, D, D+1)        → (N, D+1, D+1) # batch

    Args:
        mat: Reduced affine tensor

    Returns
    -------
        Full homogeneous affine tensor
    """
    # Detect if batched by checking number of dimensions
    if M.dim() == 2:
        # Single matrix case: (D, D+1)
        D = M.shape[0]  # spatial dimension
        full = torch.zeros(D + 1, D + 1, dtype=M.dtype, device=M.device)
        full[:D, :] = M  # copy top D rows
        full[D, D] = 1.0  # set homogeneous coordinate
        return full

    elif M.dim() == 3:
        # Batched case: (N, D, D+1)
        N, D, _ = M.shape
        full = torch.zeros(N, D + 1, D + 1, dtype=M.dtype, device=M.device)
        full[:, :D, :_] = M  # copy top D rows for each batch
        full[:, D, D] = 1.0  # set homogeneous coordinate for each batch
        return full

    else:
        raise ValueError(
            f"Expected affine transformation matrix to be 2D or 3D tensor, got {M.dim()}D"
        )


def convert_matrix_space(M: torch.Tensor, sizes):
    """
    Convert scipy affine matrix to normalized coordinates used by affine_grid.
    scipy uses pixel coordinate space with origin in the top left,
    while affine_grid uses -1 to 1 with origin in the center of the image

    Args:
        M: Torch Tensor of shape (D+1, D+1)
        sizes: image sizes [H, W, D, ...] of length D (excluding color dimension)

    Returns
    -------
        theta: torch affine matrix of shape (D, D+1)
    """
    # construct coord space transform
    scales = (torch.as_tensor(sizes) - 1) / 2.0
    M_scales = torch.diag(scales)

    D = len(sizes)
    T_np = torch.zeros(D + 1, D + 1)
    T_np[:D, :D] = M_scales
    T_np[D, D] = 1
    T_np[:D, D] = scales
    T_pn = T_np.inverse()
    print(M.shape)
    print(T_np.shape)
    transformed = T_pn @ M @ T_np

    return transformed[:, :D, :]


# ============================================================
#  main methods
# ============================================================
def affine_transform(
    input: DNDarray,
    matrix: DNDarray,
    offset=0.0,
    output_shape=None,
    output=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
) -> DNDarray:
    """
    Parameters
    ----------
    input : DNDarray
        the image or data array to transform. Input is expected to have shape [N x] [D x] H x W x C because that is consistent
        with PIL+Numpy to get Image data.
    matrix : DNDarray
        afine matrix used to transform input. can be of shape Bx3x4 (3x3, 4x4 also valid) for 2d data,
        or should be of shape Bx4x5 (4x4, 5x5 also valid) for 3d data
        B stands for the Bulk axis and can be ommited
        The row and column corresponding with Transformation of the Color-Axis is ignored right now,
        because it is not supported by the torch.affine_grid() function.
    offset : DNDarray
        offset vector that can be used instead of adding offset into affine matrix directly. only in effect when the matrix
        given has no transform vector
    output_shape :
        shape of the given output. not implemented yet
    output : DNDarray
        optional parameter to specify array in wich the output should be placed. currently not implemented yet
    order :
        type of interpolation that is used, linear to cubic allowed
    mode :
        paddding mode for values outside of array
    cval :
        value with wich the padding should be filled. currently not used because of limitation of torch.sample_grid()
    prefilter : bool
        if the input should be filtered before transformed, currently not because torch.sample_grid does not have this functionality




    currently not implemented:
        -cval: not supported by torch.grid_sample
        -ouput_shape, output: currently not priority
        -prefilter: not supported by torch.grid_sample, ad no comparable function found yet
    """
    # TODO: Implement cases 3x3, 2x3, 2x2 and 4d(3d) versions

    # input conversion
    sample_padding = MODE_TO_PADDING[mode]
    sample_mode = ORDER_TO_MODE[order]

    matrix_torch: torch.Tensor = matrix.larray

    if matrix_torch.dim() == 2:
        matrix_torch = matrix_torch.unsqueeze(0)
        is_bulk = False
    else:
        is_bulk = True

    is_2d_operation = (
        ((input.ndim == 3 and not is_bulk) or (input.ndim == 4 and is_bulk))
        and 3 <= matrix_torch.shape[1] <= 4
        and 3 <= matrix_torch.shape[2] <= 4
    )
    is_3d_operation = (
        ((input.ndim == 4 and not is_bulk) or (input.ndim == 5 and is_bulk))
        and 4 <= matrix_torch.shape[1] <= 5
        and 4 <= matrix_torch.shape[2] <= 5
    )

    # 2d image given, third dimension are/would be color vector transforms
    if is_2d_operation:
        print("2d operation")
        # remove axis that represents transforming the color dimension, because
        # torch affine_grid does not support tranformaing Color dimension out of the box
        matrix_torch = _remove_slice(matrix_torch, 2, dim=1)
        matrix_torch = _remove_slice(matrix_torch, 2, dim=2)

        if matrix_torch.shape == (
            1,
            3,
            3,
        ):
            # translation information missing, using offset value
            matrix_torch = torch.hstack([matrix_torch, offset[1:3]])
        if matrix.shape != (4, 4):
            matrix_torch = _to_full_affine(matrix_torch)

    # 3d image given
    elif is_3d_operation:
        print("3d operation")
        # remove axis that represents transforming the color dimension, because
        # torch affine_grid does not support tranformaing Color dimension out of the box
        matrix_torch = _remove_slice(matrix_torch, 3, dim=1)
        matrix_torch = _remove_slice(matrix_torch, 3, dim=2)

        if matrix_torch.shape == (
            1,
            4,
            4,
        ):
            # translation information missing, using offset value
            matrix_torch = torch.hstack(
                [matrix_torch, offset[1:4]]
            )  # TODO: make this able to handle bulk
        print(f"shape before full_affine: {matrix_torch.shape}")
        if matrix_torch.shape[1:] != (4, 4):
            matrix_torch = _to_full_affine(matrix_torch)

    else:
        raise ValueError(f"transform matrix has no valid shape {matrix.shape=}")

    # i am not quite shure why this transpose below is necessary for the right behaviour,
    # This switches wich axis in input data is influenced by wich row/column in the
    # afine matrix and is the reverse of the expected matching.
    if not is_bulk:
        dimension_order = tuple(
            idx for idx in range(input.ndim - 1, -1, -1)
        )  # reversed: (3,2,1,0) or (2,1,0)
    else:
        dimension_order = (0,) + tuple(
            idx for idx in range(input.ndim - 1, 0, -1)
        )  # reversed: (0,4,3,2,1) or (0,3,2,1)
        print(f"{dimension_order=}")

    t_input = transpose(input, dimension_order)
    print(f"{t_input.shape=}")
    input_torch = t_input.larray
    # input has no bulk axis
    if matrix_torch.size(2) == input.ndim:
        input_torch = input_torch.unsqueeze(0)
        print("adding bulk axis")

    print(f"{input_torch.shape[2:]=}")
    matrix_torch = convert_matrix_space(
        matrix_torch,
        input_torch.shape[-1:1:-1],  # shape is reversed and without bulk and color dims
    )

    size = torch.Size((input_torch.shape))
    print(f"{size=}")
    sample_grid: torch.Tensor = affine_grid(matrix_torch, size)

    transformed = grid_sample(
        input_torch,
        sample_grid,
        padding_mode=sample_padding,
        mode=sample_mode,
    )
    if matrix_torch.size(2) == input.ndim:
        transformed = transformed.squeeze()

    transformed_dnd: DNDarray = array(transformed.permute(dimension_order))

    return transformed_dnd
