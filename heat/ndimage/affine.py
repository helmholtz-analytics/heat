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
from ..core.communication import MPI
import warnings
from ..core.dndarray import DNDarray
from ..core.linalg.basics import transpose
from ..core.manipulations import hstack
from ..core.factories import array

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
    print(f"{A.size(dim)=}")
    print(f"{idx+1=}")
    if (idx + 1) < A.size(dim):
        rows_after = torch.arange(idx + 1, A.size(dim))
        print(f"{rows_after=}")
        return torch.cat(
            [A.index_select(dim, rows_before), A.index_select(dim, rows_after)], dim=dim
        )
    else:
        return A.index_select(dim, rows_before)


def _to_full_affine(mat):
    # TODO: make distributed, only local for now!
    """
    Convert reduced affine matrices to full homogeneous form.

    Works with single matrices or batches of any dimensionality:
        - (D, D+1)           → (D+1, D+1)    # single
        - (N, D, D+1)        → (N, D+1, D+1) # batch

    Args:
        mat: Reduced affine tensor

    Returns
    -------
        Full homogeneous affine tensor
    """
    # Detect if batched by checking number of dimensions
    if mat.dim() == 2:
        # Single matrix case: (D, D+1)
        D = mat.shape[0]  # spatial dimension
        full = torch.zeros(D + 1, D + 1, dtype=mat.dtype, device=mat.device)
        full[:D, :] = mat  # copy top D rows
        full[D, D] = 1.0  # set homogeneous coordinate
        return full

    elif mat.dim() == 3:
        # Batched case: (N, D, D+1)
        N, D, _ = mat.shape
        full = torch.zeros(N, D + 1, D + 1, dtype=mat.dtype, device=mat.device)
        full[:, :D, :] = mat  # copy top D rows for each batch
        full[:, D, D] = 1.0  # set homogeneous coordinate for each batch
        return full

    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {mat.dim()}D")


# TODO: make this able to handle batches of matrices
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
    print()
    print("START NORMALIZATION")
    print()
    D = len(sizes)
    print(f"{D=}")
    print(f"{sizes=}")
    print(f"{M.shape}")

    # construct coord space transform
    scales = (torch.as_tensor(sizes) - 1) / 2.0
    # this feels like an ugly hack, I am not quite shure why it is needed here, maybe something to do with scipy being inverse?
    scales = scales.flip(0)
    M_scales = torch.diag(scales)

    # scales = (sizes[0] - 1) / 2.0
    # M_scales = torch.eye(len(sizes)) * scales
    print(f"scales: {M_scales}")

    D = len(sizes)
    T_np = torch.zeros((D + 1, D + 1))
    T_np[:D, :D] = M_scales
    T_np[D, D] = 1
    T_np[:D, D] = scales
    T_pn = T_np.inverse()
    print("construct coord transforms")
    print(f"{T_pn=}")

    full_transformed = T_pn @ M @ T_np
    print(f"{full_transformed=}")

    print()
    print("END NORMALIZATION")
    print()

    return full_transformed[:, :D, :]


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
    Input is expected to have shape [N x] [D x] H x W x C because that is consistent with PIL+Numpy to get Image data.
    To be consistent with scipy the Matrix then is of shape 3x4 (3x3, 4x4 also valid), while the Row/Column for transforming
    color data gets ignored for now because it is not supported by affine_grid()
    """
    # TODO: Implement cases 3x3, 2x3, 2x2

    # input setup
    sample_padding = MODE_TO_PADDING[mode]
    sample_mode = ORDER_TO_MODE[order]

    matrix_torch: torch.Tensor = matrix.larray

    if matrix_torch.dim() == 2:
        matrix_torch = matrix_torch.unsqueeze(0)

    # 2d image given, third dimension are/would be color vector transforms
    if input.ndim == 3 and 3 <= matrix.shape[0] <= 4 and 3 <= matrix.shape[1] <= 4:
        # remove axis that represents transforming the color dimension, because
        # torch affine_grid does not support tranformaing Color dimension out of the box
        matrix_torch = _remove_slice(matrix_torch, 2, dim=1)
        matrix_torch = _remove_slice(matrix_torch, 2, dim=2)

        if matrix.shape == (
            3,
            3,
        ):  # translation information missing, using offset value
            matrix_torch = torch.hstack([matrix_torch, offset[1:3]])
        if matrix.shape != (4, 4):
            # remove axis that represents transforming the color dimension, because
            # torch affine_grid does not support that
            matrix_torch = _to_full_affine(matrix_torch)
    else:
        raise ValueError("transform matrix has no valid shape")

    # for now matrix has shape 3x3xB

    t_input = transpose(input, (2, 1, 0))  # to C x W x H
    input_torch = t_input.larray.unsqueeze(0)

    matrix_torch = convert_matrix_space(matrix_torch, t_input.shape[1:])

    size = torch.Size((1, t_input.shape[0], t_input.shape[1], t_input.shape[2]))
    print(f"{size=}")
    sample_grid: torch.Tensor = affine_grid(matrix_torch, size)

    transformed = grid_sample(
        input_torch,
        sample_grid,
        padding_mode=sample_padding,
        mode=sample_mode,
    )
    return array(transformed.squeeze(0).permute(2, 1, 0))
