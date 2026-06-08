"""
Affine transformations for N-dimensional Heat arrays.

This module implements backward-warping affine transformations
(translation, rotation, scaling) for 2D and 3D data stored as
Heat DNDarrays, using a PyTorch backend.

The affine matrix M is interpreted as a *forward* transform
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

import numpy as np
import torch
from torch.nn.functional import affine_grid
from torch.nn.functional import grid_sample
import heat as ht
from mpi4py import MPI
import warnings
from ..core.dndarray import DNDarray
from ..core import factories
from ..core import manipulations
from ..core.linalg.basics import transpose

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


def to_full_affine(mat):
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


def _matrix_pixel_to_normalized_coords(M: torch.Tensor, sizes):
    """
    Convert scipy affine matrix to PyTorch normalized coordinates.

    Args:
        M: Torch Tensor of shape (D+1, D+1) — [a_ij | t_i]
        sizes: image sizes [H, W, D, ...] of length D

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
    M_scales = torch.diag(scales)
    D = len(sizes)
    T_np = torch.zeros((D + 1, D + 1))
    T_np[:D, :D] = M_scales
    T_np[D, D] = 1
    T_np[:D, D] = scales
    T_pn = T_np.inverse()
    print("construct coord transforms")
    print(f"{T_np=}")

    full_transformed = T_pn @ M @ T_np
    print(f"{full_transformed=}")

    print()
    print("END NORMALIZATION")
    print()

    return full_transformed[:, :D, :]


def _swap_rows_cols(A, row_pair, col_pair):
    print()
    print("swapping stuff")
    print()
    print(A)
    i, j = row_pair
    k, m = col_pair

    row_idx = torch.arange(A.size(0))
    row_idx[i] = j
    row_idx[j] = i

    col_idx = torch.arange(A.size(1))
    col_idx[k] = m
    col_idx[m] = k
    A = A[row_idx[:, None], col_idx[None, :]]
    print(A)
    print()
    print("end swapping stuff")
    print()
    return A


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
    Input is expected to have shape H x W x C because that is consistent with PIL+Numpy to get Image data
    -> to be consistent with scipy the Matrix then has to be of shape 3x4 (3x3, 4x4 also valid)
    """
    # TODO: Implement cases 3x3, 2x3, 2x2

    matrix_torch: torch.Tensor
    if input.ndim == 3:  # 2d image where third dimension are color vectors
        if matrix.shape == (3, 4):
            # remove axis that represents transforming the color dimension, because
            # torch does not support that

            matrix_torch = _remove_slice(matrix.larray, 2, 0)
            matrix_torch = _remove_slice(matrix_torch, 2, 1)
            matrix_torch = to_full_affine(matrix_torch)
            matrix_torch = _swap_rows_cols(matrix_torch, (0, 1), (0, 1))
        elif matrix.shape == (4, 4):
            # remove axis that represents transforming the color dimension, because
            # torch does not support that
            matrix_torch = _remove_slice(matrix.larray, 2, 0)
            matrix_torch = _remove_slice(matrix_torch, 2, 1)
            matrix_torch = _swap_rows_cols(matrix_torch, (0, 1), (0, 1))
        else:
            raise NotImplementedError()
    else:
        raise ValueError("transform matrix has no valid shape")

    if matrix_torch.dim() == 2:
        matrix_torch = matrix_torch.unsqueeze(0)

    # for now matrix has shape 3x3xB

    t_input = transpose(input, (2, 0, 1))  # to C x H x W
    matrix_torch = _matrix_pixel_to_normalized_coords(matrix_torch, t_input.shape[1:])
    input_torch = t_input.larray.unsqueeze(0)

    sample_padding = MODE_TO_PADDING[mode]
    sample_mode = ORDER_TO_MODE[order]
    size = torch.Size((1, t_input.shape[0], t_input.shape[1], t_input.shape[2]))
    print(f"{size=}")
    sample_grid: torch.Tensor = affine_grid(matrix_torch, size)

    transformed = grid_sample(
        input_torch, sample_grid, padding_mode=sample_padding, mode=sample_mode
    )
    return ht.array(transformed.squeeze(0).permute(1, 2, 0))


# ============================================================
# Helper utilities (old, to be determined if still helpful)
# ============================================================


def _is_identity_affine(M, ND):
    """
    Check whether an affine matrix represents the identity transform.

    Parameters
    ----------
    M : array-like
        Affine matrix of shape (ND, ND+1).
    ND : int
        Number of spatial dimensions.

    Returns
    -------
    bool
        True if A is the identity matrix and b is zero.
    """
    A = M[:, :ND]
    b = M[:, ND:]
    return np.allclose(A, np.eye(ND)) and np.allclose(b, 0)


def _normalize_input(x, ND):
    """
    Normalize a Heat array to a unified internal layout.

    For sampling, inputs are reshaped to include synthetic
    batch and channel dimensions:

      - 2D: (N, C, H, W)
      - 3D: (N, C, D, H, W)

    These dimensions are internal only and do not imply
    semantic batching or channels in the input data.

    Parameters
    ----------
    x : ht.DNDarray
        Input array.
    ND : int
        Number of spatial dimensions.

    Returns
    -------
    torch.Tensor
        Local torch tensor with added batch/channel dimensions.
    tuple
        Original shape of the input array.
    """
    orig_shape = x.shape
    t = x.larray

    if ND == 2:
        if x.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            t = t.unsqueeze(0)
    else:
        if x.ndim == 3:
            t = t.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            t = t.unsqueeze(0)

    return t, orig_shape


def _make_grid(spatial, device):
    """
    Construct a coordinate grid in Heat spatial axis order.

    The grid contains integer coordinates corresponding to
    output pixel locations and is later mapped through the
    inverse affine transform.

    Parameters
    ----------
    spatial : tuple
        Spatial shape (H, W) or (D, H, W).
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Coordinate grid of shape (ND, *spatial) in Heat order.
    """
    if len(spatial) == 2:
        H, W = spatial
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([gy, gx], dim=0)
    else:
        D, H, W = spatial
        z = torch.arange(D, device=device)
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([gz, gy, gx], dim=0)
