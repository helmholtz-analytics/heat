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
import heat as ht
from mpi4py import MPI
import warnings

# ============================================================
# Helper utilities
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


# ============================================================
# Boundary handling
# ============================================================


def _apply_padding(pix, spatial, mode):
    """
    Apply boundary handling rules to integer pixel indices.

    Parameters
    ----------
    pix : torch.Tensor
        Integer pixel indices in Heat order.
    spatial : tuple
        Spatial dimensions of the input.
    mode : str
        Padding mode ('nearest', 'wrap', 'reflect', 'constant').

    Returns
    -------
    torch.Tensor
        Adjusted pixel indices.
    torch.Tensor
        Boolean mask indicating valid locations
        (used only for constant padding).
    """
    ND = len(spatial)
    final = pix.clone()
    valid = torch.ones_like(pix[0], dtype=torch.bool)

    for d in range(ND):
        size = spatial[d]
        p = pix[d]

        if mode == "constant":
            ok = (p >= 0) & (p < size)
            valid &= ok
            final[d] = p.clamp(0, size - 1)
        elif mode == "nearest":
            final[d] = p.clamp(0, size - 1)
        elif mode == "wrap":
            final[d] = torch.remainder(p, size)
        elif mode == "reflect":
            if size == 1:
                final[d] = torch.zeros_like(p)
            else:
                r = torch.abs(p)
                r = torch.remainder(r, 2 * size - 2)
                final[d] = torch.where(r < size, r, 2 * size - 2 - r)

    return final, valid


# ============================================================
# Sampling routines
# ============================================================


def _nearest_sample(x, coords, mode, constant_value):
    """
    Sample input values using nearest-neighbor interpolation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, C, ...).
    coords : torch.Tensor
        Sampling coordinates in Heat order.
    mode : str
        Boundary handling mode.
    constant_value : float
        Fill value for constant padding.

    Returns
    -------
    torch.Tensor
        Sampled output tensor.
    """
    ND = coords.shape[0]
    pix = coords.round().long()
    spatial = x.shape[2:]

    pix_c, valid = _apply_padding(pix, spatial, mode)

    if ND == 2:
        y, x_ = pix_c
        out = x[:, :, y, x_]
    else:
        z, y, x_ = pix_c
        out = x[:, :, z, y, x_]

    if mode == "constant":
        const = torch.full_like(out, constant_value)
        out = torch.where(valid.unsqueeze(0).unsqueeze(0), out, const)

    return out


def _bilinear_sample(x, coords, mode, constant_value):
    """
    Bilinear interpolation for 2D inputs.

    For non-2D data, this function falls back to
    nearest-neighbor sampling.
    """
    if coords.shape[0] != 2:
        return _nearest_sample(x, coords, mode, constant_value)

    y, x_ = coords
    H, W = x.shape[2], x.shape[3]

    y0 = torch.floor(y).long()
    x0 = torch.floor(x_).long()
    y1 = y0 + 1
    x1 = x0 + 1

    pix = torch.stack([y0, x0])
    pix_c, valid = _apply_padding(pix, (H, W), mode)

    y0c, x0c = pix_c
    y1c = y1.clamp(0, H - 1)
    x1c = x1.clamp(0, W - 1)

    Ia = x[:, :, y0c, x0c]
    Ib = x[:, :, y0c, x1c]
    Ic = x[:, :, y1c, x0c]
    Id = x[:, :, y1c, x1c]

    wy = y - y0.float()
    wx = x_ - x0.float()

    out = Ia * (1 - wy) * (1 - wx) + Ib * (1 - wy) * wx + Ic * wy * (1 - wx) + Id * wy * wx

    if mode == "constant":
        const = torch.full_like(out, constant_value)
        out = torch.where(valid.unsqueeze(0).unsqueeze(0), out, const)

    return out


# ============================================================
# Local affine transform (no MPI logic)
# ============================================================


def _affine_transform_local(x, M, order, mode, constant_value, expand):
    """
    Apply an affine transformation to a local (non-distributed) Heat array.

    Parameters
    ----------
    x : ht.DNDarray
        Local input array (split=None).
    M : array-like
        Affine matrix of shape (2,3) or (3,4).
    order : int
        Interpolation order.
    mode : str
        Boundary handling mode.
    constant_value : float
        Fill value for constant padding.
    expand : bool
        Whether to expand the output with a leading batch dimension.

    Returns
    -------
    ht.DNDarray
        Transformed array with split=None.
    """
    M = np.asarray(M)

    if M.shape == (2, 3):
        ND = 2
    elif M.shape == (3, 4):
        ND = 3
    else:
        raise ValueError("M must have shape (2,3) or (3,4)")

    is_identity = _is_identity_affine(M, ND)

    x_local, orig_shape = _normalize_input(x, ND)
    device = x_local.device

    A = torch.tensor(M[:, :ND], device=device, dtype=torch.float64)
    b = torch.tensor(M[:, ND:], device=device, dtype=torch.float64).reshape(ND, 1)
    A_inv = torch.inverse(A)

    spatial = x_local.shape[2:]
    grid_heat = _make_grid(spatial, device).reshape(ND, -1).double()

    if ND == 2:
        grid_affine = grid_heat[[1, 0]]
    else:
        z, y, x_ = grid_heat
        grid_affine = torch.stack([x_, y, z], dim=0)

    coords_affine = (A_inv @ grid_affine) - (A_inv @ b)

    if ND == 2:
        coords_heat = coords_affine[[1, 0]].reshape((2, *spatial))
    else:
        cx, cy, cz = coords_affine
        coords_heat = torch.stack(
            [
                cz.reshape(spatial),
                cy.reshape(spatial),
                cx.reshape(spatial),
            ],
            dim=0,
        )

    if order == 0:
        out = _nearest_sample(x_local, coords_heat, mode, constant_value)
    else:
        out = _bilinear_sample(x_local, coords_heat, mode, constant_value)

    out = out.squeeze(0)

    if expand:
        if out.ndim == ND + 1:
            out = out.squeeze(0)
        return ht.array(out, split=None).expand_dims(0)

    if ND == 2:
        if order == 0 or is_identity:
            return ht.array(out.squeeze(0).reshape(orig_shape), split=None)
        return ht.array(out, split=None)

    if is_identity:
        return ht.array(out.squeeze(0).reshape(orig_shape), split=None)

    return ht.array(out, split=None)


# ============================================================
# Public API (MPI-safe)
# ============================================================
def affine_transform(x, M, order=0, mode="constant", constant_value=0.0, expand=False):
    """
    Space-based affine transform for Heat arrays.

    Translation, scaling, rotation, shear are ALL handled
    via spatial resampling (backward warping).

    IMPORTANT:
    - No slice shifting
    - No axis-special casing
    - For distributed arrays with spatial splits:
        * halo exchange is NOT implemented yet
        * boundary voxels may be incomplete
    """
    M = np.asarray(M)
    if M.shape == (2, 3):
        ND = 2
    elif M.shape == (3, 4):
        ND = 3
    else:
        raise ValueError("M must have shape (2,3) or (3,4)")

    # Identity shortcut (always safe)
    if _is_identity_affine(M, ND):
        return x.copy()
    # Warn about distributed spatial splits
    if x.split is not None and x.split < ND:
        warnings.warn(
            "affine_transform: spatially distributed arrays are processed "
            "locally per rank without halo exchange. Boundary values may be "
            "incomplete. Use resplit(None) for exact results.",
            RuntimeWarning,
        )

    return _affine_transform_local(x, M, order, mode, constant_value, expand)
