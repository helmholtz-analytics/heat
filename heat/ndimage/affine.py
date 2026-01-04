"""
Affine transformations for N-dimensional Heat arrays.

This module implements backward-warping affine transformations
(translation, rotation, scaling) for 2D and 3D data stored as
Heat DNDarrays. The implementation is MPI-safe and supports
nearest-neighbor and bilinear interpolation as well as common
boundary handling modes.

The public entry point is `affine_transform`.
"""

import numpy as np
import torch
import heat as ht


# ============================================================
# Helpers
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
        True if the matrix is identity, False otherwise.
    """
    A = M[:, :ND]
    b = M[:, ND:]
    return np.allclose(A, np.eye(ND)) and np.allclose(b, 0)


def _normalize_input(x, ND):
    """
    Normalize input array to include batch and channel dimensions.

    This function converts input arrays to the internal shape
    expected by the sampling routines:
        (N, C, H, W) for 2D
        (N, C, D, H, W) for 3D

    Parameters
    ----------
    x : ht.DNDarray
        Input array.
    ND : int
        Number of spatial dimensions.

    Returns
    -------
    torch.Tensor
        Local torch tensor with batch and channel dimensions.
    tuple
        Original shape of the input array.
    """
    orig_shape = x.shape
    t = x.larray

    if ND == 2:
        if x.ndim == 2:          # (H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:        # (C, H, W)
            t = t.unsqueeze(0)
    else:
        if x.ndim == 3:          # (D, H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:        # (C, D, H, W)
            t = t.unsqueeze(0)

    return t, orig_shape


def _make_grid(spatial, device):
    """
    Construct a coordinate grid in Heat axis order.

    Parameters
    ----------
    spatial : tuple
        Spatial shape of the output volume.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Grid of shape (ND, *spatial).
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
# Padding
# ============================================================

def _apply_padding(pix, spatial, mode):
    """
    Apply boundary handling to integer pixel coordinates.

    Parameters
    ----------
    pix : torch.Tensor
        Integer pixel indices.
    spatial : tuple
        Spatial dimensions of the input.
    mode : str
        Boundary mode ('nearest', 'wrap', 'reflect', 'constant').

    Returns
    -------
    torch.Tensor
        Adjusted pixel coordinates.
    torch.Tensor
        Boolean mask indicating valid coordinates.
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
# Sampling
# ============================================================

def _nearest_sample(x, coords, mode, constant_value):
    """
    Nearest-neighbor sampling.

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

    For non-2D data, this function falls back to nearest-neighbor
    sampling.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    coords : torch.Tensor
        Sampling coordinates.
    mode : str
        Boundary handling mode.
    constant_value : float
        Fill value for constant padding.

    Returns
    -------
    torch.Tensor
        Sampled output tensor.
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

    out = (
        Ia * (1 - wy) * (1 - wx) +
        Ib * (1 - wy) * wx +
        Ic * wy * (1 - wx) +
        Id * wy * wx
    )

    if mode == "constant":
        const = torch.full_like(out, constant_value)
        out = torch.where(valid.unsqueeze(0).unsqueeze(0), out, const)

    return out


# ============================================================
# Local affine (NO MPI logic)
# ============================================================

def _affine_transform_local(x, M, order, mode, constant_value, expand):
    """
    Apply an affine transformation to a local (non-distributed) array.

    Parameters
    ----------
    x : ht.DNDarray
        Input array (split=None).
    M : array-like
        Affine matrix.
    order : int
        Interpolation order.
    mode : str
        Boundary handling mode.
    constant_value : float
        Fill value for constant padding.
    expand : bool
        Whether to expand the output with a leading dimension.

    Returns
    -------
    ht.DNDarray
        Transformed array.
    """
    M = np.asarray(M)
    ND = 2 if M.shape == (2, 3) else 3
    is_identity = _is_identity_affine(M, ND)

    x_local, orig_shape = _normalize_input(x, ND)
    device = x_local.device

    A = torch.tensor(M[:, :ND], device=device, dtype=torch.float64)
    b = torch.tensor(M[:, ND:], device=device, dtype=torch.float64).reshape(ND, 1)
    A_inv = torch.inverse(A)

    spatial = x_local.shape[2:]
    grid = _make_grid(spatial, device).reshape(ND, -1).double()

    if ND == 2:
        grid = grid[[1, 0]]
    else:
        z, y, x_ = grid
        grid = torch.stack([x_, y, z])

    coords = (A_inv @ grid) - (A_inv @ b)

    if ND == 2:
        coords = coords[[1, 0]].reshape((2, *spatial))
    else:
        cx, cy, cz = coords
        coords = torch.stack([
            cz.reshape(spatial),
            cy.reshape(spatial),
            cx.reshape(spatial),
        ])

    if order == 0:
        out = _nearest_sample(x_local, coords, mode, constant_value)
    else:
        out = _bilinear_sample(x_local, coords, mode, constant_value)

    out = out.squeeze(0)

    # Final shape handling
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

def affine_transform(x, M, order=0, mode="nearest", constant_value=0.0, expand=False):
    """
    Apply an affine transformation to a Heat array.

    This function supports both local and MPI-distributed arrays.
    Distributed inputs are gathered, transformed once, and then
    redistributed to the original split.

    Parameters
    ----------
    x : ht.DNDarray
        Input array.
    M : array-like
        Affine matrix.
    order : int, optional
        Interpolation order.
    mode : str, optional
        Boundary handling mode.
    constant_value : float, optional
        Fill value for constant padding.
    expand : bool, optional
        Whether to expand the output shape.

    Returns
    -------
    ht.DNDarray
        Transformed array.
    """
    if x.split is not None:
        x_full = x.resplit(None)
        y_full = _affine_transform_local(
            x_full, M, order, mode, constant_value, expand
        )
        return y_full.resplit(x.split)

    return _affine_transform_local(
        x, M, order, mode, constant_value, expand
    )
