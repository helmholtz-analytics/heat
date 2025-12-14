"""
Affine transformations for N-dimensional images.

This module provides utilities to apply affine transformations
(translation, rotation, scaling, shearing) to 2D and 3D images
represented as Heat DNDarrays. The implementation follows a
backward-warping approach similar to SciPy and PyTorch.
"""

import numpy as np
import torch
import heat as ht


# ============================================================
# Utility: normalize input → (N, C, spatial…)
# ============================================================

def _normalize_input(x, ND):
    """
    Normalize a Heat array to include batch and channel dimensions.

    Parameters
    ----------
    x : ht.DNDarray
        Input image or volume.
    ND : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, C, *spatial).
    tuple
        Original shape of the input array.
    """
    orig_shape = x.shape
    t = x.larray

    if ND == 2:
        if x.ndim == 2:        # (H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:      # (C, H, W)
            t = t.unsqueeze(0)
    else:
        if x.ndim == 3:        # (D, H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:      # (C, D, H, W)
            t = t.unsqueeze(0)

    return t, orig_shape


# ============================================================
# Utility: build coordinate grid in Heat order
# ============================================================

def _make_grid(spatial, device):
    """
    Construct a coordinate grid in Heat axis order.

    Parameters
    ----------
    spatial : tuple of int
        Spatial shape of the output image.
    device : torch.device
        Device on which to create the grid.

    Returns
    -------
    torch.Tensor
        Coordinate grid of shape (ND, *spatial).
    """
    if len(spatial) == 2:
        H, W = spatial
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx], dim=0)
    else:
        D, H, W = spatial
        zs = torch.arange(D, device=device)
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        return torch.stack([gz, gy, gx], dim=0)


# ============================================================
# Padding helper
# ============================================================

def _apply_padding(pix, spatial, mode, constant_value):
    """
    Apply boundary handling to sampled pixel indices.

    Parameters
    ----------
    pix : torch.Tensor
        Integer pixel coordinates.
    spatial : tuple of int
        Spatial dimensions of the image.
    mode : str
        Boundary mode ('nearest', 'wrap', 'reflect', 'constant').
    constant_value : float
        Fill value for constant padding.

    Returns
    -------
    torch.Tensor
        Clipped or wrapped pixel coordinates.
    torch.Tensor
        Boolean mask of valid coordinates.
    """
    ND = len(spatial)
    final = pix.clone()
    valid = torch.ones_like(pix[0], dtype=torch.bool)

    for d in range(ND):
        size = spatial[d]
        p = pix[d]

        if mode == "constant":
            good = (p >= 0) & (p < size)
            valid &= good
            final[d] = p.clamp(0, size - 1)
        elif mode == "nearest":
            final[d] = p.clamp(0, size - 1)
        elif mode == "wrap":
            final[d] = torch.remainder(p, size)
        elif mode == "reflect":
            r = torch.abs(p)
            r = torch.remainder(r, 2 * size - 2)
            final[d] = torch.where(r < size, r, 2 * size - 2 - r)

    return final, valid


# ============================================================
# Nearest neighbor sampler
# ============================================================

def _nearest_sample(x_local, coords_h, mode, constant_value):
    """
    Sample an image using nearest-neighbor interpolation.

    Parameters
    ----------
    x_local : torch.Tensor
        Input tensor of shape (N, C, *spatial).
    coords_h : torch.Tensor
        Coordinates in Heat order.
    mode : str
        Boundary handling mode.
    constant_value : float
        Fill value for constant padding.

    Returns
    -------
    torch.Tensor
        Sampled output tensor.
    """
    ND = coords_h.shape[0]
    pix = coords_h.round().long()
    spatial = x_local.shape[2:]

    pix_c, valid = _apply_padding(pix, spatial, mode, constant_value)

    if ND == 2:
        iy, ix = pix_c
        out = x_local[:, :, iy, ix]
    else:
        iz, iy, ix = pix_c
        out = x_local[:, :, iz, iy, ix]

    if mode == "constant":
        out = torch.where(
            valid.unsqueeze(0).unsqueeze(0),
            out,
            torch.tensor(constant_value, device=out.device, dtype=out.dtype),
        )

    return out


# ============================================================
# Bilinear sampling (2D only)
# ============================================================

def _bilinear_sample(x_local, coords_h, mode, constant_value):
    """
    Sample a 2D image using bilinear interpolation.

    Falls back to nearest-neighbor sampling for non-2D inputs.
    """
    if coords_h.shape[0] != 2:
        return _nearest_sample(x_local, coords_h, mode, constant_value)

    y, x = coords_h
    H, W = x_local.shape[2], x_local.shape[3]

    y0 = torch.floor(y).long()
    x0 = torch.floor(x).long()
    y1 = y0 + 1
    x1 = x0 + 1

    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)
    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)

    Ia = x_local[:, :, y0c, x0c]
    Ib = x_local[:, :, y0c, x1c]
    Ic = x_local[:, :, y1c, x0c]
    Id = x_local[:, :, y1c, x1c]

    wy = y - y0.float()
    wx = x - x0.float()

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


# ============================================================
# Public API
# ============================================================

def affine_transform(
    x,
    M,
    order=0,
    mode="nearest",
    constant_value=0.0,
    expand=False,
):
    """
    Apply an affine transformation to an N-dimensional Heat array.

    The transformation is performed using backward warping:
    output coordinates are mapped to input coordinates via the
    inverse affine matrix.

    Parameters
    ----------
    x : ht.DNDarray
        Input image or volume.
    M : array-like
        Affine transformation matrix of shape (2, 3) for 2D or
        (3, 4) for 3D transformations.
    order : int, optional
        Interpolation order (0 = nearest neighbor, 1 = bilinear).
    mode : str, optional
        Boundary handling mode ('nearest', 'wrap', 'reflect', 'constant').
    constant_value : float, optional
        Fill value used when mode is 'constant'.
    expand : bool, optional
        Reserved for future use. Currently has no effect.

    Returns
    -------
    ht.DNDarray
        Transformed image or volume with the same shape as the input.
    """
    M = np.asarray(M, dtype=np.float32)

    if M.shape == (2, 3):
        ND = 2
    elif M.shape == (3, 4):
        ND = 3
    else:
        raise ValueError("Affine matrix must be 2x3 (2D) or 3x4 (3D).")

    x_local, orig_shape = _normalize_input(x, ND)
    device = x_local.device

    A = torch.tensor(M[:, :ND], device=device)
    b = torch.tensor(M[:, ND:], device=device).reshape(ND, 1)

    A_inv = torch.inverse(A)

    spatial = x_local.shape[2:]
    grid_h = _make_grid(spatial, device)

    P = int(np.prod(spatial))
    grid_flat = grid_h.reshape(ND, P).float()

    if ND == 2:
        grid_pt = grid_flat[[1, 0], :]
    else:
        z, y, x_ = grid_flat
        grid_pt = torch.stack([x_, y, z], dim=0)

    coords_pt = (A_inv @ grid_pt) - (A_inv @ b)

    if ND == 2:
        coords_h = coords_pt[[1, 0], :].reshape((2, *spatial))
    else:
        cx, cy, cz = coords_pt
        coords_h = torch.stack(
            [cz.reshape(spatial),
             cy.reshape(spatial),
             cx.reshape(spatial)],
            dim=0,
        )

    if order == 0:
        out_local = _nearest_sample(x_local, coords_h, mode, constant_value)
    else:
        out_local = _bilinear_sample(x_local, coords_h, mode, constant_value)

    out = out_local.squeeze(0)

    split_val = None
    if hasattr(x, "split") and not callable(x.split):
        split_val = x.split

    out = ht.array(out, split=split_val)

    return out.reshape(orig_shape)
