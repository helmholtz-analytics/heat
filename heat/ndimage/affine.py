"""
Affine transformations for N-dimensional Heat arrays.

Distributed support:
- identity
- axis-aligned scaling
- translation (WITH halos)

Not supported (distributed):
- rotation
- shear
"""

import numpy as np
import torch
import heat as ht


# ============================================================
# Helpers
# ============================================================

def _is_identity_affine(M, ND):
    A = M[:, :ND]
    b = M[:, ND:]
    return np.allclose(A, np.eye(ND)) and np.allclose(b, 0)


def _normalize_input(x, ND):
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


def _apply_padding(pix, spatial):
    valid = torch.ones_like(pix[0], dtype=torch.bool)
    for d in range(len(spatial)):
        valid &= (pix[d] >= 0) & (pix[d] < spatial[d])
    return pix, valid


# ============================================================
# Local affine (non-distributed)
# ============================================================

def _affine_transform_local(x, M):
    M = np.asarray(M)
    ND = M.shape[0]

    x_local, orig_shape = _normalize_input(x, ND)
    device = x_local.device

    A = torch.tensor(M[:, :ND], device=device, dtype=torch.float64)
    b = torch.tensor(M[:, ND], device=device, dtype=torch.float64).reshape(ND, 1)
    A_inv = torch.inverse(A)

    spatial = x_local.shape[2:]
    grid = _make_grid(spatial, device).reshape(ND, -1).double()

    # Heat → affine coordinate order
    if ND == 3:
        z, y, x_ = grid
        grid_aff = torch.stack([x_, y, z], dim=0)
    else:
        grid_aff = grid[[1, 0]]

    src = (A_inv @ grid_aff) - (A_inv @ b)
    src = src.round().long()

    # affine → Heat order
    if ND == 3:
        x_, y, z = src
        src = torch.stack([z, y, x_], dim=0)
    else:
        src = src[[1, 0]]

    src, valid = _apply_padding(src, spatial)

    out = torch.zeros(grid.shape[1], device=device, dtype=x_local.dtype)
    out[valid] = x_local[0, 0][tuple(src[:, valid])]
    out = out.reshape(spatial)

    return ht.array(out.reshape(orig_shape), split=None)


# ============================================================
# Public API (MPI-aware)
# ============================================================

def affine_transform(x, M):
    """
    Distributed affine transform with halo-aware translation.
    """

    M = np.asarray(M)
    ND = M.shape[0]
    A = M[:, :ND]
    b = M[:, ND]

    if not np.allclose(A, np.diag(np.diag(A))):
        raise NotImplementedError("rotation / shear not supported")

    split = x.split

    # --------------------------------------------------
    # Non-distributed case
    # --------------------------------------------------
    if split is None:
        return _affine_transform_local(x, M)

    # --------------------------------------------------
    # Distributed case
    # --------------------------------------------------
    local = x.larray
    local_shape = local.shape

    # Heat axis → affine axis mapping
    if ND == 3:
        affine_axis = 2 - split
    else:
        affine_axis = 1 - split

    shift = int(round(M[affine_axis, -1]))
    halo = abs(shift)

    # --------------------------------------------------
    # No halo needed
    # --------------------------------------------------
    if halo == 0:
        y_local = _affine_transform_local(
            ht.array(local, split=None), M
        ).larray
        return ht.array(y_local, split=split)

    # --------------------------------------------------
    # HALO PATH (correct)
    # --------------------------------------------------

    # 1. Exchange halos
    x.get_halo(halo)

    # 2. Get halo tensor
    x_halo = x.array_with_halos      # torch.Tensor
    xh = ht.array(x_halo, split=None)

    # 3. Halo-aware sampling
    device = x_halo.device
    coords = torch.meshgrid(
        *[torch.arange(s, device=device) for s in local_shape],
        indexing="ij"
    )
    coords = torch.stack(coords).reshape(ND, -1).double()

    A_inv = torch.diag(
        1.0 / torch.tensor(np.diag(A), device=device)
    )
    b_t = torch.tensor(b, device=device).reshape(ND, 1)

    src = (A_inv @ coords) - (A_inv @ b_t)
    src = src.round().long()

    # 🔴 CRITICAL HALO FIX
    src[split] += halo

    # Sample from halo tensor
    valid = torch.ones(src.shape[1], dtype=torch.bool, device=device)
    for d in range(ND):
        valid &= (src[d] >= 0) & (src[d] < x_halo.shape[d])

    out = torch.zeros(src.shape[1], device=device, dtype=x_halo.dtype)
    out[valid] = x_halo[tuple(src[:, valid])]
    out = out.reshape(local_shape)

    return ht.array(out, split=split)
