import numpy as np
import torch
import heat as ht


# ============================================================
# Utility: normalize input → (N, C, spatial…)
# Why? Sampling code expects a batch+channel dimension.
# ============================================================

def _normalize_input(x, ND):
    orig_shape = x.shape
    t = x.larray

    # 2D image: (H, W) or (C, H, W)
    if ND == 2:
        if x.ndim == 2:   # (H,W)
            t = t.unsqueeze(0).unsqueeze(0)   # → (1,1,H,W)
        elif x.ndim == 3: # (C,H,W)
            t = t.unsqueeze(0)               # → (1,C,H,W)

    # 3D image: (D,H,W) or (C,D,H,W)
    else:
        if x.ndim == 3:   # (D,H,W)
            t = t.unsqueeze(0).unsqueeze(0)   # → (1,1,D,H,W)
        elif x.ndim == 4: # (C,D,H,W)
            t = t.unsqueeze(0)               # → (1,C,D,H,W)

    return t, orig_shape


# ============================================================
# Utility: build coordinate grid in Heat order
# (z,y,x) for 3D, (y,x) for 2D
# Why? This represents output pixel positions.
# ============================================================

def _make_grid(spatial, device):
    if len(spatial) == 2:
        H, W = spatial
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx], dim=0)  # (2,H,W)
    else:
        D, H, W = spatial
        zs = torch.arange(D, device=device)
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
        return torch.stack([gz, gy, gx], dim=0)  # (3,D,H,W)


# ============================================================
# Padding helper for nearest/bilinear sampling
# Why? Coordinates may go outside image → we map to valid pixels.
# ============================================================

def _apply_padding(pix, spatial, mode, constant_value):
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
            # reflect between [0, size-1]
            r = torch.abs(p)
            r = torch.remainder(r, 2 * size - 2)
            final[d] = torch.where(r < size, r, 2 * size - 2 - r)

    return final, valid


# ============================================================
# Nearest neighbor sampler
# Why? Fastest and simplest.
# ============================================================

def _nearest_sample(x_local, coords_h, mode, constant_value):
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
# Why? Smooth transformations for images.
# ============================================================

def _bilinear_sample(x_local, coords_h, mode, constant_value):
    if coords_h.shape[0] != 2:   # fallback for 3D
        return _nearest_sample(x_local, coords_h, mode, constant_value)

    y, x = coords_h
    H, W = x_local.shape[2], x_local.shape[3]

    # neighbors
    y0 = torch.floor(y).long()
    x0 = torch.floor(x).long()
    y1 = y0 + 1
    x1 = x0 + 1

    # clamp neighbors
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)
    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)

    Ia = x_local[:, :, y0c, x0c]
    Ib = x_local[:, :, y0c, x1c]
    Ic = x_local[:, :, y1c, x0c]
    Id = x_local[:, :, y1c, x1c]

    # interpolation weights
    wy = y - y0.float()
    wx = x - x0.float()

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd

    return out


# ============================================================
# Main Heat affine transform
# Implements PyTorch/SciPy-style backward warp:
#   input_coord = A_inv @ output_coord - A_inv @ b
# ============================================================

def affine_transform(
    x,
    M,
    order=0,
    mode="nearest",
    constant_value=0.0,
    expand=False,   # kept for later; currently no-op
):

    M = np.asarray(M, dtype=np.float32)

    # Determine 2D or 3D
    if M.shape == (2, 3):
        ND = 2
    elif M.shape == (3, 4):
        ND = 3
    else:
        raise ValueError("Affine matrix must be 2x3 (2D) or 3x4 (3D).")

    # Normalize input
    x_local, orig_shape = _normalize_input(x, ND)
    device = x_local.device

    # Matrix: A (rotation/scale/shear) and b (translation)
    A = torch.tensor(M[:, :ND], device=device)
    b = torch.tensor(M[:, ND:], device=device).reshape(ND, 1)

    # Backward warp matrix
    A_inv = torch.inverse(A)

    spatial = x_local.shape[2:]  # H,W or D,H,W

    # Output grid in Heat axis order
    grid_h = _make_grid(spatial, device)  # (ND, ...)

    # Flatten grid_h → (ND, P)
    P = int(np.prod(spatial))
    grid_flat = grid_h.reshape(ND, P).float()

    # Reorder to PyTorch axis order for math:
    if ND == 2:
        # Heat: (y,x), PT: (x,y)
        grid_pt = grid_flat[[1, 0], :]
    else:
        # Heat: (z,y,x), PT: (x,y,z)
        z, y, x = grid_flat
        grid_pt = torch.stack([x, y, z], dim=0)

    # Backward warp: input_coord = A_inv @ grid_pt - A_inv @ b
    coords_pt = (A_inv @ grid_pt) - (A_inv @ b)

    # Convert back to Heat order
    if ND == 2:
        coords_h = coords_pt[[1, 0], :].reshape((2, *spatial))
    else:
        # PT (x,y,z) → Heat (z,y,x)
        cx, cy, cz = coords_pt
        coords_h = torch.stack(
            [cz.reshape(spatial),
             cy.reshape(spatial),
             cx.reshape(spatial)], dim=0
        )

    # Choose interpolation
    if order == 0:
        out_local = _nearest_sample(x_local, coords_h, mode, constant_value)
    else:
        out_local = _bilinear_sample(x_local, coords_h, mode, constant_value)

    # Convert back to original shape
    out = out_local.squeeze(0)  # remove batch dim
# x.split should be a value (None/int/tuple), but some Heat objects expose it as a METHOD
    split_val = None
    if hasattr(x, "split") and not callable(x.split):
       split_val = x.split

    out = ht.array(out, split=split_val)

    return out.reshape(orig_shape)
