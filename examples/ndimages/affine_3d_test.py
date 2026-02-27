"""
Non-distributed affine demo on a NIfTI volume (Heat).

Applies:
- 2D rotation (centered)
- 2D scaling (centered)
- 2D translation
- 2D shear
- 3D rotation (centered)

Handles Heat channel dimensions correctly.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import heat as ht

from heat.ndimage.affine import affine_transform


# ============================================================
# Helpers
# ============================================================

def centered_linear_2d(A, H, W):
    """2×3 affine around image center (y, x)."""
    c = np.array([H / 2, W / 2], dtype=np.float32)
    b = c - A @ c
    return np.hstack([A, b[:, None]]).astype(np.float32)


def centered_linear_3d(A, D, H, W):
    """3×4 affine around volume center (z, y, x)."""
    c = np.array([D / 2, H / 2, W / 2], dtype=np.float32)
    b = c - A @ c
    return np.hstack([A, b[:, None]]).astype(np.float32)


def show(title, img):
    """Safe grayscale visualization."""
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")


# ============================================================
# Load MRI
# ============================================================

nii = nib.load(
    "PATH"
)
x_np = nii.get_fdata().astype(np.float32)

print("Loaded MRI:", x_np.shape)

# Heat array (NO split)
x = ht.array(x_np)

D, H, W = x_np.shape
mid = D // 2

# 2D middle slice
slice2d = x[mid]


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(12, 8))

# ------------------------------------------------------------
# Original
# ------------------------------------------------------------
plt.subplot(2, 3, 1)
show("Original (middle slice)", slice2d.larray.cpu().numpy())

# ------------------------------------------------------------
# Rotate 20° (2D)
# ------------------------------------------------------------
theta = np.deg2rad(20)
A_rot = np.array(
    [[np.cos(theta), -np.sin(theta)],
     [np.sin(theta),  np.cos(theta)]],
    dtype=np.float32,
)
M_rot = centered_linear_2d(A_rot, H, W)
y_rot = affine_transform(slice2d, M_rot, order=1)

plt.subplot(2, 3, 2)
show("Rotate 20°", y_rot.larray.cpu().numpy())

# ------------------------------------------------------------
# Scale ×1.2
# ------------------------------------------------------------
A_scale = np.array([[1.2, 0], [0, 1.2]], dtype=np.float32)
M_scale = centered_linear_2d(A_scale, H, W)
y_scale = affine_transform(slice2d, M_scale, order=1)

plt.subplot(2, 3, 3)
show("Scale ×1.2", y_scale.larray.cpu().numpy())

# ------------------------------------------------------------
# Translate (+20, −20)
# NOTE: backward warping → negate translation
# ------------------------------------------------------------
M_tr = np.eye(2, 3, dtype=np.float32)
M_tr[:, 2] = [-20, 20]
y_tr = affine_transform(slice2d, M_tr, order=1)

plt.subplot(2, 3, 4)
show("Translate (+20, −20)", y_tr.larray.cpu().numpy())

# ------------------------------------------------------------
# Shear (0.3)
# ------------------------------------------------------------
A_shear = np.array([[1, 0.3], [0, 1]], dtype=np.float32)
M_shear = centered_linear_2d(A_shear, H, W)
y_shear = affine_transform(slice2d, M_shear, order=1)

plt.subplot(2, 3, 5)
show("Shear (0.3)", y_shear.larray.cpu().numpy())

# ------------------------------------------------------------
# 3D rotation around Z-axis (35°)
# ------------------------------------------------------------
theta3 = np.deg2rad(35)
A3 = np.array(
    [[1, 0, 0],
     [0, np.cos(theta3), -np.sin(theta3)],
     [0, np.sin(theta3),  np.cos(theta3)]],
    dtype=np.float32,
)
M3 = centered_linear_3d(A3, D, H, W)
y3 = affine_transform(x, M3, order=1)

# REMOVE channel dimension before slicing
vol3 = y3.larray.squeeze(0)   # (D, H, W)

plt.subplot(2, 3, 6)
show("3D Rotation (Z-axis 35°)", vol3[mid].cpu().numpy())

plt.tight_layout()
plt.show()
