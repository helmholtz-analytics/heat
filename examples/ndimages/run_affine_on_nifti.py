"""
End-user example: apply an affine transformation to a 3D NIfTI image
and visualize the result directly in Python.

This script:
1. Loads x.nii.gz
2. Applies a 3D affine transformation using heat.ndimage.affine_transform
3. Saves the transformed volume as x_transformed.nii.gz
4. Displays a side-by-side comparison of the middle slice

Requirements:
- nibabel
- matplotlib
- heat
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import heat as ht
from heat.ndimage.affine import affine_transform


# ============================================================
# STEP 1: Load NIfTI file
# ============================================================

print("Loading x.nii.gz ...")

nii = nib.load("heat/datasets/flair.nii.gz")
x_np = nii.get_fdata().astype(np.float32)

print("Input shape:", x_np.shape)


# ============================================================
# STEP 2: Convert to Heat array
# ============================================================

x = ht.array(x_np)

print("Converted to Heat array.")


# ============================================================
# STEP 3: Define affine transform (3D)
# ============================================================

"""
Affine matrix (3x4):

[ a11 a12 a13 tx ]
[ a21 a22 a23 ty ]
[ a31 a32 a33 tz ]

Below: translate volume by +20 voxels in x-direction
"""
D, H, W = x_np.shape
cx, cy, cz = D / 2, H / 2, W / 2
s = 1.4

M = [
    [s, 0, 0, cx * (1 - s)],
    [0, s, 0, cy * (1 - s)],
    [0, 0, s, cz * (1 - s)],
]


# ============================================================
# STEP 4: Apply affine transform
# ============================================================

print("Applying affine transform...")

y = affine_transform(
    x,
    M,
    order=1,           # bilinear interpolation
    mode="nearest"
)

print("Transformation complete.")


# ============================================================
# STEP 5: Convert back to NumPy
# ============================================================

y_np = y.numpy()

# Remove leading batch/channel dimension if present
if y_np.ndim == 4:
    y_np = y_np[0]

print("Output shape:", y_np.shape)


# ============================================================
# STEP 6: Save transformed volume
# ============================================================

out_nii = nib.Nifti1Image(y_np, affine=nii.affine)
nib.save(out_nii, "x_transformed.nii.gz")

print("Saved x_transformed.nii.gz")


# ============================================================
# STEP 7: Visualize middle slice
# ============================================================

mid = x_np.shape[0] // 2

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(x_np[mid], cmap="gray")
ax[0].set_title("Original (middle slice)")
ax[0].axis("off")

ax[1].imshow(y_np[mid], cmap="gray")
ax[1].set_title("Transformed (middle slice)")
ax[1].axis("off")

plt.tight_layout()
plt.show()
