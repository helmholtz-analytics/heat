"""
End-user demo: affine transformations on a 3D MRI volume
(using Heat affine_transform – final implementation).
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import heat as ht
from heat.ndimage.affine import affine_transform


# ============================================================
# Helper: normalize output to (D,H,W)
# ============================================================

def to_volume(y):
    """
    Convert Heat affine output to plain (D,H,W) NumPy array,
    regardless of whether a leading dimension exists.
    """
    y_np = y.numpy()
    if y_np.ndim == 4:   # (1,D,H,W)
        return y_np[0]
    return y_np          # (D,H,W)


# ============================================================
# STEP 1: Load MRI
# ============================================================

nii = nib.load(
    "/Users/marka.k/1900_Image_transformations/heat/heat/datasets/flair.nii.gz"
)
x_np = nii.get_fdata().astype(np.float32)
x = ht.array(x_np)

D, H, W = x_np.shape
cx, cy, cz = D / 2, H / 2, W / 2

print("Loaded MRI with shape:", x_np.shape)


# ============================================================
# STEP 2: Define affine matrices
# ============================================================

# 1️⃣ Identity
M_identity = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
]

# 2️⃣ Scaling (zoom around center)
s = 1.4
M_scale = [
    [s, 0, 0, cx * (1 - s)],
    [0, s, 0, cy * (1 - s)],
    [0, 0, s, cz * (1 - s)],
]

# 3️⃣ Rotation (around Z axis)
theta = np.deg2rad(20)
c, s_ = np.cos(theta), np.sin(theta)
M_rotate = [
    [ c, -s_, 0, cx - c * cx + s_ * cy],
    [ s_,  c, 0, cy - s_ * cx - c * cy],
    [ 0,  0, 1, 0],
]

# 4️⃣ Translation
tx = 15
M_translate = [
    [1, 0, 0, tx],
    [0, 1, 0,  0],
    [0, 0, 1,  0],
]


# ============================================================
# STEP 3: Apply affine transforms
# ============================================================

print("Applying affine transformations...")

y_identity  = to_volume(affine_transform(x, M_identity,  order=1))
y_scale     = to_volume(affine_transform(x, M_scale,     order=1))
y_rotate    = to_volume(affine_transform(x, M_rotate,    order=1))
y_translate = to_volume(affine_transform(x, M_translate, order=1))

print("Transformations complete.")


# ============================================================
# STEP 4: Save transformed volumes
# ============================================================

nib.save(nib.Nifti1Image(y_identity,  nii.affine), "mri_identity.nii.gz")
nib.save(nib.Nifti1Image(y_scale,     nii.affine), "mri_scaled.nii.gz")
nib.save(nib.Nifti1Image(y_rotate,    nii.affine), "mri_rotated.nii.gz")
nib.save(nib.Nifti1Image(y_translate, nii.affine), "mri_translated.nii.gz")

print("Saved transformed NIfTI files.")


# ============================================================
# STEP 5: Visualization (5x5 grid)
# ============================================================

slice_indices = np.linspace(0, D - 1, 5, dtype=int)

volumes = [
    x_np,
    y_identity,
    y_scale,
    y_rotate,
    y_translate,
]

titles = [
    "Original",
    "Identity",
    "Scale (1.4×)",
    "Rotate (20°)",
    "Translate (+x)",
]

fig, axes = plt.subplots(5, 5, figsize=(12, 12))

for row in range(5):
    for col in range(5):
        axes[row, col].imshow(
            volumes[row][slice_indices[col]],
            cmap="gray"
        )
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(titles[row], fontsize=10)

        if row == 0:
            axes[row, col].set_title(f"Slice {slice_indices[col]}", fontsize=9)

plt.tight_layout()
plt.show()
