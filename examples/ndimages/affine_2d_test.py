import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import heat as ht
from heat.ndimage.affine import affine_transform

# ------------------------------------------------------------
# Load RGB image
# ------------------------------------------------------------
img = Image.open(
    "PATH"
).convert("RGB")

img_np = np.asarray(img, dtype=np.float32)

# HWC → CHW
x = ht.array(img_np.transpose(2, 0, 1))

H, W = img_np.shape[:2]
cx, cy = W / 2, H / 2               # NOTE: (x, y)
c = np.array([cx, cy], dtype=np.float32)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.ravel()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def centered_linear(A_xy):
    """
    Build a 2×3 affine matrix for a linear transform A applied about the image center.

    IMPORTANT: Heat interprets M in affine coords (x, y).
    So center must be (cx, cy) and translation b is (tx, ty).
    """
    b = c - A_xy @ c
    return np.hstack([A_xy, b[:, None]]).astype(np.float32)


def apply(M, title, idx, mode="nearest", constant_value=0.0):
    y = affine_transform(
        x,
        M,
        order=0,               # nearest neighbor keeps colors crisp
        mode=mode,
        constant_value=constant_value,
        expand=False
    )
    axs[idx].imshow(
        y.larray.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    )
    axs[idx].set_title(title)
    axs[idx].axis("off")


# ------------------------------------------------------------
# Identity
# ------------------------------------------------------------
apply(np.eye(2, 3, dtype=np.float32), "Identity", 0)

# ------------------------------------------------------------
# Translation +100 px RIGHT  (b = [tx, ty] = [100, 0])
# Tip: use mode="constant" to make the shift super obvious
# ------------------------------------------------------------
M_tr = np.eye(2, 3, dtype=np.float32)
M_tr[:, 2] = [1000, 0]   # (tx, ty)
apply(M_tr, "Translate +100px (right)", 1, mode="constant", constant_value=0.0)

# ------------------------------------------------------------
# Rotate 30° around center (in x,y coords)
# ------------------------------------------------------------
theta = np.deg2rad(30)
A_rot = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
], dtype=np.float32)
apply(centered_linear(A_rot), "Rotate 30°", 2)

# ------------------------------------------------------------
# Scale ×1.5 around center (in x,y coords)
# ------------------------------------------------------------
A_scale = np.array([
    [1.5, 0],
    [0, 1.5]
], dtype=np.float32)
apply(centered_linear(A_scale), "Scale ×1.5", 3)

# ------------------------------------------------------------
# Combo: centered (scale→rotate) + then translate (tx,ty)
# ------------------------------------------------------------
A_combo = A_rot @ A_scale
t = np.array([100, -50], dtype=np.float32)  # (tx, ty)
b_combo = c - A_combo @ c + t
M_combo = np.hstack([A_combo, b_combo[:, None]]).astype(np.float32)
apply(M_combo, "Combo", 4)

# ------------------------------------------------------------
# Hide unused subplot
# ------------------------------------------------------------
axs[5].axis("off")

plt.tight_layout()
plt.show()
