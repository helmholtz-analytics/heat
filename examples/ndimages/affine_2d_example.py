"""
Example for 2D images with 3 operations popup view
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as dnimg
from PIL import Image
import heat as ht
from heat.ndimage.affine import affine_transform


def checker_image(w: int, h: int, checker_size: int):
    new_image = Image.new("RGB", (h, w), (255, 0, 0))  # create a new 15x15 image
    pixels = new_image.load()  # create the pixel map

    box_size = checker_size
    for i in range(0, h, box_size):
        for j in range(0, w, box_size):
            y = int(i / box_size)
            x = int(j / box_size)
            if (y & 1) ^ (x & 1):
                for di in range(box_size):
                    for dj in range(box_size):
                        pixels[i + di, j + dj] = (0, 0, 0)
            else:
                for di in range(box_size):
                    for dj in range(box_size):
                        pixels[i + di, j + dj] = (0, 0, (50 + i + j) % 255)
    return new_image


# ------------------------------------------------------------
# Load RGB image
# ------------------------------------------------------------
img: Image = checker_image(256, 512, 32)

img_np = np.asarray(img, dtype=np.float32)  #
print(f"shape of image as numpy array {img_np.shape}")  # HWC

# HWC
heat_img = ht.array(img_np)  # HWC
print(f"shape of image converted from numpy to heat array {heat_img.shape}")

H, W = img_np.shape[:2]
cx, cy = W / 2, H / 2  # NOTE: (x, y)
img_center = np.array([cx, cy, 0], dtype=np.float32)
# img_center = np.array([0, 0, 0], dtype=np.float32)

fig, axs = plt.subplots(5, 2, figsize=(10, 16))
axs = axs.ravel()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def centered_linear(A_xy):
    """
    Build a 3×4 affine matrix for a linear transform A applied about the image center.
    """
    b = img_center - A_xy @ img_center
    # b =  img_center + A_xy @ (-img_center)
    # b = np.array([0, 256, 0], dtype=np.float32)
    return np.hstack([A_xy, b[:, None]]).astype(np.float32)


def apply(M: np.ndarray, title, idx, mode="nearest", constant_value=0.0):

    print(f"matrix shape: {M.shape}")
    print(M)
    print(f"image shape: {heat_img.shape}")

    index = idx * 2

    result = affine_transform(
        heat_img, ht.array(M), order=0, mode=mode, cval=constant_value, prefilter=True
    )

    compare = dnimg.affine_transform(
        img_np, M, order=0, mode=mode, cval=constant_value, prefilter=True
    )
    axs[index].imshow(result.larray.permute(0, 1, 2).cpu().numpy().astype(np.uint8))
    axs[index + 1].imshow(compare.astype(np.uint8))
    axs[index].set_title(title)
    axs[index].axis("off")
    axs[index + 1].set_title("")
    axs[index + 1].axis("off")
    axs[index].scatter(img_center[0], img_center[1])
    axs[index + 1].scatter(img_center[0], img_center[1])


# ------------------------------------------------------------
# Identity
# ------------------------------------------------------------
apply(np.eye(3, 4, dtype=np.float32), "Identity", 0)

# ------------------------------------------------------------
# Translation +100 px RIGHT  (b = [tx, ty] = [100, 0])
# Tip: use mode="constant" to make the shift super obvious
# ------------------------------------------------------------
M_tr = np.eye(3, 4, dtype=np.float32)
M_tr[:, 3] = [0, 256, 0]  # (tx, ty, tz)
apply(M_tr, "Translate +100px (right)", 1, mode="constant", constant_value=0.0)

# ------------------------------------------------------------
# Rotate 30° around center (in x,y coords)
# ------------------------------------------------------------
theta = np.deg2rad(30)
A_rot = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
    dtype=np.float32,
)
apply(centered_linear(A_rot), "Rotate 30°", 2)

# ------------------------------------------------------------
# Scale ×1.5 around center (in x,y coords)
# ------------------------------------------------------------
A_scale = np.array([[0.75, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=np.float32)
apply(centered_linear(A_scale), "Scale ×1.5", 3)

# ------------------------------------------------------------
# Combo: centered (scale→rotate) + then translate (tx,ty)
# ------------------------------------------------------------
A_combo = A_rot @ A_scale
t = np.array([100, -50, 0], dtype=np.float32)  # (tx, ty)
b_combo = img_center - A_combo @ img_center + t
M_combo = np.hstack([A_combo, b_combo[:, None]]).astype(np.float32)
apply(M_combo, "Combo", 4)

# ------------------------------------------------------------
# Hide unused subplot
# ------------------------------------------------------------

plt.tight_layout()
plt.show()
