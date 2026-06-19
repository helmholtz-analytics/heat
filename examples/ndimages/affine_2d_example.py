"""
Example for 2D images with 3 operations popup view
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
from PIL import Image
import heat as ht
from heat.ndimage.affine import affine_transform

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
SIZE = (256, 512)
SCALE = (1.5, 0.75)
ROTATE = np.deg2rad(30)
TRANSLATE = (100, 50)


def create_checker_image(w: int, h: int, checker_size: int) -> Image.Image:
    """creates a PIL image for testing

    :param w: _description_
    :type w: int
    :param h: _description_
    :type h: int
    :param checker_size: _description_
    :type checker_size: int
    :return: _description_
    :rtype: _type_
    """
    new_image: Image.Image = Image.new(
        "RGB", (h, w), (255, 0, 0)
    )  # create a new 15x15 image
    pixels = new_image.load()  # create the pixel map

    box_size = checker_size
    for i in range(0, h, box_size):
        for j in range(0, w, box_size):
            y = int(i / box_size)
            x = int(j / box_size)
            if (y & 1) ^ (x & 1):
                for di in range(box_size):
                    for dj in range(box_size):
                        pixels[i + di, j + dj] = (255, 255, 255)
            else:
                for di in range(box_size):
                    for dj in range(box_size):
                        pixels[i + di, j + dj] = (0, 0, (50 + i + j) % 255)
    return new_image


# ------------------------------------------------------------
# Load RGB image
# ------------------------------------------------------------

img: Image = create_checker_image(*SIZE, 32)

img_np = np.asarray(img, dtype=np.float32)  #
print(f"shape of image as numpy array {img_np.shape}")  # HWC

img_heat = ht.array(img_np)  # HWC
print(f"shape of image converted from numpy to heat array {img_heat.shape}")

H, W = img_np.shape[:2]
cx, cy = H / 2, W / 2
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
    b = img_center - (A_xy @ img_center)
    # b = np.array([0, 0, 0], dtype=np.float32)
    return np.hstack([A_xy, b[:, None]]).astype(np.float32)


def apply(M: np.ndarray, title, row_idx, mode="constant", constant_value=0.0):

    heat_M = ht.array(M)

    idx = row_idx * 2

    result = affine_transform(
        img_heat,
        heat_M,
        offset=100,
        order=1,
        mode=mode,
        cval=constant_value,
        prefilter=True,
    )

    compare = ndimg.affine_transform(
        img_np, M, offset=100, order=1, mode=mode, cval=constant_value, prefilter=True
    )
    print(f"resulting shape: {result.shape}")
    print(f"compare shape: {compare.shape}")
    axs[idx].imshow(result.numpy().astype(np.uint8))
    axs[idx + 1].imshow(compare.astype(np.uint8))
    axs[idx].set_title(title)
    axs[idx].axis("off")
    axs[idx + 1].set_title("")
    axs[idx + 1].axis("off")
    axs[idx].scatter(img_center[1], img_center[0])
    axs[idx + 1].scatter(img_center[1], img_center[0])


# ------------------------------------------------------------
# Identity
# ------------------------------------------------------------
apply(np.eye(3, 4, dtype=np.float32), "Identity", 0)

# ------------------------------------------------------------
# Translation
# ------------------------------------------------------------
M_tr = np.eye(3, 4, dtype=np.float32)
M_tr[:, 3] = [TRANSLATE[0], TRANSLATE[1], 0]  # (tx, ty, tz)
apply(M_tr, f"Translate {TRANSLATE}", 1, mode="constant", constant_value=0.0)

# ------------------------------------------------------------
# Rotate 30° around center (in x,y coords)
# ------------------------------------------------------------
theta = np.deg2rad(30)
A_rot = np.array(
    [
        [np.cos(theta), -np.sin(ROTATE), 0],
        [np.sin(ROTATE), np.cos(theta), 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)
apply(centered_linear(A_rot), f"Rotate {ROTATE}", 2)

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------
A_scale = np.array([[SCALE[0], 0, 0], [0, SCALE[1], 0], [0, 0, 1]], dtype=np.float32)
apply(centered_linear(A_scale), f"Scale {SCALE}", 3)

# ------------------------------------------------------------
# Combo: centered (scale→rotate) + then translate (tx,ty)
# ------------------------------------------------------------
A_combo = A_rot @ A_scale
t = np.array([100, -50, 0], dtype=np.float32)  # (tx, ty)
b_combo = img_center - A_combo @ img_center + t
M_combo = np.hstack([A_combo, b_combo[:, None]]).astype(np.float32)
apply(M_combo, "Combo", 4)


plt.tight_layout()
plt.show()
