"""
Example for 2D images with 3 operations popup view
"""

import numpy as np
from math import radians, cos, sin
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import heat as ht
from heat.ndimage.affine import affine_transform

from heat.ndimage.util import create_checker, centered_linear

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
SIZE = (256, 256)
SCALE = (1.5, 1.1)
ROTATE = np.deg2rad(30)
TRANSLATE = (100, 50)

PADDING_VALUE=0.0
OFFSET = ht.array((30, 0, 0))
ORDER = 1
MODE="grid-constant"

# image tensor is interpreded as Height x Width x Color
img_heat: ht.DNDarray = create_checker(SIZE, 32)

img_np = img_heat.numpy()
dims = img_np.shape

fig, axs = plt.subplots(5, 2, figsize=(10, 16))
axs = axs.ravel()


def apply(affine_matrix: ht.DNDarray, title, row_idx):
    idx = row_idx * 2

    result = affine_transform(
        img_heat,
        affine_matrix,
        order=ORDER,
        mode=MODE,
        cval=PADDING_VALUE,
        prefilter=True,
        offset=OFFSET,
    )

    compare = ndimg.affine_transform(
        img_np,
        affine_matrix.numpy(),
        order=ORDER,
        mode=MODE,
        cval=PADDING_VALUE,
        prefilter=True,
        offset=OFFSET.numpy(),
    )

    axs[idx].imshow(result.numpy().astype(np.uint8))
    axs[idx + 1].imshow(compare.astype(np.uint8))
    axs[idx].set_title(title)
    axs[idx + 1].set_title("")
    axs[idx].scatter(dims[1] / 2, dims[0] / 2)
    axs[idx + 1].scatter(dims[1] / 2, dims[0] / 2)


# ------------------------------------------------------------
# Identity
# ------------------------------------------------------------
apply(ht.eye((3, 4), dtype=ht.float32), "Identity", 0)

# ------------------------------------------------------------
# Translation
# ------------------------------------------------------------
M_TR = ht.eye((3, 4), dtype=ht.float32)
M_TR[:, 3] = [TRANSLATE[0], TRANSLATE[1], 0]  # (tx, ty, tz)
apply(M_TR, f"Translate {TRANSLATE}", 1)

# ------------------------------------------------------------
# Rotate 30° around center (in x,y coords)
# ------------------------------------------------------------
THETA = radians(30)
A_ROT = ht.array(
    [
        [cos(THETA), -sin(ROTATE), 0],
        [sin(ROTATE), cos(THETA), 0],
        [0, 0, 1],
    ],
    dtype=ht.float32,
)
apply(A_ROT, f"Rotate {ROTATE} with seperate offset vector", 2)

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------
A_SCALE = ht.array([[SCALE[0], 0, 0], [0, SCALE[1], 0], [0, 0, 1]], dtype=ht.float32)
apply(centered_linear(A_SCALE, dims), f"Scale {SCALE}", 3)

# ------------------------------------------------------------
# Combo: centered (scale→rotate) + then translate (tx,ty)
# ------------------------------------------------------------
a_combo = A_ROT @ A_SCALE
t = ht.array([100, -50, 0], dtype=ht.float32)  # (tx, ty)
img_center = ht.array(dims) / 2
b_combo = img_center - a_combo @ img_center + t
m_combo = ht.hstack([a_combo, b_combo[:, None]]).astype(np.float32)
apply(m_combo, "Combo", 4)

plt.tight_layout()
plt.show()
