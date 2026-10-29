"""

"""

from math import radians, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import heat as ht
from heat.ndimage.affine import affine_transform
import scipy.ndimage as ndimg
from heat.ndimage.util import create_checker, center_transform

DEPTH = 32
WIDTH = 128
HEIGHT = 64

SLICE_AXIS = 0

MODE = "grid-constant"
CONSTANT_VALUE = 0.0
ORDER = 1

vol = create_checker((DEPTH, HEIGHT, WIDTH), 8)
print("finished generating image")

dims = ht.array(vol.shape)

fig, axs = plt.subplots(6, 2, figsize=(10, 16))
axs = axs.ravel()


def apply(M: ht.DNDarray, title, row_idx):

    idx = row_idx * 2

    result = affine_transform(
        vol,
        M,
        order=ORDER,
        mode=MODE,
        cval=CONSTANT_VALUE,
    )

    compare = ndimg.affine_transform(
        vol.numpy(),
        M.numpy(),
        order=ORDER,
        mode=MODE,
        cval=CONSTANT_VALUE,
    )

    if vol.ndim == 4:
        match SLICE_AXIS:
            case 0:
                slice1 = result[dims[SLICE_AXIS] // 2, :, :]
                slice2 = compare[dims[SLICE_AXIS] // 2, :, :]
            case 1:
                slice1 = result[:, dims[SLICE_AXIS] // 2, :]
                slice2 = compare[:, dims[SLICE_AXIS] // 2, :]
            case 2:
                slice1 = result[:, :, dims[SLICE_AXIS] // 2]
                slice2 = compare[:, :, dims[SLICE_AXIS] // 2]
    else:
        slice1 = result
        slice2 = compare

    result_slice = slice1.numpy().astype(np.uint8)
    compare_slice = slice2.astype(np.uint8)

    slice_dims = result_slice.shape

    axs[idx].imshow(result_slice)
    axs[idx + 1].imshow(compare_slice)
    axs[idx].set_title(title)
    axs[idx + 1].set_title("")
    axs[idx].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 1].scatter(slice_dims[1] / 2, slice_dims[0] / 2)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------
apply(ht.eye((4, 5,), dtype=ht.float32,), "Identity", 0,)
# ------------------------------------------------------------
# Rotate 20° (3D)
# ------------------------------------------------------------
THETA = radians(20)
A_ROT = ht.array(
    [
        [cos(THETA), -sin(THETA), 0, 0],
        [sin(THETA),  cos(THETA), 0, 0],
        [         0,           0, 1, 0],
        [         0,           0, 0, 1],
    ],
    dtype=ht.float32,
)
m_rot = center_transform(A_ROT, dims)
apply(m_rot, "20 degrees", 1)
# ------------------------------------------------------------
# Scale
# ------------------------------------------------------------
A_SCALE = ht.array(
    [[0.8, 0  , 0, 0],
     [0  , 1.2, 0, 0],
     [0  , 0  , 2, 0],
     [0  , 0  , 0, 1]],
    dtype=ht.float32
)
m_scale = center_transform(A_SCALE, dims)
apply(m_scale, "scale by 1.2", 2)

# ------------------------------------------------------------
# Translate
# ------------------------------------------------------------
m_tr = ht.eye((4, 5), dtype=ht.float32)
m_tr[:, 4] = [-15, 20, 30, 0]
apply(m_tr, "Translate", 3)

# ------------------------------------------------------------
# Shear
# ------------------------------------------------------------
A_SHEAR = ht.array(
    [[1, 0.3, 0.5, 0.2],
     [0, 1  , 0  , 0  ],
     [0, 0  , 1  , 0  ],
     [0, 0  , 0  , 1  ]],
    dtype=ht.float32
)
m_shear = center_transform(A_SHEAR, dims)
apply(m_shear, "Shear (0.3)", 4)

# ------------------------------------------------------------
# 3D rotation around first axis (depth)
# ------------------------------------------------------------
theta3 = radians(35)
A3 = ht.array(
    [
        [1,           0,            0, 0],
        [0, cos(theta3), -sin(theta3), 0],
        [0, sin(theta3),  cos(theta3), 0],
        [0,           0,            0, 1],
    ],
    dtype=ht.float32,
)
M3 = center_transform(A3, dims)
apply(M3, "35 deg rotation around depth axis", 5)

plt.tight_layout()
plt.show()
