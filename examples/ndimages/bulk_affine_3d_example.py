"""
Demo to compare scipy and heat affine transforms
"""

from math import radians, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import heat as ht
from heat.ndimage.affine import affine_transform
from heat.ndimage.util import center_transform, create_checker

# SETUP
DEPTH = 128
WIDTH = 256
HEIGHT = 4

SLICE_AXIS = 2 #wich axis is not displayed in visualization

MODE = "grid-constant"
CONSTANT_VALUE = 128

CHECKER_1 = create_checker((DEPTH, WIDTH, HEIGHT), 16, dtype=ht.float32)
CHECKER_2 = create_checker((DEPTH, WIDTH, HEIGHT), 8, dtype=ht.float32)
VOLS = ht.stack((CHECKER_1, CHECKER_2))
VOLS.resplit_(0)

OFFSETS = ht.array(((20, 0, 0, 0), (30, 0, 0, 0)), dtype=ht.float32)

dims = VOLS.shape

fig, axs = plt.subplots(6, 2, figsize=(10, 16))
axs = axs.ravel()


def apply(matrices: ht.DNDarray, row_idx):

    idx = row_idx * 4

    # heat
    result = affine_transform(
        VOLS,
        matrices,
        order=1,
        mode=MODE,
        cval=CONSTANT_VALUE,
        offset=OFFSETS,
    )

    # scipy
    compare = [
        ndimg.affine_transform(
            vol.numpy(),
            matrix.numpy(),
            order=1,
            mode=MODE,
            cval=CONSTANT_VALUE,
            offset=offset.numpy(),
        )
        for vol, matrix, offset in zip(VOLS, matrices, OFFSETS)
    ]

    result_numpy = result.numpy()

    if result_numpy.ndim == 5:
        match SLICE_AXIS:
            case 0:
                slice1 = result_numpy[0, dims[SLICE_AXIS+1] // 2, :, :]
                slice2 = result_numpy[1, dims[SLICE_AXIS+1] // 2, :, :]
                compare1 = compare[0][dims[SLICE_AXIS+1] // 2, :, :]
                compare2 = compare[1][dims[SLICE_AXIS+1] // 2, :, :]
            case 1:
                slice1 = result_numpy[0, :, dims[SLICE_AXIS+1] // 2, :]
                slice2 = result_numpy[1, :, dims[SLICE_AXIS+1] // 2, :]
                compare1 = compare[0][:, dims[SLICE_AXIS+1] // 2, :]
                compare2 = compare[1][:, dims[SLICE_AXIS+1] // 2, :]
            case 2:
                slice1 = result_numpy[0, :, :, dims[SLICE_AXIS+1] // 2]
                slice2 = result_numpy[1, :, :, dims[SLICE_AXIS+1] // 2]
                compare1 = compare[0][:, :, dims[SLICE_AXIS+1] // 2]
                compare2 = compare[1][:, :, dims[SLICE_AXIS+1] // 2]
    else:
        slice1 = result[0]
        slice2 = result[1]
        compare1 = compare[0]
        compare2 = compare[1]

    result_slice_1 = slice1.astype(np.uint8)
    result_slice_2 = slice2.astype(np.uint8)

    compare_slice_1 = compare1.astype(np.uint8)
    compare_slice_2 = compare2.astype(np.uint8)
    slice_dims = result_slice_1.shape

    axs[idx    ].imshow(result_slice_1)
    axs[idx + 1].imshow(result_slice_2)
    axs[idx + 2].imshow(compare_slice_1)
    axs[idx + 3].imshow(compare_slice_2)
    axs[idx    ].set_title("heat")
    axs[idx + 2].set_title("scipy")
    axs[idx    ].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 1].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 2].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 3].scatter(slice_dims[1] / 2, slice_dims[0] / 2)


# ------------------------------------------------------------
# Identity
# ------------------------------------------------------------
matrix= ht.expand_dims(ht.eye((4,4),dtype=ht.float32),0)
matrix = ht.tile(matrix, [2,1,1])
apply(matrix, 0)


# ------------------------------------------------------------
# Rotate 20° and -20°
# Offset values specified globally get applied because no
# translation is specified by the input matrix
# ------------------------------------------------------------
theta = radians(20)
matrix_rot = ht.array(
    [
        [
            [cos(theta), -sin(theta), 0, 0],
            [sin(theta),  cos(theta), 0, 0],
            [         0,           0, 1, 0],
            [         0,           0, 0, 1],
        ],
        [
            [cos(-theta), -sin(-theta), 0, 0],
            [sin(-theta),  cos(-theta), 0, 0],
            [          0,            0, 1, 0],
            [          0,            0, 0, 1],
        ],
    ],
    dtype=ht.float32,
    split=0,
)
apply(matrix_rot, 1)


# ------------------------------------------------------------
# Scaling
# gets centered, so there is a translation in the matrix itself
# offset variable gets ignored
# ------------------------------------------------------------
matrix_scale = ht.array(
    [
        [
            [0.8, 0  , 0, 0],
            [0  , 1.2, 0, 0],
            [0  , 0  , 2, 0],
            [0  , 0  , 0, 1]
        ],
        [
            [1.2, 0,   0  , 0],
            [0  , 2.2, 0  , 0],
            [0,   0,   0.3, 0],
            [0,   0,   0  , 1]],
    ],
    dtype=ht.float32,
    split=0,
)
matrix_scale = center_transform(matrix_scale, dims)
apply(matrix_scale, 2)

plt.tight_layout()
plt.show()
