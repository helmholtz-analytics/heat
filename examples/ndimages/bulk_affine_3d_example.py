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

from math import radians, cos, sin
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
import heat as ht
from heat.ndimage.affine import affine_transform

from affine_helpers import centered_linear, create_checker_volume

# SETUP
DEPTH = 32
WIDTH = 255
HEIGHT = 128

SLICE_AXIS = 2

vols = ht.stack(
    (create_checker_volume(32, 255, 128, 16), create_checker_volume(32, 255, 128, 8))
)
vols.resplit_(0)
print("finished generating image")

dims = vols.shape

fig, axs = plt.subplots(6, 2, figsize=(10, 16))
axs = axs.ravel()


def apply(Ms: ht.DNDarray, title, row_idx):
    mode = "constant"
    constant_value = 0.0
    offsets = ht.array(((30, 0, 0, 0), (30, 0, 0, 0)), dtype=ht.float32)
    idx = row_idx * 4

    # print("VOLS PRINT")
    # print(f"{vols=}")

    # print("MATRIX PRINT")
    # print(f"{Ms=}")
    result = affine_transform(
        vols,
        Ms,
        order=1,
        mode=mode,
        cval=constant_value,
        prefilter=False,
        offset=offsets,
    )

    # print(f"{result=}")

    compare = [
        ndimg.affine_transform(
            vol.numpy(),
            M.numpy(),
            order=1,
            mode=mode,
            cval=constant_value,
            prefilter=True,
            offset=offset.numpy(),
        )
        for vol, M, offset in zip(vols, Ms, offsets)
    ]

    result_numpy = result.numpy()
    # print(f"{result_numpy.shape=}")

    if result_numpy.ndim == 5:
        match SLICE_AXIS:
            case 0:
                slice1 = result_numpy[0, dims[SLICE_AXIS] // 2, :, :]
                slice2 = result_numpy[1, dims[SLICE_AXIS] // 2, :, :]
                compare1 = compare[0][dims[SLICE_AXIS] // 2, :, :]
                compare2 = compare[1][dims[SLICE_AXIS] // 2, :, :]
            case 1:
                slice1 = result_numpy[0, :, dims[SLICE_AXIS] // 2, :]
                slice2 = result_numpy[1, :, dims[SLICE_AXIS] // 2, :]
                compare1 = compare[0][:, dims[SLICE_AXIS] // 2, :]
                compare2 = compare[1][:, dims[SLICE_AXIS] // 2, :]
            case 2:
                slice1 = result_numpy[0, :, :, dims[SLICE_AXIS] // 2]
                slice2 = result_numpy[1, :, :, dims[SLICE_AXIS] // 2]
                compare1 = compare[0][:, :, dims[SLICE_AXIS] // 2]
                compare2 = compare[1][:, :, dims[SLICE_AXIS] // 2]
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

    print(f"resulting shape: {result.shape}")
    axs[idx].imshow(result_slice_1)
    axs[idx + 1].imshow(result_slice_2)
    axs[idx + 2].imshow(compare_slice_1)
    axs[idx + 3].imshow(compare_slice_2)
    axs[idx].set_title(title)
    axs[idx].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 1].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 2].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 3].scatter(slice_dims[1] / 2, slice_dims[0] / 2)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------
apply(
    ht.stack(
        (
            ht.eye(
                (
                    4,
                    4,
                ),
                dtype=ht.float32,
            ),
            ht.eye(
                (
                    4,
                    4,
                ),
                dtype=ht.float32,
            ),
        )
    ).resplit_(0),
    "Identity",
    0,
)
# ------------------------------------------------------------
# Rotate 20° (3D)
# ------------------------------------------------------------
theta = radians(20)
A_rot = ht.array(
    [
        [
            [cos(theta), -sin(theta), 0, 0],
            [sin(theta), cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        [
            [cos(-theta), -sin(-theta), 0, 0],
            [sin(-theta), cos(-theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    ],
    dtype=ht.float32,
    split=0,
)
print(f"shape after creation: {A_rot.shape}")
# M_rot = centered_linear(A_rot, dims) replaced with offset in apply!
apply(A_rot, "20 degrees", 1)
# ------------------------------------------------------------
# Scale ×1.2
# ------------------------------------------------------------
A_scale = ht.array(
    [
        [[0.8, 0, 0, 0], [0, 1.2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]],
        [[1.2, 0, 0, 0], [0, 2.2, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 1]],
    ],
    dtype=ht.float32,
    split=0,
)
print(f"shape after creation: {A_scale.shape}")
M_scale = centered_linear(A_scale, dims)
print("after centered linear {M_scale=}")
apply(M_scale, "scale by 1.2", 2)

# # ------------------------------------------------------------
# # Translate (+20, −20)
# # ------------------------------------------------------------
# M_tr = ht.eye((4, 5), dtype=ht.float32)
# M_tr[:, 4] = [-15, 20, 30, 0]
# apply(M_tr, "Translate (+20, −20)", 3)

# # ------------------------------------------------------------
# # Shear (0.3)
# # ------------------------------------------------------------
# A_shear = ht.array(
#     [[1, 0.3, 0.5, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=ht.float32
# )
# M_shear = centered_linear(A_shear, dims)
# apply(M_shear, "Shear (0.3)", 4)

# # ------------------------------------------------------------
# # 3D rotation around Z-axis (35°)
# # ------------------------------------------------------------
# theta3 = radians(35)
# A3 = ht.array(
#     [
#         [1, 0, 0, 0],
#         [0, cos(theta3), -sin(theta3), 0],
#         [0, sin(theta3), cos(theta3), 0],
#         [0, 0, 0, 1],
#     ],
#     dtype=ht.float32,
# )
# M3 = centered_linear(A3, dims)
# apply(M3, "35 deg rotation around depth axis", 5)

plt.tight_layout()
plt.show()
