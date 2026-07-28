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
import heat as ht
from heat.ndimage.affine import affine_transform
import scipy.ndimage as ndimg


# ============================================================
# Helpers
# ============================================================
def create_checker_volume(d: int, w: int, h: int, checker_size: int) -> ht.DNDarray:
    """creates a DND array for testing

    :param w: width
    :type w: int
    :param h: height
    :type h: int
    :param checker_size: size in pixel that one checker should have
    :type checker_size: int
    :return: heat array
    :rtype: heat.DNDarray
    """
    print("start creating test volume")
    array = ht.full([d, h, w, 3], 255, dtype=ht.float32)

    for k in range(0, d, checker_size):
        for i in range(0, h, checker_size):
            for j in range(0, w, checker_size):
                y = i // checker_size
                x = j // checker_size
                z = k // checker_size
                if (y & 1) ^ (x & 1) ^ (z & 1):
                    color_difference = 50 / checker_size
                    color = (0, 0, (color_difference * (20 + i + j + k)) % 256)
                    array[
                        k : checker_size + k, i : checker_size + i, j : checker_size + j
                    ] = color
    print("finish creating test volume")
    return array


def centered_linear(A, dims):
    # """3×4 affine around volume center (z, y, x)."""
    # offsets = dims[1:] / 2
    # c = ht.tile(offsets, (dims[0],1))
    # b = c - A @ c
    # return ht.hstack([A, b[:, None]]).astype(np.float32)
    return A


# def show(title, volume, slice_point):
#     volume_slice = volume[slice_point, :, :]
#     img = volume_slice.numpy().astype(np.uint8)

#     # if img.ndim == 3:
#     #     img = img[0]
#     plt.imshow(img)
#     plt.title(title)
#     plt.axis("off")


# ============================================================
# Create Image
# ============================================================
# Heat array (NO split)
DEPTH = 32
WIDTH = 255
HEIGHT = 128

SLICE_AXIS = 2

vol = ht.stack(
    (create_checker_volume(32, 255, 128, 16), create_checker_volume(32, 255, 128, 8))
)
print("finished generating image")

dims = ht.array(vol.shape)
dims[0] // 2

fig, axs = plt.subplots(6, 2, figsize=(10, 16))
axs = axs.ravel()


def apply(M: ht.DNDarray, title, row_idx):
    mode = "constant"
    constant_value = 0.0

    idx = row_idx * 2

    result = affine_transform(
        vol,
        M,
        order=1,
        mode=mode,
        cval=constant_value,
        prefilter=False,
    )
    print("SHAPE COMPARISON")
    print(f"{vol.shape}")
    print(f"{vol.numpy().shape}")

    # compare = ndimg.affine_transform(
    #     vol.numpy(), M.numpy(), order=1, mode=mode, cval=constant_value, prefilter=True
    # )

    if vol.ndim == 5:
        match SLICE_AXIS:
            case 0:
                slice1 = result[0, dims[SLICE_AXIS] // 2, :, :]
                slice2 = result[1, dims[SLICE_AXIS] // 2, :, :]
            case 1:
                slice1 = result[0, :, dims[SLICE_AXIS] // 2, :]
                slice2 = result[1, :, dims[SLICE_AXIS] // 2, :]
            case 2:
                slice1 = result[0, :, :, dims[SLICE_AXIS] // 2]
                slice2 = result[1, :, :, dims[SLICE_AXIS] // 2]
    else:
        slice1 = result[0]
        slice2 = result[1]

    result_slice = slice1.numpy().astype(np.uint8)
    compare_slice = slice2.numpy().astype(np.uint8)

    slice_dims = result_slice.shape

    print(f"resulting shape: {result.shape}")
    axs[idx].imshow(result_slice)
    axs[idx + 1].imshow(compare_slice)
    axs[idx].set_title(title)
    axs[idx + 1].set_title("")
    axs[idx].scatter(slice_dims[1] / 2, slice_dims[0] / 2)
    axs[idx + 1].scatter(slice_dims[1] / 2, slice_dims[0] / 2)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------
apply(
    ht.stack(
        (
            ht.eye(
                (
                    4,
                    5,
                ),
                dtype=ht.float32,
            ),
            ht.eye(
                (
                    4,
                    5,
                ),
                dtype=ht.float32,
            ),
        )
    ),
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
)
print(f"shape after creation: {A_rot.shape}")
M_rot = centered_linear(A_rot, dims)
apply(M_rot, "20 degrees", 1)
# ------------------------------------------------------------
# Scale ×1.2
# ------------------------------------------------------------
A_scale = ht.array(
    [
        [[0.8, 0, 0, 0], [0, 1.2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]],
        [[1.2, 0, 0, 0], [0, 2.2, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 1]],
    ],
    dtype=ht.float32,
)
print(f"shape after creation: {A_scale.shape}")
M_scale = centered_linear(A_scale, dims)
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
