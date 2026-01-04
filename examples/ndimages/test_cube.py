"""
Synthetic 3D star / landmark cube test for affine_transform.

Works for:
- batched outputs (1, D, H, W)
- unbatched inputs (D, H, W)
- MPI (mpirun -np 2)
"""

import numpy as np
import heat as ht
import matplotlib.pyplot as plt
from heat.ndimage.affine import affine_transform


# ============================================================
# 1. Create a synthetic 3D star cube
# ============================================================

def make_star_cube(size=64, arm_length=20, arm_thickness=2):
    cube = np.zeros((size, size, size), dtype=np.float32)
    c = size // 2

    # Main axes
    cube[c-arm_length:c+arm_length, c, c] = 5
    cube[c, c-arm_length:c+arm_length, c] = 5
    cube[c, c, c-arm_length:c+arm_length] = 5

    # Diagonals
    for i in range(-arm_length, arm_length):
        cube[c+i, c+i, c] = 4
        cube[c+i, c-i, c] = 4

    # Landmarks
    cube[c, c, c] = 10
    cube[c, c, c+10] = 8
    cube[c, c+10, c] = 8
    cube[c+10, c, c] = 8

    return ht.array(cube)


# ============================================================
# 2. Build test volume
# ============================================================

x = make_star_cube()
D, H, W = x.shape
c = D // 2

print("Cube shape:", x.shape)


# ============================================================
# 3. Affine matrices
# ============================================================

M_identity = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
]

M_translate = [
    [1, 0, 0, 8],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
]

theta = np.deg2rad(30)
ct, st = np.cos(theta), np.sin(theta)
M_rotate = [
    [ ct, -st, 0, 0],
    [ st,  ct, 0, 0],
    [  0,   0, 1, 0],
]


# ============================================================
# 4. Apply affine transforms
# ============================================================

print("Applying affine transforms...")

y_id  = affine_transform(x, M_identity,  order=1)
y_tr  = affine_transform(x, M_translate, order=0)
y_rot = affine_transform(x, M_rotate,    order=1)


# ============================================================
# 5. Numeric checks
# ============================================================

print("\nNumeric checks:")

assert ht.allclose(x, y_id)
print("✓ Identity transform OK")

assert y_tr.numpy()[0, c, c, c+18] == 8
print("✓ Translation moves landmark correctly")


# ============================================================
# 6. Visualization (robust)
# ============================================================

def to_volume(arr):
    """
    Convert numpy array to (D, H, W) no matter what.
    """
    if arr.ndim == 4:     # (1, D, H, W)
        return arr[0]
    if arr.ndim == 3:     # (D, H, W)
        return arr
    raise ValueError(f"Unexpected shape {arr.shape}")


def show_slice(arr, title):
    vol = to_volume(arr)
    z = vol.shape[0] // 2
    plt.imshow(vol[z], cmap="gray")
    plt.title(title)
    plt.axis("off")


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
show_slice(x.numpy(), "Original")

plt.subplot(1, 3, 2)
show_slice(y_tr.numpy(), "Translate +X")

plt.subplot(1, 3, 3)
show_slice(y_rot.numpy(), "Rotate 30°")

plt.tight_layout()
plt.show()

print("\nAll tests completed successfully.")
