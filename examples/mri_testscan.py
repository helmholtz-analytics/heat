import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import heat as ht
from heat.ndimage.affine import affine_transform
"""
MRI sample data © 2018 Adam Wolf, used under MIT License.
See heat/datasets/mri_sample_LICENSE.txt for details.
"""


# ============================================================
# Load MRI volume
# ============================================================
def load_mri(path):
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)  # (D,H,W)
    vol = vol / np.max(vol) * 255
    return vol


# ============================================================
# Slice helper
# ============================================================
def middle_slice(volume):
    mid = volume.shape[0] // 2
    return volume[mid, :, :]


# ============================================================
# Convert numpy MRI slice → Heat array
# ============================================================
def to_heat_slice(slice2d):
    return ht.array(slice2d, dtype=ht.float32)


# ============================================================
# Plot grid
# ============================================================
def show_results(titles, images, save_path=None):
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(16, 10))
    axs = axs.flatten()

    for ax, title, img in zip(axs, titles, images):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    for ax in axs[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    # -------- LOAD MRI --------
    vol = load_mri("/Users/marka.k/1900_Image_transformations/heat/heat/datasets/flair.nii.gz")     
    print("Loaded MRI:", vol.shape)

    orig = middle_slice(vol)             # (H,W)
    x = to_heat_slice(orig)

    H, W = orig.shape

    # ========================================================
    # Define transforms (all center-aware)
    # ========================================================

    cx, cy = W/2, H/2

    # Helper to shift center for rotation/scale
    def recenter(M):
            """
            Input: 2x3 affine matrix.
            Output: 2x3 affine matrix recentered around the image center.
            """

            cx, cy = W/2, H/2

            # Convert 2×3 → 3×3 homogeneous
            M3 = np.array([
                [M[0,0], M[0,1], M[0,2]],
                [M[1,0], M[1,1], M[1,2]],
                [0,      0,      1     ]
            ], dtype=np.float32)

            # Center shift matrices
            T1 = np.array([
                [1, 0, -cx],
                [0, 1, -cy],
                [0, 0,  1 ]
            ], np.float32)

            T2 = np.array([
                [1, 0, cx],
                [0, 1, cy],
                [0, 0, 1 ]
            ], np.float32)

            # Recenter: T2 * M * T1
            M_centered = T2 @ M3 @ T1

            # Return as 2×3
            return np.array([
                [M_centered[0,0], M_centered[0,1], M_centered[0,2]],
                [M_centered[1,0], M_centered[1,1], M_centered[1,2]],
            ], dtype=np.float32)

    # ROTATION
    angle = np.radians(20)
    M_rot = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0]
    ], np.float32)
    M_rot = recenter(M_rot)

    # SCALE
    s = 1.2
    M_scale = np.array([
        [s, 0, 0],
        [0, s, 0]
    ], np.float32)
    M_scale = recenter(M_scale)

    # TRANSLATE
    M_trans = np.array([
        [1, 0, 20],
        [0, 1, -20]
    ], np.float32)

    # SHEAR
    sh = 0.3
    M_shear = np.array([
        [1, sh, 0],
        [0, 1, 0]
    ], np.float32)
    M_shear = recenter(M_shear)

    # 3D ROTATION ABOUT Z AXIS APPLIED TO 2D SLICE (equivalent)
    angle = np.radians(35)
    M_rotZ = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0]
    ], np.float32)
    M_rotZ = recenter(M_rotZ)

    # ========================================================
    # Run transformations
    # ========================================================

    out_rot    = affine_transform(x, M_rot,    order=1).numpy()
    out_scale  = affine_transform(x, M_scale,  order=1).numpy()
    out_trans  = affine_transform(x, M_trans,  order=1).numpy()
    out_shear  = affine_transform(x, M_shear,  order=1).numpy()
    out_rotZ   = affine_transform(x, M_rotZ,   order=1).numpy()

    # ========================================================
    # Show + save
    # ========================================================

    titles = [
        "Original (middle slice)",
        "Rotate 20°",
        "Scale ×1.2",
        "Translate (+20, -20)",
        "Shear (0.3)",
        "3D Rotation (Z-axis 35°)",
    ]

    imgs = [orig, out_rot, out_scale, out_trans, out_shear, out_rotZ]

    save_path = "mri_affine_demo.png"

    show_results(titles, imgs, save_path=save_path)

    print(f"\nSaved result figure → {save_path}")


if __name__ == "__main__":
    main()
