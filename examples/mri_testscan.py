"""
MRI affine transformation demo using Heat.

This example loads a sample MRI volume, extracts a middle slice,
applies various affine transformations (rotation, scaling,
translation, shear), and visualizes the results.

MRI sample data © 2018 Adam Wolf, used under the MIT License.
See heat/datasets/mri_sample_LICENSE.txt for details.
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import heat as ht
from heat.ndimage.affine import affine_transform


# ============================================================
# Load MRI volume
# ============================================================

def load_mri(path):
    """
    Load an MRI volume from a NIfTI file.

    Parameters
    ----------
    path : str
        Path to the .nii or .nii.gz MRI file.

    Returns
    -------
    numpy.ndarray
        MRI volume as a float32 array of shape (D, H, W),
        normalized to the range [0, 255].
    """
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    vol = vol / np.max(vol) * 255
    return vol


# ============================================================
# Slice helper
# ============================================================

def middle_slice(volume):
    """
    Extract the middle axial slice from a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray
        3D array of shape (D, H, W).

    Returns
    -------
    numpy.ndarray
        2D slice of shape (H, W).
    """
    mid = volume.shape[0] // 2
    return volume[mid, :, :]


# ============================================================
# Convert numpy MRI slice → Heat array
# ============================================================

def to_heat_slice(slice2d):
    """
    Convert a 2D NumPy array into a Heat array.

    Parameters
    ----------
    slice2d : numpy.ndarray
        2D image slice.

    Returns
    -------
    ht.DNDarray
        Heat array representation of the slice.
    """
    return ht.array(slice2d, dtype=ht.float32)


# ============================================================
# Plot grid
# ============================================================

def show_results(titles, images, save_path=None):
    """
    Display a grid of images with titles.

    Parameters
    ----------
    titles : list of str
        Titles for each subplot.
    images : list of numpy.ndarray
        Images to display.
    save_path : str, optional
        If given, the figure is saved to this path.
    """
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
    """
    Run the MRI affine transformation demo.

    This function:
    1. Loads a sample MRI volume
    2. Extracts the middle slice
    3. Applies multiple affine transformations using Heat
    4. Visualizes and saves the results
    """
    # -------- LOAD MRI --------
    vol = load_mri(
        "/Users/marka.k/1900_Image_transformations/heat/heat/datasets/flair.nii.gz"
    )
    print("Loaded MRI:", vol.shape)

    orig = middle_slice(vol)
    x = to_heat_slice(orig)

    H, W = orig.shape

    # ========================================================
    # Define transforms (center-aware)
    # ========================================================

    def recenter(M):
        """
        Recenter a 2D affine matrix around the image center.

        Parameters
        ----------
        M : numpy.ndarray
            Affine matrix of shape (2, 3).

        Returns
        -------
        numpy.ndarray
            Recentered affine matrix of shape (2, 3).
        """
        cx, cy = W / 2, H / 2

        M3 = np.array(
            [
                [M[0, 0], M[0, 1], M[0, 2]],
                [M[1, 0], M[1, 1], M[1, 2]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        T1 = np.array(
            [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]],
            dtype=np.float32,
        )

        T2 = np.array(
            [[1, 0, cx], [0, 1, cy], [0, 0, 1]],
            dtype=np.float32,
        )

        M_centered = T2 @ M3 @ T1

        return np.array(
            [
                [M_centered[0, 0], M_centered[0, 1], M_centered[0, 2]],
                [M_centered[1, 0], M_centered[1, 1], M_centered[1, 2]],
            ],
            dtype=np.float32,
        )

    # ROTATION
    angle = np.radians(20)
    M_rot = recenter(
        np.array(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle),  np.cos(angle), 0]],
            dtype=np.float32,
        )
    )

    # SCALE
    s = 1.2
    M_scale = recenter(
        np.array([[s, 0, 0], [0, s, 0]], dtype=np.float32)
    )

    # TRANSLATION
    M_trans = np.array([[1, 0, 20], [0, 1, -20]], dtype=np.float32)

    # SHEAR
    sh = 0.3
    M_shear = recenter(
        np.array([[1, sh, 0], [0, 1, 0]], dtype=np.float32)
    )

    # ROTATION (Z-axis equivalent)
    angle = np.radians(35)
    M_rotZ = recenter(
        np.array(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle),  np.cos(angle), 0]],
            dtype=np.float32,
        )
    )

    # ========================================================
    # Run transformations
    # ========================================================

    out_rot = affine_transform(x, M_rot, order=1).numpy()
    out_scale = affine_transform(x, M_scale, order=1).numpy()
    out_trans = affine_transform(x, M_trans, order=1).numpy()
    out_shear = affine_transform(x, M_shear, order=1).numpy()
    out_rotZ = affine_transform(x, M_rotZ, order=1).numpy()

    # ========================================================
    # Show + save
    # ========================================================

    titles = [
        "Original (middle slice)",
        "Rotate 20°",
        "Scale ×1.2",
        "Translate (+20, −20)",
        "Shear (0.3)",
        "Rotation (Z-axis 35°)",
    ]

    imgs = [orig, out_rot, out_scale, out_trans, out_shear, out_rotZ]

    save_path = "mri_affine_demo.png"
    show_results(titles, imgs, save_path=save_path)

    print(f"\nSaved result figure → {save_path}")


if __name__ == "__main__":
    main()
