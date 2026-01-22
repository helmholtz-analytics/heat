"""
Distributed affine demo USING HALOS CORRECTLY.

- Split along Z (axis 0)
- Translation along Z
- Explicit halo exchange via array_with_halos (PROPERTY)
- Each rank visualizes its OWN correct output
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import heat as ht
from mpi4py import MPI

from heat.ndimage.affine import affine_transform


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ============================================================
    # Load MRI on rank 0
    # ============================================================
    if rank == 0:
        nii = nib.load(
            "/Users/marka.k/1900_Image_transformations/heat/heat/datasets/flair.nii.gz"
        )
        x_np = nii.get_fdata().astype(np.float32)
        print("Loaded MRI:", x_np.shape)
    else:
        x_np = None

    x_np = comm.bcast(x_np, root=0)

    # ============================================================
    # Distributed array (split Z)
    # ============================================================
    x = ht.array(x_np, split=0)
    local = x.larray
    local_z = local.shape[0]

    print(f"[rank {rank}] local shape = {tuple(local.shape)}")

    # ============================================================
    # Define Z translation
    # ============================================================
    shift = 2   # +2 slices along Z
    ND = 3
    M = np.eye(ND, ND + 1)
    M[0, -1] = shift

    halo = abs(shift)

    # ============================================================
    # HALO EXCHANGE (CORRECT API)
    # ============================================================
    x.get_halo(halo)               # side effect
    x_halo = x.array_with_halos    # <-- PROPERTY, not callable

    print(
        f"[rank {rank}] halo tensor shape = {tuple(x_halo.shape)} "
        f"(local={local_z}, halo={halo})"
    )

    # ============================================================
    # Apply affine on HALO tensor (LOCAL op)
    # ============================================================
    y_halo = affine_transform(
        ht.array(x_halo, split=None),
        M,
    )

    y_halo_local = y_halo.larray

    # ============================================================
    # Crop halo → get correct local result
    # ============================================================
    y_local = y_halo_local[halo : halo + local_z]

    print(f"[rank {rank}] output local shape = {tuple(y_local.shape)}")

    # ============================================================
    # Visualization: show first NON-EMPTY slice
    # ============================================================
    slice_idx = None
    for i in range(y_local.shape[0]):
        if y_local[i].abs().max().item() > 1e-3:
            slice_idx = i
            break

    if slice_idx is None:
        print(f"[rank {rank}] all local slices are empty")
    else:
        plt.figure(figsize=(5, 5))
        plt.imshow(y_local[slice_idx].cpu().numpy(), cmap="gray")
        plt.title(f"rank {rank}, local slice {slice_idx}")
        plt.axis("off")
        plt.show()

    comm.Barrier()

    if rank == 0:
        print("\nDONE — halo-based affine works correctly\n")


if __name__ == "__main__":
    main()
