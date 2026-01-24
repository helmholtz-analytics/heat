"""
Distributed affine demo — halo exchange handled INSIDE affine.py.

- Split along Z (axis 0)
- Translation along Z
- NO explicit halo exchange here
- NO global indexing
- NO cropping
- Each rank visualizes its OWN strongest local slice
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

    if rank == 0:
        nii = nib.load(
            "PATH"
        )
        x_np = nii.get_fdata().astype(np.float32)
        print("Loaded MRI:", x_np.shape)
    else:
        x_np = None

    x_np = comm.bcast(x_np, root=0)


    x = ht.array(x_np, split=0)
    print(f"[rank {rank}] local input shape = {tuple(x.larray.shape)}")
    """
    s = 1
    ND = 3
    M = np.eye(ND, ND + 1)
    M[0, 0] = s   # scale Z
    """
    
    shift = 2          # integer shift along Z
    ND = 3

    M = np.eye(ND, ND + 1)
    M[0, -1] = shift   # translate along Z
    

    y = affine_transform(x, M)
    y_local = y.larray
    print(
    f"[rank {rank}] local nonzero slices:",
    (y_local.abs().amax(dim=(1,2)) > 1e-3).nonzero().flatten().tolist()
)

    if y_local.shape[0] == 0:
        print(f"")
    else:
        scores = y_local.abs().amax(dim=(1, 2))
        best_score, best_idx = scores.max(dim=0)

        if best_score.item() < 1e-3:
            print(f"")
        else:
            idx = best_idx.item()
            plt.figure(figsize=(5, 5))
            plt.imshow(y_local[idx].cpu().numpy())
            plt.title(f"rank {rank}, strongest local slice {idx}")
            plt.axis("off")
            plt.show()

    comm.Barrier()

    if rank == 0:
        print("\nDONE — distributed affine demo completed cleanly\n")


if __name__ == "__main__":
    main()
