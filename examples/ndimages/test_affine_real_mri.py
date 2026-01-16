import nibabel as nib
import numpy as np
import heat as ht
import torch
from mpi4py import MPI

from heat.ndimage.affine import distributed_affine_transform


def chunk_bounds_1d(n, rank, size):
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    return start, stop


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --------------------------------------------------
    # Load MRI (rank 0 only)
    # --------------------------------------------------
    if rank == 0:
        path = "/Users/marka.k/1900_Image_transformations/heat/heat/datasets/flair.nii.gz"
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        print("Loaded MRI shape:", data.shape)
    else:
        data = None

    # Broadcast for test setup only
    data = comm.bcast(data, root=0)

    # --------------------------------------------------
    # Create distributed Heat array
    # --------------------------------------------------
    x = ht.array(data, split=0)

    D, H, W = x.gshape
    z0, z1 = chunk_bounds_1d(D, rank, size)

    local = x.larray

    # --------------------------------------------------
    # HARD PROOF OF SPLITTING
    # --------------------------------------------------
    nonzero_z = torch.nonzero(local.sum(dim=(1, 2))).flatten()

    print(
        f"\nRank {rank}\n"
        f"  owns global z range   : [{z0}, {z1})\n"
        f"  local tensor shape    : {tuple(local.shape)}\n"
        f"  local sum             : {float(local.sum()):.2f}\n"
        f"  first nonzero local z : {int(nonzero_z[0]) if len(nonzero_z) else 'NONE'}\n"
        f"  last  nonzero local z : {int(nonzero_z[-1]) if len(nonzero_z) else 'NONE'}\n"
    )

    comm.Barrier()

    # --------------------------------------------------
    # Apply distributed affine (small translation)
    # --------------------------------------------------
    M = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 5],
        ],
        dtype=torch.float32,
    )

    y = distributed_affine_transform(x, M, order=0, mode="constant", cval=0.0)

    comm.Barrier()

    local_out = y.larray
    nonzero_out_z = torch.nonzero(local_out.sum(dim=(1, 2))).flatten()

    print(
        f"Rank {rank} AFTER affine\n"
        f"  output local shape    : {tuple(local_out.shape)}\n"
        f"  output local sum      : {float(local_out.sum()):.2f}\n"
        f"  output first z        : {int(nonzero_out_z[0]) if len(nonzero_out_z) else 'NONE'}\n"
        f"  output last  z        : {int(nonzero_out_z[-1]) if len(nonzero_out_z) else 'NONE'}\n"
    )

    comm.Barrier()

    if rank == 0:
        print("\n✅ SPLIT VERIFICATION COMPLETE")


if __name__ == "__main__":
    main()
