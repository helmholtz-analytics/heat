import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import heat as ht
from mpi4py import MPI

from heat.ndimage.affine import affine_transform


# ============================================================
# Helpers
# ============================================================

def canonicalize_to_ZHW(t):
    """
    Force tensor to shape (Z, H, W)
    """
    while t.ndim > 3:
        t = t.squeeze(0)

    if t.ndim == 2:
        t = t.unsqueeze(0)

    if t.ndim != 3:
        raise RuntimeError(f"Unexpected tensor shape: {t.shape}")

    return t


def strongest_slice(vol):
    """
    vol: torch.Tensor (Z, H, W)
    """
    scores = vol.abs().amax(dim=(1, 2))
    score, idx = scores.max(dim=0)
    return int(idx.item()), float(score.item())


def apply_affine(x, M):
    y = affine_transform(x, M, order=0, mode="nearest")
    y_local = canonicalize_to_ZHW(y.larray)
    return y_local


# ============================================================
# MAIN
# ============================================================

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --------------------------------------------------------
    # Load MRI on rank 0
    # --------------------------------------------------------
    if rank == 0:
        nii = nib.load(
            "PATH"
        )
        vol = nii.get_fdata().astype(np.float32)
        print(f"[rank 0] Loaded MRI: {vol.shape}", flush=True)
    else:
        vol = None

    vol = comm.bcast(vol, root=0)

    # --------------------------------------------------------
    # Heat array (distributed over Z)
    # --------------------------------------------------------
    x = ht.array(vol, split=0)
    print(f"[rank {rank}] local input shape = {x.larray.shape}", flush=True)

    comm.Barrier()

    # ========================================================
    # Define affine transforms (SPACE-based)
    # ========================================================
    M_identity = np.eye(3, 4, dtype=np.float32)

    M_scale = np.eye(3, 4, dtype=np.float32)
    M_scale[0, 0] = 1.2
    M_scale[1, 1] = 1.2
    M_scale[2, 2] = 1.2

    M_translate = np.eye(3, 4, dtype=np.float32)
    M_translate[0, 3] = 10.0   # +10 in Z (SPACE)

    cases = [
        ("Identity", M_identity),
        ("Scale ×1.2", M_scale),
        ("Translate +10 Z", M_translate),
    ]

    # ========================================================
    # Apply all transforms
    # ========================================================
    results = []

    for name, M in cases:
        y_local = apply_affine(x, M)
        idx, score = strongest_slice(y_local)

        print(
            f"[rank {rank}] {name}: strongest slice idx={idx}, score={score:.3e}",
            flush=True,
        )

        results.append((name, y_local, idx, score))

    # ========================================================
    # Visualization — ONE WINDOW PER RANK
    # ========================================================
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Rank {rank} — SPACE affine operations", fontsize=14)

    for ax, (name, vol_local, idx, score) in zip(axs, results):
        if score < 1e-3:
            ax.set_title(f"{name}\nEMPTY")
            ax.axis("off")
            continue

        ax.imshow(vol_local[idx].cpu().numpy(), cmap="gray")
        ax.set_title(f"{name}\nslice {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    comm.Barrier()

    if rank == 0:
        print("\nDONE — multi-operation SPACE-affine demo completed\n", flush=True)


if __name__ == "__main__":
    main()
