"""
Internal utilities for bridging Heat DNDarrays and PyTorch DTensor.
"""

import os
import torch
import warnings
from typing import Optional, Sequence, Tuple, Union, Callable

import torch.distributed as dist

from .dndarray import DNDarray

try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
    from torch.distributed.tensor import DTensor, Shard, Replicate
    from torch.distributed.tensor.placement_types import Placement, Partial

    DTENSOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    DTENSOR_AVAILABLE = False
    DTensor = None
    DeviceMesh = None
    Shard = Replicate = Partial = None

_DEVICE_MESHES = {}


def get_or_create_mesh(device, comm) -> Optional["DeviceMesh"]:
    """
    Retrieves or initializes a 1D DeviceMesh matching the Heat communicator
    and target compute device (e.g. NCCL for GPUs).
    """
    if not DTENSOR_AVAILABLE:
        return None

    mesh_type = "cuda" if str(device)[:3] == "gpu" else "cpu"
    comm_id = comm.handle.py2f() if hasattr(comm.handle, "py2f") else id(comm.handle)
    cache_key = (mesh_type, comm_id)

    if cache_key in _DEVICE_MESHES:
        return _DEVICE_MESHES[cache_key]

    if not dist.is_initialized():
        # Set up torch.distributed env vars from MPI world state
        os.environ.setdefault("RANK", str(comm.rank))
        os.environ.setdefault("WORLD_SIZE", str(comm.size))
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

        if mesh_type == "cuda" and torch.cuda.is_available():
            local_rank = int(
                os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", comm.rank % torch.cuda.device_count())
            )
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "mpi" if dist.is_mpi_available() else "gloo"

        dist.init_process_group(backend=backend)

    mesh = init_device_mesh(mesh_type, (comm.size,))
    _DEVICE_MESHES[cache_key] = mesh
    return mesh


def is_dtensor_eligible(*arrays) -> bool:
    """
    Checks if all input arrays can be safely represented as DTensors.
    Requires GPU execution, identical communicators, and evenly split dimensions.
    """
    if not DTENSOR_AVAILABLE or not arrays:
        return False

    first_comm = arrays[0].comm
    for a in arrays:
        # DTensor execution is primarily beneficial for GPU/CUDA backends
        if str(a.device)[:3] != "gpu":
            return False
        # Must share communicator
        if a.comm != first_comm or not a.comm.is_distributed():
            return False
        # DTensor 1D Shard requires clean divisibility (no remainder chunks)
        if a.split is not None and (a.gshape[a.split] % a.comm.size != 0):
            return False

    return True


def dndarray_to_dtensor(dndarray: "DNDarray", mesh: "DeviceMesh") -> "DTensor":
    """
    Wraps the local torch.Tensor of a DNDarray into a DTensor.
    """
    placements = [Replicate()] if dndarray.split is None else [Shard(dndarray.split)]
    return DTensor.from_local(dndarray.larray, mesh, placements)


def dtensor_to_dndarray(
    dtensor: "DTensor",
    target_split: Optional[int],
    reference_array: "DNDarray",
    out_shape: Tuple[int, ...],
) -> "DNDarray":
    """
    Redistributes a DTensor to target placement and wraps it back into a DNDarray.
    """
    from .dndarray import DNDarray

    target_placement = Replicate() if target_split is None else Shard(target_split)

    # Force network resolution if result contains unreduced Partial sums or placement mismatch
    if (
        any(isinstance(p, Partial) for p in dtensor.placements)
        or dtensor.placements[0] != target_placement
    ):
        dtensor = dtensor.redistribute(dtensor.device_mesh, [target_placement])

    local_tensor = dtensor.to_local()
    return DNDarray(
        local_tensor,
        out_shape,
        reference_array.dtype,
        split=target_split,
        device=reference_array.device,
        comm=reference_array.comm,
        balanced=True,
    )


def try_dtensor_op(
    torch_op: Callable,
    *args: "DNDarray",
    target_split: Optional[int],
    out_shape: Optional[Tuple[int, ...]] = None,
    **kwargs,
) -> Optional["DNDarray"]:
    if not is_dtensor_eligible(*args):
        return None

    # ensure intermediate allocation fits in GPU VRAM
    if str(args[0].device)[:3] == "gpu" and torch.cuda.is_available():
        device_idx = args[0].larray.device
        free_mem, _ = torch.cuda.mem_get_info(device_idx)

        # Estimate memory for matmul (A @ B)
        if torch_op is torch.matmul and len(args) == 2:
            m = args[0].gshape[-2]
            n = args[1].gshape[-1]
            elem_bytes = args[0].dtype.torch_type().itemsize

            # If contracting axis is split, DTensor allocates full (m, n) for Partial(SUM)
            inner_split = (args[0].split == args[0].ndim - 1) or (args[1].split == args[1].ndim - 2)
            needed_bytes = (
                (m * n * elem_bytes) if inner_split else ((m * n * elem_bytes) // args[0].comm.size)
            )

            # Leave at least 20% headroom
            if needed_bytes > 0.8 * free_mem:
                return None  # Cleanly bypass DTensor and use MPI without raising OOM

    try:
        mesh = get_or_create_mesh(args[0].device, args[0].comm)
        if mesh is None:
            return None

        dt_args = [dndarray_to_dtensor(a, mesh) for a in args]
        dt_res = torch_op(*dt_args, **kwargs)

        final_shape = out_shape if out_shape is not None else tuple(dt_res.shape)
        return dtensor_to_dndarray(dt_res, target_split, args[0], final_shape)
    except Exception as e:
        warnings.warn(f"DTensor execution failed, falling back to MPI routine: {e}")
        return None
