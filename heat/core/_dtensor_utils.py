"""
Internal utilities for bridging Heat DNDarrays and PyTorch DTensor.
"""

import os
import torch
import warnings
from typing import Optional, Sequence, Tuple, Union, Callable

from .communication import MPI, MPI_WORLD
from .dndarray import DNDarray
import torch.distributed as dist
import atexit
import contextlib


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


def _cleanup_dist(dist_module=dist):
    """
    Cleans up the PyTorch distributed process group on interpreter exit
    to avoid NCCL resource leak warnings.
    """
    try:
        if dist_module.is_available() and dist_module.is_initialized():
            dist_module.destroy_process_group()
    except Exception:
        pass


# Register the cleanup hook
atexit.register(_cleanup_dist)


def destroy_dtensor_mesh():
    """
    Destroys the NCCL process group, clears cached device meshes,
    and returns all reserved GPU memory back to the driver.
    """
    global _DEVICE_MESHES

    # 1. Clear cached mesh objects
    _DEVICE_MESHES.clear()

    # 2. Destroy PyTorch distributed process group
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # 3. Synchronize and return memory to CUDA driver
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@contextlib.contextmanager
def dtensor_context():
    """
    Context manager that guarantees all NCCL rings, process groups,
    and cached memory are destroyed upon exiting the block.
    """
    try:
        yield
    finally:
        destroy_dtensor_mesh()


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
    Requires GPU execution, global MPI_WORLD communicator, balanced chunks,
    and evenly split dimensions.
    """
    if not DTENSOR_AVAILABLE or not arrays:
        return False

    first_comm = arrays[0].comm
    for a in arrays:
        # DTensor execution is beneficial for GPU/CUDA backends
        if str(a.device)[:3] != "gpu":
            return False

        # Guard: Only allow the global world communicator.
        # Sub-communicators created by algorithms (e.g. HSVD, TS-QR) must use native MPI.
        if a.comm != MPI_WORLD or a.comm != first_comm or not a.comm.is_distributed():
            return False

        # Guard: DTensor 1D Shard requires balanced allocations across all ranks
        if not a.is_balanced():
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
    empty_cache: bool = False,
    **kwargs,
) -> Optional["DNDarray"]:
    if not is_dtensor_eligible(*args):
        return None

    comm = args[0].comm

    # Collective memory check
    if str(args[0].device)[:3] == "gpu" and torch.cuda.is_available():
        device_idx = args[0].larray.device
        local_free_mem, _ = torch.cuda.mem_get_info(device_idx)
        comm_handle = comm.handle if hasattr(comm, "handle") else comm
        min_free_mem = comm_handle.allreduce(local_free_mem, op=MPI.MIN)

        if torch_op is torch.matmul and len(args) == 2:
            m = args[0].gshape[-2]
            n = args[1].gshape[-1]
            elem_bytes = args[0].dtype.torch_type().itemsize
            inner_split = (args[0].split == args[0].ndim - 1) or (args[1].split == args[1].ndim - 2)
            needed_bytes = (
                (m * n * elem_bytes) if inner_split else ((m * n * elem_bytes) // comm.size)
            )
            if needed_bytes > 0.8 * min_free_mem:
                return None

    try:
        mesh = get_or_create_mesh(args[0].device, comm)
        if mesh is None:
            return None

        dt_args = [dndarray_to_dtensor(a, mesh) for a in args]
        dt_res = torch_op(*dt_args, **kwargs)

        final_shape = out_shape if out_shape is not None else tuple(dt_res.shape)
        result = dtensor_to_dndarray(dt_res, target_split, args[0], final_shape)

        # Free python references to the DTensor graph immediately
        del dt_args, dt_res

        # Release cached memory blocks back to driver if requested
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    except Exception as e:
        if comm.rank == 0:
            warnings.warn(f"DTensor execution failed, falling back to MPI routine: {e}")
        return None
