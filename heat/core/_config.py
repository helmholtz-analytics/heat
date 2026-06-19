"""
Everything you need to know about the configuration of Heat
"""

from mpi4py import MPI
from numpy import isin
import torch
import platform
import mpi4py
import subprocess
import os
import warnings
import re
import dataclasses
from enum import Enum


class MPILibrary(Enum):
    OpenMPI = "ompi"
    IntelMPI = "impi"
    MVAPICH = "mvapich"
    MPICH = "mpich"
    CrayMPI = "craympi"
    ParaStationMPI = "psmpi"
    Other = "other"


@dataclasses.dataclass
class MPILibraryInfo:
    name: MPILibrary
    version: str
    cuda_compatible: bool = False
    rocm_compatible: bool = False
    gpu_compatible: bool = False
    incompatible_operations: list[str] | None = None


# Helper function to match version patterns
def _match_version_pattern(
    version: str, patterns: dict[str, dict[str, list[str] | None]]
) -> dict[str, list[str] | None]:
    """
    Match a version string against pattern keys (e.g., '5.0.x', '4.1.x', '*').
    Returns the incompatibilities dict for the matching pattern, or {} if no match.

    Parameters
    ----------
    version : str
        The version string to match (e.g., 'v5.0.1', '4.1.2')
    patterns : dict[str, dict[str, list[str] | None]]
        Dictionary mapping version patterns to incompatibilities

    Returns
    -------
    dict[str, list[str] | None]
        The incompatibilities for the matched version pattern, or {} if no match
    """
    # First check for wildcard pattern
    if "*" in patterns:
        return patterns["*"]

    # Then try to match specific version patterns
    for pattern, incompatibilities in patterns.items():
        # Convert pattern like '5.0.x' to regex '5\.0\.\d+'
        regex_pattern = pattern.replace(".", r"\.").replace("x", r"\d+")
        if re.match(f"^{regex_pattern}$", version):
            return incompatibilities

    return {}


def _get_mpi_library() -> MPILibraryInfo:
    library_info = mpi4py.MPI.Get_library_version().split()
    incompatibilities_list_id = "rocm" if CUDA_IS_ACTUALLY_ROCM else "cuda"

    match library_info:
        case ["Open", "MPI", *_]:
            library = MPILibrary.OpenMPI
            version = library_info[2]

            try:
                parsable_ompi_info = subprocess.check_output(
                    ["ompi_info", "--parsable", "--all"]
                ).decode("utf-8")
                ompi_info = subprocess.check_output(["ompi_info"]).decode("utf-8")

                # Check for CUDA support flag
                cuda_support_flag = "mpi_built_with_cuda_support:value:true" in parsable_ompi_info

                # Check for extensions
                match = re.search(r"MPI extensions: (.*)", ompi_info)
                extensions = [ext.strip() for ext in match.group(0).split(":")[1].split(",")]
                cuda = cuda_support_flag and "cuda" in extensions
                if version.startswith("v4."):
                    rocm = cuda
                elif version.startswith("v5."):
                    rocm = "rocm" in extensions or "hip" in extensions

            finally:
                cuda = False
                rocm = False
                gpu_comp = False
                device_incompatibilities = None

        case ["Intel(R)", "MPI", *_]:
            library = MPILibrary.IntelMPI
            version = library_info[3]

            cuda = False
            rocm = False

        case ["MPICH", "Version:", *_]:
            library = MPILibrary.MPICH
            version = library_info[2]

            cuda = os.environ.get("MV2_USE_CUDA", "0") == "1"
            rocm = os.environ.get("MV2_USE_ROCM", "0") == "1"

        case ["MVAPICH", "Version:", *_]:
            library = MPILibrary.MVAPICH
            version = library_info[2]

            cuda = os.environ.get("MPIR_CVAR_ENABLE_HCOLL") == "1"
            rocm = False

        case ["CrayMPI", *_]:
            library = MPILibrary.CrayMPI
            version = library_info[1]
            incompatibilities = _match_version_pattern(version, INCOMPATIBILITIES.get(library, {}))

            cuda = os.environ.get("MPICH_GPU_SUPPORT_ENABLED") == "1"
            rocm = os.environ.get("MPICH_GPU_SUPPORT_ENABLED") == "1"

        case ["===", "ParaStation", "MPI", *_]:
            library = MPILibrary.ParaStationMPI
            version = library_info[3]
            cuda = os.environ.get("PSP_CUDA") == "1"
            rocm = False

        case _:
            library = MPILibrary.Other
            version = "unknown"
            cuda = False
            rocm = False

    incompatibilities = _match_version_pattern(version, INCOMPATIBILITIES.get(library, {}))
    device_incompatibilities = (
        incompatibilities[incompatibilities_list_id]
        if incompatibilities_list_id in incompatibilities
        else None
    )
    gpu_comp = (rocm and CUDA_IS_ACTUALLY_ROCM) or (cuda and not CUDA_IS_ACTUALLY_ROCM)
    gpu_comp = gpu_comp and isinstance(device_incompatibilities, list)

    return MPILibraryInfo(library, version, cuda, rocm, gpu_comp, device_incompatibilities)


# Library / version / device
# Structure: MPILibrary -> version_pattern -> device -> incompatibilities
# Incompatibilities can be:
#   - None: All operations are incompatible for this device
#   - [] (empty list): All operations are compatible for this device
#   - [list of operation names]: Only the listed operations are incompatible
INCOMPATIBILITIES: dict[MPILibrary, dict[str, dict[str, list[str] | None]]] = {
    MPILibrary.IntelMPI: {"*": {"cuda": None, "rocm": None}},
    MPILibrary.OpenMPI: {
        "5.0.x": {
            "cuda": [
                "Accumulate",
                "Compare_and_swap",
                "Fetch_and_op",
                "Get_Accumulate",
                "Iallgather",
                "Iallgatherv",
                "Iallreduce",
                "Ialltoall",
                "Ialltoallv",
                "Ialltoallw",
                "Ibcast",
                "Iscan",
                "Iexscan",
                "Rget",
                "Rput",
                "Ireduce",
            ],
            "rocm": None,
        },
        "4.1.x": {
            "cuda": [],  # All operations compatible
            "rocm": [],  # All operations compatible (ROCm handled same as CUDA in 4.1.x)
        },
    },
    MPILibrary.MVAPICH: {
        "*": {
            "cuda": [],  # All operations compatible when MV2_USE_CUDA=1
            "rocm": [],  # All operations compatible when MV2_USE_ROCM=1
        }
    },
    MPILibrary.MPICH: {
        "*": {
            "cuda": [],  # All operations compatible when MPIR_CVAR_ENABLE_HCOLL=1
            "rocm": None,  # ROCm not supported
        }
    },
    MPILibrary.CrayMPI: {
        "*": {
            "cuda": [],  # All operations compatible when MPICH_GPU_SUPPORT_ENABLED=1
            "rocm": [],  # All operations compatible when MPICH_GPU_SUPPORT_ENABLED=1
        }
    },
    MPILibrary.ParaStationMPI: {
        "*": {
            "cuda": [],  # All operations compatible when PSP_CUDA=1
            "rocm": None,  # ROCm not supported
        }
    },
    MPILibrary.Other: {
        "*": {"cuda": None, "rocm": None}
    },  # Unknown library, assume compatibility unless proven otherwise
}


PLATFORM = platform.platform()
TORCH_VERSION = torch.__version__
TORCH_CUDA_IS_AVAILABLE = torch.cuda.is_available()
CUDA_IS_ACTUALLY_ROCM = "rocm" in TORCH_VERSION

mpi_library = _get_mpi_library()
CUDA_AWARE_MPI, ROCM_AWARE_MPI = mpi_library.cuda_compatible, mpi_library.rocm_compatible
GPU_AWARE_MPI = mpi_library.gpu_compatible

# warn the user if CUDA/ROCm-aware MPI is not available, but PyTorch can use GPUs with CUDA/ROCm
if TORCH_CUDA_IS_AVAILABLE and not GPU_AWARE_MPI:
    if not CUDA_IS_ACTUALLY_ROCM and not CUDA_AWARE_MPI:
        warnings.warn(
            f"Heat has CUDA GPU-support (PyTorch version {TORCH_VERSION} and `torch.cuda.is_available() = True`), but CUDA-awareness of MPI could not be detected. This may lead to performance degradation as direct MPI-communication between GPUs is not possible.",
            UserWarning,
        )

    elif CUDA_IS_ACTUALLY_ROCM and not ROCM_AWARE_MPI:
        warnings.warn(
            f"Heat has ROCm GPU-support (PyTorch version {TORCH_VERSION} and `torch.cuda.is_available() = True`), but ROCm-awareness of MPI could not be detected. This may lead to performance degradation as direct MPI-communication between GPUs is not possible.",
            UserWarning,
        )
