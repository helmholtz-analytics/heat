"""
Everything you need to know about the configuration of Heat
"""

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
    ParastationMPI = "psmpi"


@dataclasses.dataclass
class MPILibraryInfo:
    name: MPILibrary
    version: str


def _get_mpi_library() -> MPILibraryInfo:
    library = mpi4py.MPI.Get_library_version().split()
    match library:
        case ["Open", "MPI", *_]:
            return MPILibraryInfo(MPILibrary.OpenMPI, library[2])
        case ["Intel(R)", "MPI", *_]:
            return MPILibraryInfo(MPILibrary.IntelMPI, library[3])
        case ["MPICH", "Version:", *_]:
            return MPILibraryInfo(MPILibrary.MPICH, library[2])
        ### Missing libraries
        case _:
            print("Did not find a matching library")


def _check_gpu_aware_mpi(library: MPILibraryInfo) -> tuple[bool, bool]:
    match library.name:
        case MPILibrary.OpenMPI:
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
                if library.version.startswith("v4."):
                    rocm = cuda
                elif library.version.startswith("v5."):
                    rocm = "rocm" in extensions or "hip" in extensions
                return cuda, rocm
            except Exception as e:  # noqa E722
                return False, False
        case MPILibrary.IntelMPI:
            return False, False
        case MPILibrary.MVAPICH:
            cuda = os.environ.get("MV2_USE_CUDA") == "1"
            rocm = os.environ.get("MV2_USE_ROCM") == "1"
            return cuda, rocm
        case MPILibrary.MPICH:
            cuda = os.environ.get("MPIR_CVAR_ENABLE_HCOLL") == "1"
            rocm = False
            return cuda, rocm
        case MPILibrary.CrayMPI:
            cuda = os.environ.get("MPICH_GPU_SUPPORT_ENABLED") == "1"
            rocm = os.environ.get("MPICH_GPU_SUPPORT_ENABLED") == "1"
            return cuda, rocm
        case MPILibrary.ParastationMPI:
            cuda = os.environ.get("PSP_CUDA") == "1"
            rocm = False
            return cuda, rocm
        case _:
            return False, False


PLATFORM = platform.platform()
TORCH_VERSION = torch.__version__
TORCH_CUDA_IS_AVAILABLE = torch.cuda.is_available()
CUDA_IS_ACTUALLY_ROCM = "rocm" in TORCH_VERSION

mpi_library = _get_mpi_library()
CUDA_AWARE_MPI, ROCM_AWARE_MPI = _check_gpu_aware_mpi(mpi_library)
GPU_AWARE_MPI = False

# warn the user if CUDA/ROCm-aware MPI is not available, but PyTorch can use GPUs with CUDA/ROCm
if TORCH_CUDA_IS_AVAILABLE:
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
    else:
        GPU_AWARE_MPI = True
