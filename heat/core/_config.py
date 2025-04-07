"""
Everything you need to know about the configuration of Heat
"""

import torch
import platform
import mpi4py
import subprocess
import os
import warnings

PLATFORM = platform.platform()
MPI_LIBRARY_VERSION = mpi4py.MPI.Get_library_version()
TORCH_VERSION = torch.__version__
TORCH_CUDA_IS_AVAILABLE = torch.cuda.is_available()
CUDA_IS_ACTUALLY_ROCM = "rocm" in TORCH_VERSION

CUDA_AWARE_MPI = False
# check whether there is CUDA-aware OpenMPI
try:
    buffer = subprocess.check_output(["ompi_info", "--parsable", "--all"])
    CUDA_AWARE_MPI = b"mpi_built_with_cuda_support:value:true" in buffer
except:  # noqa E722
    pass
# do the same for MVAPICH
CUDA_AWARE_MPI = CUDA_AWARE_MPI or os.environ.get("MV2_USE_CUDA") == "1"
# do the same for MPICH
CUDA_AWARE_MPI = CUDA_AWARE_MPI or os.environ.get("MPIR_CVAR_ENABLE_HCOLL") == "1"
# do the same for ParaStationMPI
CUDA_AWARE_MPI = CUDA_AWARE_MPI or os.environ.get("PSP_CUDA") == "1"

ROCM_AWARE_MPI = False
# TODO: do these checks for ROCM-aware MPI

# warn the user if CUDA/ROCm-aware MPI is not available, but PyTorch can use GPUs with CUDA/ROCm
if TORCH_CUDA_IS_AVAILABLE:
    if not CUDA_IS_ACTUALLY_ROCM and not CUDA_AWARE_MPI:
        warnings.warn(
            f"Heat has CUDA GPU-support (PyTorch version {TORCH_VERSION} and `torch.cuda.is_available() = True`), but CUDA-awareness of MPI could not be detected. This may lead to performance degradation as direct MPI-communication between GPUs is not possible.",
            UserWarning,
        )
    if CUDA_IS_ACTUALLY_ROCM and not ROCM_AWARE_MPI:
        warnings.warn(
            f"Heat has ROCm GPU-support (PyTorch version {TORCH_VERSION} and `torch.cuda.is_available() = True`), but ROCm-awareness of MPI could not be detected. This may lead to performance degradation as direct MPI-communication between GPUs is not possible.",
            UserWarning,
        )
