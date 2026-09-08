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
import functools
import importlib
import importlib.util
from enum import Enum
from types import ModuleType
from typing import Optional, Tuple

__all__ = [
    "MPILibrary",
    "MPILibraryInfo",
    "PLATFORM",
    "TORCH_VERSION",
    "TORCH_CUDA_IS_AVAILABLE",
    "CUDA_IS_ACTUALLY_ROCM",
    "CUDA_AWARE_MPI",
    "ROCM_AWARE_MPI",
    "GPU_AWARE_MPI",
    "mpi_library",
]


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


def _get_mpi_library() -> MPILibraryInfo:
    library = mpi4py.MPI.Get_library_version().split()
    match library:
        case ["Open", "MPI", *_]:
            return MPILibraryInfo(MPILibrary.OpenMPI, library[2])
        case ["Intel(R)", "MPI", *_]:
            return MPILibraryInfo(MPILibrary.IntelMPI, library[3])
        case ["MPICH", "Version:", *_]:
            return MPILibraryInfo(MPILibrary.MPICH, library[2])
        case ["MVAPICH", "Version:", *_]:
            return MPILibraryInfo(MPILibrary.MVAPICH, library[2])
        case ["===", "ParaStation", "MPI", *_]:
            return MPILibraryInfo(MPILibrary.ParaStationMPI, library[3])
        case _:
            return MPILibraryInfo(MPILibrary.Other, "unknown")


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
                # Seems to be broken, disabled by default for now
                # return cuda, rocm
                return False, False
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
        case MPILibrary.ParaStationMPI:
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


# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------

#: Third-party packages Heat can use if present, mapped to the name of the ``pyproject``
#: extra that installs them, i.e. ``pip install heat[<extra>]``.
OPTIONAL_DEPENDENCIES = {
    "h5py": "hdf5",
    "netCDF4": "netcdf",
    "zarr": "zarr",
    "pandas": "pandas",
}


@functools.lru_cache(maxsize=None)
def is_installed(name: str) -> bool:
    """
    Returns ``True`` if the module ``name`` appears to be installed, ``False`` otherwise.

    This is a cheap presence check based on :func:`importlib.util.find_spec`; it locates
    the module without importing it, so it does not pay the cost of pulling in a heavy
    library. A module can be present but fail to import (a broken build, an ABI mismatch
    against a system library); use :func:`has_dependency` when that distinction matters.

    Parameters
    ----------
    name : str
        Import name of the module, e.g. ``"h5py"``.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        # find_spec imports parent packages and may raise for a broken installation
        return False


@functools.lru_cache(maxsize=None)
def _import_optional(name: str) -> Tuple[Optional[ModuleType], Optional[str]]:
    """
    Imports the optional module ``name``, returning ``(module, error)``.

    Exactly one element is ever non-``None``. Any exception is captured rather than
    propagated, so that a broken installation of an optional dependency degrades that one
    file format instead of making ``import heat`` fail outright.

    Parameters
    ----------
    name : str
        Import name of the module, e.g. ``"h5py"``.
    """
    if not is_installed(name):
        return None, f"no module named {name!r}"
    try:
        return importlib.import_module(name), None
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"


def optional_import(name: str) -> Optional[ModuleType]:
    """
    Imports and returns the optional module ``name``, or ``None`` if it is unavailable.

    Parameters
    ----------
    name : str
        Import name of the module, e.g. ``"h5py"``.
    """
    return _import_optional(name)[0]


def has_dependency(name: str) -> bool:
    """
    Returns ``True`` if the optional module ``name`` is installed and imports cleanly.

    Parameters
    ----------
    name : str
        Import name of the module, e.g. ``"h5py"``.
    """
    return optional_import(name) is not None


def require_dependency(name: str) -> ModuleType:
    """
    Imports and returns the optional module ``name``, raising if it is unavailable.

    Parameters
    ----------
    name : str
        Import name of the module, e.g. ``"h5py"``.

    Raises
    ------
    RuntimeError
        If the module is not installed, or is installed but fails to import.
    """
    module, error = _import_optional(name)
    if module is None:
        extra = OPTIONAL_DEPENDENCIES.get(name)
        hint = f" Install it with `pip install heat[{extra}]`." if extra else ""
        raise RuntimeError(f"{name} is required for this operation ({error}).{hint}")
    return module


@functools.lru_cache(maxsize=None)
def hdf5_has_mpi() -> bool:
    """
    Returns ``True`` if the installed ``h5py`` was built with parallel (MPI-IO) support.
    """
    h5py = optional_import("h5py")
    return bool(h5py.get_config().mpi) if h5py is not None else False


@functools.lru_cache(maxsize=None)
def netcdf_has_parallel() -> bool:
    """
    Returns ``True`` if the installed ``netCDF4`` was built with parallel I/O support.
    """
    netcdf = optional_import("netCDF4")
    if netcdf is None:
        return False
    return bool(
        netcdf.__dict__.get("__has_parallel4_support__", False)
        or netcdf.__dict__.get("__has_pnetcdf_support__", False)
        or netcdf.__dict__.get("__has_nc_par__", False)
    )
