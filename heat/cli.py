"""
Heat command line interface module.
"""

import torch
import platform
import mpi4py
import argparse

from heat.core.version import __version__ as ht_version
from heat.core.communication import CUDA_AWARE_MPI


def cli() -> None:
    """
    Command line interface entrypoint.
    """
    parser = argparse.ArgumentParser(
        prog="heat", description="Commmand line utilities of the Helmholtz Analytics Toolkit"
    )
    parser.add_argument(
        "-i", "--info", action="store_true", help="Print version and platform information"
    )

    args = parser.parse_args()
    if args.info:
        plaform_info()
    else:
        parser.print_help()


def plaform_info():
    """
    Print the current software stack being used by heat, including available devices.
    """
    print("HeAT: Helmholtz Analytics Toolkit")
    print(f"  Version: {ht_version}")
    print(f"  Platform: {platform.platform()}")

    print(f"  mpi4py Version: {mpi4py.__version__}")
    print(f"  MPI Library Version: {mpi4py.MPI.Get_library_version()}")

    print(f"  Torch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        def_device = torch.cuda.current_device()
        print(f"    Device count: {torch.cuda.device_count()}")
        print(f"    Default device: {def_device}")
        print(f"    Device name: {torch.cuda.get_device_name(def_device)}")
        print(f"    Device name: {torch.cuda.get_device_properties(def_device)}")
        print(
            f"   Device memory: {torch.cuda.get_device_properties(def_device).total_memory / 1024**3} GiB"
        )
        print(f"    CUDA Aware MPI: {CUDA_AWARE_MPI}")
