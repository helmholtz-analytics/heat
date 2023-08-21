"""
handle different devices. Current options: CPU (default), GPU
"""

from __future__ import annotations

import torch

from typing import Any, Optional, Union

from . import communication


__all__ = ["Device", "cpu", "get_device", "sanitize_device", "use_device"]


class Device:
    """
    Implements a compute device. HeAT can run computations on different compute devices or backends.
    A device describes the device type and id on which said computation should be carried out.

    Parameters
    ----------
    device_type : str
        Represents HeAT's device name
    device_id : int
        The device id
    torch_device : str
        The corresponding PyTorch device type

    Examples
    --------
    >>> ht.Device("cpu", 0, "cpu:0")
    device(cpu:0)
    >>> ht.Device("gpu", 0, "cuda:0")
    device(gpu:0)
    """

    def __init__(self, device_type: str, device_id: int, torch_device: str):
        self.__device_type = device_type
        self.__device_id = device_id
        self.__torch_device = torch_device

    @property
    def device_type(self) -> str:
        """
        Return the type of :class:`~heat.core.device.Device` as a string.
        """
        return self.__device_type

    @property
    def device_id(self) -> int:
        """
        Return the identification number of :class:`~heat.core.device.Device`.
        """
        return self.__device_id

    @property
    def torch_device(self) -> str:
        """
        Return the type and id of :class:`~heat.core.device.Device` as a PyTorch device string object.
        """
        return self.__torch_device

    def __repr__(self) -> str:
        """
        Return the unambiguous information of :class:`~heat.core.device.Device`.
        """
        return f"device({self.__str__()})"

    def __str__(self) -> str:
        """
        Return the descriptive information of :class:`~heat.core.device.Device`.
        """
        return f"{self.device_type}:{self.device_id}"

    def __eq__(self, other: Any) -> bool:
        """
        Overloads the `==` operator for local equal check.

        Parameters
        ----------
        other : Any
            The object to compare with
        """
        if isinstance(other, Device):
            return self.device_type == other.device_type and self.device_id == other.device_id
        elif isinstance(other, torch.device):
            return self.device_type == other.type and self.device_id == other.index
        else:
            return NotImplemented


# create a CPU device singleton
cpu = Device("cpu", 0, "cpu")
"""
The standard CPU Device

Examples
--------
>>> ht.cpu
device(cpu:0)
>>> ht.ones((2, 3), device=ht.cpu)
DNDarray([[1., 1., 1.],
          [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)
"""

# define the default device to be the CPU
__default_device = cpu
# add a device string for the CPU device
__device_mapping = {cpu.device_type: cpu}

# add gpu support if available
if torch.cuda.device_count() > 0:
    # GPUs are assigned round-robin to the MPI processes
    gpu_id = communication.MPI_WORLD.rank % torch.cuda.device_count()
    # create a new GPU device
    gpu = Device("gpu", gpu_id, f"cuda:{gpu_id}")
    """
    The standard GPU Device

    Examples
    --------
    >>> ht.cpu
    device(cpu:0)
    >>> ht.ones((2, 3), device=ht.gpu)
    DNDarray([[1., 1., 1.],
          [1., 1., 1.]], dtype=ht.float32, device=gpu:0, split=None)
    """
    # add a GPU device string
    __device_mapping[gpu.device_type] = gpu
    __device_mapping["cuda"] = gpu
    # the GPU device should be exported as global symbol
    __all__.append("gpu")


def get_device() -> Device:
    """
    Retrieves the currently globally set default :class:`~heat.core.device.Device`.
    """
    return __default_device


def sanitize_device(device: Optional[Union[str, Device]] = None) -> Device:
    """
    Sanitizes a device or device identifier, i.e. checks whether it is already an instance of :class:`~heat.core.device.Device` or a string with
    known device identifier and maps it to a proper :class:`~heat.core.device.Device`.

    Parameters
    ----------
    device : str or Device, optional
        The device to be sanitized

    Raises
    ------
    ValueError
        If the given device id is not recognized
    """
    if device is None:
        return get_device()

    if isinstance(device, Device):
        return device

    try:
        return __device_mapping[device.strip().lower()]
    except (AttributeError, KeyError, TypeError):
        raise ValueError(f'Unknown device, must be one of {", ".join(__device_mapping.keys())}')


def use_device(device: Optional[Union[str, Device]] = None) -> None:
    """
    Sets the globally used default :class:`~heat.core.device.Device`.

    Parameters
    ----------
    device : str or Device
        The device to be set
    """
    global __default_device
    __default_device = sanitize_device(device)
