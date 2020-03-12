import torch

from .communication import MPI_WORLD


__all__ = ["cpu", "get_device", "sanitize_device", "use_device"]


class Device:
    """
    Implements a compute device.

    HeAT can run computations on different compute devices or backends. A device describes the device type and id on
    which said computation should be carried out.

    Parameters
    ----------
    device_type : str
        represents HeAT's device name
    device_id : int
        the device id
    torch_device : str
        the corresponding PyTorch device type

    Examples
    --------
    >>> # array on cpu
    >>> cpu_array = ht.ones((2, 3), device=ht.cpu)
    >>> # array on gpu
    >>> gpu_array = ht.ones((2, 3), device=ht.gpu)
    device(cpu:0)
    """

    def __init__(self, device_type, device_id, torch_device):
        self.__device_type = device_type
        self.__device_id = device_id
        self.__torch_device = torch_device

    @property
    def device_type(self):
        return self.__device_type

    @property
    def device_id(self):
        return self.__device_id

    @property
    def torch_device(self):
        return self.__torch_device

    def __repr__(self):
        return "device({})".format(self.__str__())

    def __str__(self):
        return "{}:{}".format(self.device_type, self.device_id)


# create a CPU device singleton
cpu = Device("cpu", 0, "cpu:0")

# define the default device to be the CPU
__default_device = cpu
# add a device string for the CPU device
__device_mapping = {cpu.device_type: cpu}

# add gpu support if available
if torch.cuda.device_count() > 0:
    # GPUs are assigned round-robin to the MPI processes
    gpu_id = MPI_WORLD.rank % torch.cuda.device_count()
    # create a new GPU device
    gpu = Device("gpu", gpu_id, "cuda:{}".format(gpu_id))
    # add a GPU device string
    __device_mapping[gpu.device_type] = gpu
    # the GPU device should be exported as global symbol
    __all__.append("gpu")


def get_device():
    """
    Retrieves the currently globally set default device.

    Returns
    -------
    defaults device : Device
        The currently set default device.
    """
    return __default_device


def sanitize_device(device):
    """
    Sanitizes a device or device identifier, i.e. checks whether it is already an instance of Device or a string with
    known device identifier and maps it to a proper Device.

    Parameters
    ----------
    device : str, Device or None
        The device to be sanitized

    Returns
    -------
    sanitized_device : Device
        The matching Device instance

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
        raise ValueError(
            "Unknown device, must be one of {}".format(", ".join(__device_mapping.keys()))
        )


def use_device(device=None):
    """
    Sets the globally used default device.

    Parameters
    ----------
    device : str, Device or None
        The device to be set
    """
    global __default_device
    __default_device = sanitize_device(device)
