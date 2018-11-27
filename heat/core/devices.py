import torch

from .communication import MPI_WORLD


def cpu_index():
    """
    Retrieves the index of the CPU to be used for this MPI process.

    Returns
    -------
    cpu_index : int
        The CPU index
    """
    return 'cpu'


def gpu_index():
    """
    Retrieves the index of the GPU to be used for this MPI process.

    Returns
    -------
    gpu_index : int
        The GPU index
    """
    return 'cuda:{}'.format(MPI_WORLD.rank % torch.cuda.device_count())


__default_device = 'cpu'
__device_mapping = {
    'cpu': cpu_index,
    'gpu': gpu_index
}


def get_default_device():
    """
    Retrieves the currently globally set default device.

    Returns
    -------
    defaults device : str
        The default device
    """
    return __default_device


def sanitize_device(device):
    """
    Sanitizes a device identifier, i.e. checks whether it is of correct type, string content and normalizes the
    capitalization.

    Returns
    -------
    device_id : str
        The sanitized device id

    Raises
    ------
    TypeError
        If the given device id is not recognized
    """
    if device is None:
        device = __default_device

    try:
        return __device_mapping[device.lower()]()
    except (AttributeError, KeyError):
        raise TypeError('Unknown device, must be one of %s'.format(', '.join(__device_mapping.keys())))


def use_device(device=None):
    """
    Sets the default device to be used globally.

    Raises
    ------
    TypeError
        If the given device id is not recognized
    """
    global __default_device
    if device is None:
        return

    try:
        device = device.lower()
        __default_device = device in __device_mapping and device
    except (AttributeError, KeyError):
        raise TypeError('Unknown device, must be one of %s'.format(', '.join(__device_mapping.keys())))
