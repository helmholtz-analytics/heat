Module heat.core.devices
========================
handle different devices. Current options: CPU (default), GPU

Variables
---------

`cpu`
:   The standard CPU Device

    Examples
    --------
    >>> ht.cpu
    device(cpu:0)
    >>> ht.ones((2, 3), device=ht.cpu)
    DNDarray([[1., 1., 1.],
              [1., 1., 1.]], dtype=ht.float32, device=cpu:0, split=None)

Functions
---------

`get_device() ‑> heat.core.devices.Device`
:   Retrieves the currently globally set default :class:`~heat.core.device.Device`.

`sanitize_device(device: Optional[Union[str, Device]] = None) ‑> heat.core.devices.Device`
:   Sanitizes a device or device identifier, i.e. checks whether it is already an instance of :class:`~heat.core.device.Device` or a string with
    known device identifier and maps it to a proper :class:`~heat.core.device.Device`.

    Parameters
    ----------
    device : str or Device, optional
        The device to be sanitized

    Raises
    ------
    ValueError
        If the given device id is not recognized

`use_device(device: Optional[Union[str, Device]] = None) ‑> None`
:   Sets the globally used default :class:`~heat.core.device.Device`.

    Parameters
    ----------
    device : str or Device
        The device to be set

Classes
-------

`Device(device_type: str, device_id: int, torch_device: str)`
:   Implements a compute device. Heat can run computations on different compute devices or backends.
    A device describes the device type and id on which said computation should be carried out.

    Parameters
    ----------
    device_type : str
        Represents Heat's device name
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
    >>> ht.Device("gpu", 0, "mps:0")  # on Apple M1/M2
    device(gpu:0)

    ### Instance variables

    `device_id: int`
    :   Return the identification number of :class:`~heat.core.device.Device`.

    `device_type: str`
    :   Return the type of :class:`~heat.core.device.Device` as a string.

    `torch_device: str`
    :   Return the type and id of :class:`~heat.core.device.Device` as a PyTorch device string object.
