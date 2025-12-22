Module heat.utils.data.mnist
============================
File for the MNIST dataset definition in heat

Classes
-------

`MNISTDataset(root: str, train: bool = True, transform: Callable = None, target_transform: Callable = None, download: bool = True, split: int = 0, ishuffle: bool = False, test_set: bool = False)`
:   Dataset wrapper for `torchvision.datasets.MNIST <https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST>`_.
    This implements all of the required functions mentioned in :class:`heat.utils.data.Dataset`. The ``__getitem__`` and ``__len__`` functions are inherited from
    `torchvision.datasets.MNIST <https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST>`_.

    Parameters
    ----------
    root : str
        Directory containing the MNIST dataset
    train : bool, optional
        If the data is the training dataset or not, default is True
    transform : Callable, optional
        Transform to be applied to the data dataset in the ``__getitem__`` function, default is ``None``
    target_transform : Callable, optional
        Transform to be applied to the target dataset in the ``__getitem__`` function, default is ``None``
    download : bool, optional
        If the data does not exist in the directory, download it if True (default)
    split : int, optional
        On which access to split the data when it is loaded into a ``DNDarray``
    ishuffle : bool, optional
        Flag indicating whether to use non-blocking communications for shuffling the data between epochs
        Note: if True, the ``Ishuffle()`` function must be defined within the class
        Default: ``False``
    test_set : bool, optional
        If this dataset is the testing set then keep all of the data local
        Default: ``False``

    Attributes
    ----------
    htdata : DNDarray
        full data
    httargets : DNDarray
        full target data
    comm : communication.MPICommunicator
        heat communicator for sending data between processes
    _cut_slice : slice
        slice to remove the last element if all are not equal in length
    lcl_half : int
        integer value of half of the data on the process
    data : torch.Tensor
        the local data on a process
    targets : torch.Tensor
        the local targets on a process
    ishuffle : bool
        flag indicating if non-blocking communications are used for shuffling the data between epochs
    test_set : bool
        if this dataset is the testing set then keep all of the data local

    Notes
    -----
    For other attributes see `torchvision.datasets.MNIST <https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST>`_.

    ### Ancestors (in MRO)

    * torchvision.datasets.mnist.MNIST
    * torchvision.datasets.vision.VisionDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Methods

    `Ishuffle(self)`
    :   Uses the :func:`datatools.dataset_ishuffle` function to shuffle the data between the processes

    `Shuffle(self)`
    :   Uses the :func:`datatools.dataset_shuffle` function to shuffle the data between the processes
