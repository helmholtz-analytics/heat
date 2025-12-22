Module heat.utils.data.datatools
================================
Function and classes useful for loading data into neural networks

Functions
---------

`dataset_ishuffle(dataset: heat.utils.data.datatools.Dataset | torch.utils.data.dataset.Dataset, attrs: List[list])`
:   Shuffle the given attributes of a dataset across multiple processes, using non-blocking communications.
    This will send half of the data to rank + 1. The data must be received by the :func:`dataset_irecv` function.

    This function will be called by the DataLoader automatically if ``dataset.ishuffle = True``. This is set either
    during the definition of the class of its initialization by a given paramete.

    Parameters
    ----------
    dataset : Dataset
        the dataset to shuffle
    attrs : List[List[str, str], ... ]
        List of lists each of which contains 2 strings. The strings are the handles corresponding to the Dataset
        attributes corresponding to the global data DNDarray and the local data of that array, i.e. [["htdata, "data"],]
        would shuffle the htdata around and set the correct amount of data for the ``dataset.data`` attribute. For
        multiple parameters multiple lists are required. I.e. [["htdata", "data"], ["httargets", "targets"]]

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.

`dataset_shuffle(dataset: heat.utils.data.datatools.Dataset | torch.utils.data.dataset.Dataset, attrs: List[list])`
:   Shuffle the given attributes of a dataset across multiple processes. This will send half of the data to rank + 1.
    Once the new data is received, it will be shuffled into the existing data on the process.
    This function will be called by the DataLoader automatically if ``dataset.ishuffle = False``.
    attrs should have the form [[torch.Tensor, DNDarray], ... i.e. [['data', 'htdata`]] assume that all of the attrs have the same dim0 shape as the local data

    Parameters
    ----------
    dataset : Dataset
        the dataset to shuffle
    attrs : List[List[str, str], ... ]
        List of lists each of which contains 2 strings. The strings are the handles corresponding to the Dataset
        attributes corresponding to the global data DNDarray and the local data of that array, i.e. [["data, "htdata"],]
        would shuffle the htdata around and set the correct amount of data for the ``dataset.data`` attribute. For
        multiple parameters multiple lists are required. I.e. [["data", "htdata"], ["targets", "httargets"]]

    Notes
    -----
    ``dataset.comm`` must be defined for this function to work.

Classes
-------

`DataLoader(dataset: torch.utils.data.dataset.Dataset | heat.utils.data.partial_dataset.PartialH5Dataset, batch_size: int = 1, num_workers: int = 0, collate_fn: Callable = None, pin_memory: bool = False, drop_last: bool = False, timeout: int | float = 0, worker_init_fn: Callable = None)`
:   The combines either a :func:`DNDarray <heat.core.dndarray.DNDarray>` or a torch `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_
    with a sampler. This provides an iterable over the local dataset and it will shuffle the data at the end of the
    iterator. If a :func:`DNDarray <heat.core.dndarray.DNDarray>` is given, then a :func:`Dataset` will be created
    internally.

    Currently, this only supports only map-style datasets with single-process loading. It uses the random
    batch sampler. The rest of the ``DataLoader`` functionality mentioned in `torch.utils.data.dataloader <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader>`_ applies.

    Arguments:
        dataset : :func:`Dataset`, torch `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_, :func:`heat.utils.data.partial_dataset.PartialH5Dataset`
            A torch dataset from which the data will be returned by the created iterator
        batch_size : int, optional
            How many samples per batch to load\n
             Default: 1
        num_workers : int, optional
            How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.\n
            Default: 0
        collate_fn : callable, optional
            Merges a list of samples to form a mini-batch of torch.Tensor(s).  Used when using batched loading from a
            map-style dataset.\n
            Default: None
        pin_memory : bool, optional
            If ``True``, the data loader will copy torch.Tensors into CUDA pinned memory before returning them.
            If your data elements are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below. \n
            Default: False
        drop_last : bool, optional
            Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
            the last batch will be smaller.\n
            Default: ``False``
        timeout : int or float, optional
            If positive, the timeout value for collecting a batch from workers. Should always be non-negative.\n
            Default: 0
        worker_init_fn : callable, optional
            If not ``None``, this will be called on each worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading.\n
            default: None

    Attributes
    ----------
    dataset : :func:`Dataset`, torch `Dataset <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_, :func:`heat.utils.data.partial_dataset.PartialH5Dataset`
        The dataset created from the local data
    DataLoader : `torch.utils.data.dataloader <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.DataLoader>`_
        The local DataLoader object. Used in the creation of the iterable and the length
    _first_iter : bool
        Flag indicating if the iterator created is the first one. If it is not, then the data will be shuffled before
        the iterator is created
    last_epoch : bool
        Flag indicating last epoch

`Dataset(array, transforms: List | Callable | None = None, ishuffle: bool | None = False, test_set: bool | None = False)`
:   An abstract class representing a given dataset. This inherits from torch.utils.data.Dataset.

    This class is a general example for what should be done to create a Dataset. When creating a dataset all of the
    standard attributes should be set, the ``__getitem__``, ``__len__``, and ``shuffle`` functions must be defined.

        - ``__getitem__`` : how an item is given to the network
        - ``__len__`` : the number of data elements to be given to the network in total
        - ``Shuffle()`` : how the data should be shuffled between the processes. The function shown below is for a dataset composed of only data and without targets. The function :func:`dataset_shuffle` abstracts this. For this function only the dataset and a list of attributes to shuffle are given.\n
        - ``Ishuffle()`` : A non-blocking version of ``Shuffle()``, this is handled in the abstract function :func:`dataset_ishuffle`. It works similarly to :func:`dataset_shuffle`.

    As the amount of data across processes can be non-uniform, the dataset class will slice off the remaining elements
    on whichever processes have more data than the others. This should only be 1 element.
    The shuffle function will shuffle all of the data on the process.

    It is recommended that for ``DNDarray`` s, the split is either 0 or None

    Parameters
    ----------
    array : DNDarray
        DNDarray for which to great the dataset
    transform : Callable
        Transformation to call before a data item is returned
    ishuffle : bool, optional
        flag indicating whether to use non-blocking communications for shuffling the data between epochs
        Note: if ``True``, the ``Ishuffle()`` function must be defined within the class\n
        Default: False

    Attributes
    ----------
    These are the required attributes.

    htdata : DNDarray
        Full data
    _cut_slice : slice
        Slice to cut off the last element to get a uniform amount of data on each process
    comm : MPICommunicator
        Communication object used to send the data between processes
    lcl_half : int
        Half of the number of data elements on the process
    data : torch.Tensor
        The local data to be used in training
    transforms : Callable
        Transform to be called during the getitem function
    ishuffle : bool
        Flag indicating if non-blocking communications are used for shuffling the data between epochs

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Methods

    `Ishuffle(self)`
    :   Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.

    `Shuffle(self)`
    :   Send half of the local data to the process ``self.comm.rank + 1`` if available, else wrap around. After
        receiving the new data, shuffle the local tensor.
