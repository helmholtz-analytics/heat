Module heat.sparse.dcsx_matrix
==============================
Provides DCSR_matrix, a distributed compressed sparse row matrix

Classes
-------

`DCSC_matrix(array: torch.Tensor, gnnz: int, gshape: Tuple[int, ...], dtype: datatype, split: Union[int, None], device: Device, comm: Communication, balanced: bool)`
:   Distributed Compressed Sparse Column Matrix. It is composed of
    PyTorch sparse_csc_tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor (layout ==> torch.sparse_csc)
        Local sparse array
    gnnz: int
        Total number of non-zero elements across all processes
    gshape : Tuple[int,...]
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        If split is not None, it denotes the axis on which the array is divided between processes.
        DCSR_matrix only supports distribution along axis 0.
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.

    ### Ancestors (in MRO)

    * heat.sparse.dcsx_matrix.__DCSX_matrix

    ### Instance variables

    `lindices: torch.Tensor`
    :   Local indices of the ``DCSC_matrix``

    `lindptr: torch.Tensor`
    :   Local indptr of the ``DCSC_matrix``

`DCSR_matrix(array: torch.Tensor, gnnz: int, gshape: Tuple[int, ...], dtype: datatype, split: Union[int, None], device: Device, comm: Communication, balanced: bool)`
:   Distributed Compressed Sparse Row Matrix. It is composed of
    PyTorch sparse_csr_tensors local to each process.

    Parameters
    ----------
    array : torch.Tensor (layout ==> torch.sparse_csr)
        Local sparse array
    gnnz: int
        Total number of non-zero elements across all processes
    gshape : Tuple[int,...]
        The global shape of the array
    dtype : datatype
        The datatype of the array
    split : int or None
        If split is not None, it denotes the axis on which the array is divided between processes.
        DCSR_matrix only supports distribution along axis 0.
    device : Device
        The device on which the local arrays are using (cpu or gpu)
    comm : Communication
        The communications object for sending and receiving data
    balanced: bool or None
        Describes whether the data are evenly distributed across processes.

    ### Ancestors (in MRO)

    * heat.sparse.dcsx_matrix.__DCSX_matrix

    ### Instance variables

    `lindices: torch.Tensor`
    :   Local indices of the ``DCSR_matrix``

    `lindptr: torch.Tensor`
    :   Local indptr of the ``DCSR_matrix``
