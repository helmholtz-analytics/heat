Module heat.nn.data_parallel
============================
General data parallel neural network classes.

Classes
-------

`DataParallel(module: torch.nn.modules.module.Module, comm: heat.core.communication.MPICommunication, optimizer: heat.optim.dp_optimizer.DataParallelOptimizer | List | Tuple, blocking_parameter_updates: bool = False)`
:   Implements data parallelism across multiple processes. This means that the same model will be run locally
    on each process. Creation of the model is similar to PyTorch, the only changes are using HeAT layers (ht.nn.layer)
    in the initialization of the network/optimizer. If there is not a HeAT layer, it will fall back to the PyTorch layer
    of the same name. The same is true for the optimizer. It's possible to use more than one optimizer, but
    communication during parameter updates is limited to blocking. The same limitation takes effect when passing an
    optimizer that does not deal exactly with the set of model's parameters. For the given model both the
    ``__init__()`` and ``forward()`` functions must be defined in the class defining the network.

    An example of this is shown in `examples/mnist.py <https://github.com/helmholtz-analytics/heat/blob/504-docstring-formatting/examples/nn/mnist.py>`_.

    It is highly recommended that a HeAT DataLoader is used, see :func:`ht.utils.data.DataLoader <heat.utils.data.datatools.DataLoader>`.
    The default communications scheme for this is blocking. The blocking scheme will average the model parameters during
    the backwards step, synchronizing them before the next model iteration.

    Usage of more than one optimizer forces MPI communication to be parameter updates to use blocking communications.

    Attributes
    ----------
    module : torch.nn.Module
        The local module
    comm : MPICommunication
        Communicator to use
    optimizer : heat.DataParallelOptimizer, List, Tuple
        Individual or sequence of DataParallelOptimizers to be used
    blocking_parameter_updates : bool, optional
        Flag indicating the usage of blocking communications for parameter updates
        Default: non-blocking updates (``False``)

    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, *inputs: tuple, **kwargs: dict) ‑> torch.Tensor`
    :   Do the forward step for the network, receive the parameters from the last

`DataParallelMultiGPU(module: torch.nn.modules.module.Module, optimizer: heat.optim.dp_optimizer.DASO, comm: heat.core.communication.MPICommunication = <heat.core.communication.MPICommunication object>)`
:   Creates data parallel networks local to each node using PyTorch's distributed class. This does NOT
    do any global synchronizations. To make optimal use of this structure, use :func:`ht.optim.DASO <heat.optim.dp_optimizer.DASO>`.

    Notes
    -----
    The PyTorch distributed process group must already exist before this class is initialized.

    Parameters
    ----------
    module: torch.nn.Module
        an implemented PyTorch model
    optimizer: optim.DASO
        A DASO optimizer. Other optimizers are not yet implemented. The DASO optimizer should be
        defined prior to calling this class.
    comm: MPICommunication, optional
        A global communicator.
        Default: :func:`MPICommunication <heat.core.comm.MPICommunication>`

    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, *inputs: Tuple, **kwargs: Dict) ‑> torch.Tensor`
    :   Calls the forward method for the torch model
