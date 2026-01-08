Module heat.optim.dp_optimizer
==============================
MPI enabled data parallel optimizers

Classes
-------

`DASO(local_optimizer: torch.optim.optimizer.Optimizer, total_epochs: int, comm: heat.core.communication.MPICommunication = <heat.core.communication.MPICommunication object>, warmup_epochs: int = 4, cooldown_epochs: int = 4, scheduler: <module 'torch.optim.lr_scheduler' from '/Users/asthagupta/Documents/TUHH/Seminar_Winter of Code/heat/.venv/lib/python3.12/site-packages/torch/optim/lr_scheduler.py'> = None, stability_level: float = 0.05, max_global_skips: int = 8, sending_chunk_size: int = 10000000, downcast_type: torch.dtype = torch.bfloat16, use_mpi_groups: bool = True, skip_reduction_factor: int = 2, local_skip_factor: int = 4, verbose: bool = False)`
:   Optimizer wrapper to use the Distributed Asynchronous and Selective Optimization (DASO) method.

    This optimizer uses a local torch optimizer combined with the :func:`nn.DataParallelMultiGPU <heat.nn.data_parallel.DataParallelMultiGPU>`
    to create local DPNNs on each node consisting of the GPUs on each node. Then those networks communicate
    globally with MPI groups, each of which has a single GPU on each node.

    DASO uses both local and global synchronization operations. Local synchronization operations are intended to be
    done very frequently while global synchronizations are conducted asynchronously as the next batches are
    computed.

    This implementation requires that all nodes have the name number of GPUs.

    There are four phases to training:

        1. initialization: steps 1 to 8 below
        2. Warmup phase: blocking averaging update occurs for global synchronization step
        3. Cycling phase: for the global synchronization, the data is sent after a number of batches. the number of batches between synchronizations is referred to as `global_skips`. After the data is sent a number of batches pass before it is received (`batches_to_wait`). both of these cycle downward from `max_global_skips` for the global skips and 1/4th this value for `batches_to_wait`. When both values are equal to 1 and the loss is stable it will be reset to the initial values, then will decay again.
        4. Cooldown phase: blocking averaging update occurs for global synchronization step

    As example usage of this can be found in `heat/examples/nn/imagenet-DASO.py <https://github.com/helmholtz-analytics/heat/blob/504-docstring-formatting/examples/nn/imagenet-DASO.py>`_.

    The recommended checklist for using this class is as follows:

        1. initialize the local PyTorch process group and set the default device of the local GPUs.
        2. define the torch network
        3. define the `local_optimizer` -> a torch optimizer of your choice (tested with SGD)
        4. optional, choose a learning rate scheduler. This is only for those learning rates which will also step the optimizer
        5. initialize DASO with the local optimizers and parameters
        6. initialize :func:`nn.DataParallelMultiGPU <heat.nn.data_parallel.DataParallelMultiGPU>` with the torch network and DASO
        7. If using automatic mixed precision (:class:`torch.cuda.amp`), initialize the gradient scaler and add it to DASO (:func:`add_scaler`)
        8. ensure that the DataLoaders evenly distribute the data between all the processes. This can be done by using the `torch.utils.data.distributed.DistributedSampler <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`_ with the `num_replicas` and `rank` parameters
        9. call `daso_optimizer.epoch_loss_logic(training_loss)` at the end of
        10. set the number of batches per epoch (`daso_optimizer.last_batch = number_of_batches`)
        11. ensure that the step function used in training is that of the DASO optimizer

    Parameters
    ----------
    local_optimizer: torch.optim.Optimizer
        This optimizer handles the optimization of the local NN. Example: `torch.optim.SGD`. \n
        This can be any optimizer, although tests were only completed with SGD. Other optimizers may show
        unexpected behavior.
    total_epochs: int
        The total number of epochs for training. Needed to determine when to enter the cooldown phase.
    comm: MPICommunication, optional
        The MPI communicator to use for training. \n
        Default: :func:`MPI_WORLD <heat.core.comm.MPI_WORLD>`
    warmup_epochs: int, optional
        The number of epochs to complete with a blocking averaging operation after each batch before entering
        the cycling phase.\n
        Default: 4
    cooldown_epochs: int, optional
        The number of epochs with blocking averaging operations after each batch at the end of training.\n
        Default: 4
    scheduler: torch.optim.lr_scheduler, optional
        Local PyTorch learning rate scheduler. This must be used in the case that the scheduler's `step` function
        is supposed to be called instead of the optimizer's `step` function.\n
        Default: None
    stability_level: float, optional
        This can be viewed as the percent change threshold that the loss must exceed to be judged as improving.
        When the loss is within this percent change for 2 epochs, then it is judged as stable.\n
        Default: 0.05
    max_global_skips: int, optional
        The maximum number of batches between the beginning of a global synchronization process.\n
        Default: 8
    sending_chunk_size: int, optional
        During the global synchronization step, the network parameters are split into chunks of data to overlap
        communication and computation. This value is the maximum chunk size.\n
        Default: 10,000,000
    downcast_type: torch.dtype, optional
        Options: [torch.bfloat16, torch.half, torch.float]
        When the network parameters are sent during the global synchronization step, they are cast down to
        a smaller dtype, by default this is `torch.bfloat16`. Smaller torch dtypes are not implemented.
        torch.bfloat16.\n
        Default: torch.bfloat16
    use_mpi_groups: bool, optional
        Use MPI groups to divide the global communicator. If True, use MPI GROUPs, otherwise, use MPI SPLIT.\n
        Default: True
    skip_reduction_factor: int, optional
        How much to reduce the global/local skips by when the loss has stabilized.\n
        Default: 2
    local_skip_factor: int, optional
        How many local skips occur per global skip, i.e. number of local skips = global_skips // local_skip_factor.\n
        Default: 4
    verbose: bool, optional
        If true, print out a collection of debug messages.\n
        Default: False

    ### Methods

    `add_scaler(self, scaler: torch.cuda.amp.grad_scaler.GradScaler) ‑> None`
    :   Create a reference to torch's `torch.cuda.amp.GradScaler <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ used in torch's automatic mixed
        precision.

        Parameters
        ----------
        scaler: torch.cuda.amp.GradScaler
            the gradient scaler to be used

    `epoch_loss_logic(self, loss: torch.Tensor | int | float, loss_globally_averaged: bool = False) ‑> None`
    :   Function controlling the number of batches between global synchronizations and the batches to wait before
        receiving the sent parameters. The warm-up and cool-down phases are also controlled here.

        This function should be called at the end of each epoch with the training loss value at the end of the epoch.

        The number of batches between local synchronizations can also be modified here with minor code adjustments.

        Parameters
        ----------
        loss: torch.Tensor or float
            loss value of the current epoch
        loss_globally_averaged: bool, optional
            boolean if the loss is already globally averaged

    `print0(self, *args, **kwargs) ‑> None`
    :   Print a message on rank 0 if the class parameter `verbose` is set.

    `reset(self) ‑> None`
    :   Reset the optimizer to its base state

    `set_model(self, model: torch.nn.modules.module.Module) ‑> None`
    :   Set the local model for the optimizer.
        This should be called during the init of :func:`nn.DataParallelMultiGPU <heat.nn.data_parallel.DataParallelMultiGPU>`.
        However, this can also be called manually.

        Parameters
        ----------
        model: torch.nn.Module
            the local torch model.

    `step(self) ‑> None`
    :   Perform a single optimization step.
        This will perform the `step` operations of the local optimizer,
        local learning rate scheduler (if defined), and the gradient scaler used in automatic mixed
        precision (if defined).

        Also in the step is the logic used for when to send and receive the global/local synchronizations.
        Global Syncs occur on batches for which the modulus of the batch number and the `global_skip` number is 0.
        If `batches_to_wait` > 0, the next batches have only local syncs. After that number of batches,
        the data during the global sync phase is received.

        Local synchronization can also be turned off if desired by increasing `local_skips` above 1.

        Notes
        -----
        self.last_batch must be set!

    `zero_grad(self) ‑> None`
    :   Reset gradients of local optimizer's parameters.

`DataParallelOptimizer(torch_optimizer: torch.optim.optimizer.Optimizer, blocking: bool = False)`
:   Uses a torch.optim.Optimizer for data parallelism. It should be used in combination with DataParallel (DP) class.
    To optimize a DP module, DP optimizer has to be passed to DP module during its initialization.
    See :func:`nn.DataParallel <heat.nn.data_parallel.DataParallel>` for a basic example of usage.

    Attributes
    ----------
    torch_optimizer : torch.optim.Optimizer
        the wrapped Torch optimizer
    blocking : bool
        use blocking communications or not. will typically be overwritten by :func:`nn.DataParallel <heat.nn.data_parallel.DataParallel>`

    ### Methods

    `step(self) ‑> None`
    :   Force torch optimizer to update model parameters. For blocking, optimizer immediately updates parameters. For
        non-blocking, optimizer will update parameters during next forward.

    `zero_grad(self) ‑> None`
    :   Reset gradients of optimizer's params.
