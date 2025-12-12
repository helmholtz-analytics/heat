Module heat.core.vmap
=====================
Vmap module.
This implements a functionality similar to PyTorchs vmap function.
Requires PyTorch 2.0.0 or higher.

Functions
---------

`vmap(func: Callable[[Tuple[torch.Tensor]], Tuple[torch.Tensor]], out_dims: Tuple[int] | int = 0, randomness: str = 'error', *, chunk_size: int = None) ‑> Callable[[Tuple[heat.core.dndarray.DNDarray]], Tuple[heat.core.dndarray.DNDarray]]`
:   Apply a function to a DNDarray in a vectorized way.
    `heat.vmap` return a callable that can be applied to DNDarrays.
    Vectorization will automatically take place along the split axis/axes of the DNDarray(s);
    therefore, unlike in PyTorch, there is no argument `in_dims`.
    What we here refer to as "split axis/dimension" in the Heat terminology is often referred to as "batch axis/dimension" in the PyTorch terminology.

    Parameters
    ----------
    func : callable
        The function to apply in a vmapped way to the DNDarray(s). It must take PyTorch tensor(s) as positional arguments.
        Additional parameters, not to be vmapped over, can be passed as keyword arguments. The callable returned by
        by `heat.vmap` will also accept these keyword arguments.
    out_dims : int or tuple of int, optional
        The dimensions of the output(s) that are mapped over; identical to the split dimension(s) of the output(s).
        Default is 0.
    randomness : {'error', 'different', 'same'}, optional
        Determines how to handle randomness in the function to be vmapped. This argument is directly passed to the underlying PyTorch vmaps;
        see the corresponding PyTorch documentation for more information and the note below.
        If 'error' (default), an error is raised if the function to be mapped contains randomness.
    chunk_size : int, optional
        The size of the chunks to use for the process-local computation.
        If None (default), apply a single PyTorch vmap over the process-local chunks of data. If not None, then compute the process-local PyTorch vmap `chunk_size`
        many samples at a time. Note that `chunk_size=1` is equivalent to computing the process-local PyTorch vmap's with a for-loop.
        If you run into memory issues computing the vmap, please try a non-None chunk_size.

    Note
    ------
    This function is a wrapper around PyTorch's `torch.vmap` function. In essence, a PyTorch vmap is applied to the input function `func` on each MPI process separately.
    This process-local PyTorch-vmapped function is then applied to the process-local chunks of the input DNDarray(s).

    Please note that the options 'same' and 'different' for `randomness` will result in behaviour different from the one known by PyTorch as (at least currently)
    no actions are taken to synchronize randomness across the MPI processes.
