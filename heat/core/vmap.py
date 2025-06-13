"""
Vmap module.
This implements a functionality similar to PyTorchs vmap function.
Requires PyTorch 2.0.0 or higher.
"""

import torch

from .dndarray import DNDarray
from .factories import array
from .communication import MPI_WORLD
from typing import Union, Tuple, Optional, Callable

__all__ = ["vmap"]


def vmap(
    func: Callable[[Tuple[torch.Tensor]], Tuple[torch.Tensor]],
    out_dims: Union[Tuple[int], int] = 0,
    randomness: str = "error",
    *,
    chunk_size: int = None,
) -> Callable[[Tuple[DNDarray]], Tuple[DNDarray]]:
    """
    Apply a function to a DNDarray in a vectorized way.
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
    """
    # check PyTorch version, return error if not 2.0.0 or higher
    if torch.__version__ < "2.0.0":
        raise RuntimeError("The function `heat.vmap` requires PyTorch 2.0.0 or higher.")
    # rough check of input argument types
    if not callable(func):
        raise TypeError("The input function `func` must be callable.")
    if randomness not in ["error", "different", "same"]:
        raise ValueError(
            "The input argument `randomness` must be one of the strings 'error', 'different', or 'same'."
        )
    if chunk_size is not None and not isinstance(chunk_size, int):
        raise TypeError("The input argument `chunk_size` must be None or an integer.")
    else:
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("If an integer, the input argument `chunk_size` must be at least 1.")

    def vmapped_func(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, DNDarray):
                raise TypeError(
                    f"All inputs to the vmapped-version of your function must be DNDarrays, but one is {type(arg)}."
                )
        in_dims = tuple([arg.split for arg in args])

        # apply Torch vmap to the input function and the result to the local arrays of the input DNDarray
        torch_vmap_func = torch.vmap(
            func, in_dims, out_dims, randomness=randomness, chunk_size=chunk_size
        )
        out_larrays = torch_vmap_func(*[arg.larray for arg in args], **kwargs)

        if isinstance(out_larrays, torch.Tensor):
            # circumvent misinterpretation of the following call of len() in case of a single output
            out_larrays = [out_larrays]
        if isinstance(out_dims, int):
            out_split = [out_dims] * len(out_larrays)
        else:
            out_split = out_dims
            if len(out_split) != len(out_larrays):
                raise ValueError(
                    f"The number of output DNDarrays ({len(out_larrays)}) must match the number of their split dimensions provided in `out_dims` ({len(out_split)})."
                )
        # generate output DNDarray(s)
        out_dndarrays = [
            array(out_larrays[k], is_split=out_split[k]) for k in range(len(out_larrays))
        ]
        return tuple(out_dndarrays)

    return vmapped_func
