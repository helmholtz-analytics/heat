"""
This implements a functionality similar to PyTorchs vmap function.
"""

import torch
from .dndarray import DNDarray
from .factories import array

__all__ = ["vmap"]


def vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    """
    This function is used to apply a function to a DNDarray in a vectorized way.
    `heat.vmap` return a callable that can be applied to a DNDarray.
    Vectorization needs to take place at least along the split axis.

    Parameters
    ----------
    func : callable
        The function to apply to the DNDarray. It must take a DNDarray as its first argument.
    in_dims : str or sequence of str
        The dimensions of the input DNDarray that should be mapped over.
        Default is 0.
    out_dims : str or sequence of str
        The dimensions of the output DNDarray that should be mapped over.
        Default is 0.
    randomness : {'error', 'different', 'same'}, optional
        Determines how to handle randomness in the function to be vmapped.
        If 'error' (default), an error is raised if the function to be mapped contains randomness.
        If 'different', randomness will be different for each batch; if 'same', randomness will be the same for each batch.
        (This argument is directly passed to the underlying PyTorch vmaps; see the corresponding PyTorch documentation for more information.)
    chunk_size : int or sequence of int, optional
        The size of the chunks to use for the process-local computation.
        If None (default), apply a single PyTorch vmap over the process-local chunks of data. If not None, then compute the process-local PyTorch vmap `chunk_size`
        many samples at a time. Note that `chunk_size=1` is equivalent to computing the process-local PyTorch vmap's with a for-loop.
        If you run into memory issues computing the vmap, please try a non-None chunk_size.

    Note
    ------
    This function is a wrapper around PyTorch's `torch.vmap` function. In essence, a PyTorch vmap is applied to the input function `func` on each MPI process separately.
    This process-local PyTorch-vmapped function is then applied to the process-local chunks of the input DNDarray.
    """
    # rough check of input argument types
    if not isinstance(func, callable):
        raise TypeError("The input function `func` must be callable.")
    if not isinstance(in_dims, (int, tuple)):
        raise TypeError("The input argument `in_dims` must be an integer or a tuple of integers.")
    if not isinstance(out_dims, (int, tuple)):
        raise TypeError("The input argument `out_dims` must be an integer or a tuple of integers.")
    if not isinstance(chunk_size, (int, tuple, type(None))):
        raise TypeError(
            "The input argument `chunk_size` must be an integer, a tuple of integers, or None."
        )
    # check input argument values
    if isinstance(in_dims, int):
        in_dims = (in_dims,)
    for dim in in_dims:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("`in_dims` may only contain *non-negative integers*.")
    if isinstance(out_dims, int):
        out_dims = (out_dims,)
    for dim in out_dims:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("`out_dims` may only contain *non-negative integers*.")
    if not len(in_dims) == len(out_dims):
        raise ValueError("The input arguments `in_dims` and `out_dims` must have the same length.")
    if randomness not in ["error", "different", "same"]:
        raise ValueError(
            "The input argument `randomness` must be one of the strings 'error', 'different', or 'same'."
        )
    if chunk_size is not None:
        if isinstance(chunk_size, int):
            chunk_size = (chunk_size,)
        for size in chunk_size:
            if not isinstance(size, int) or size < 1:
                raise ValueError(
                    "If a tuple, the input argument `chunk_size` must be a tuple of *positive integers*."
                )
        if not len(chunk_size) == len(in_dims):
            raise ValueError(
                "If not None, the input argument `chunk_size` must have the same length as `in_dims`."
            )

    def vmapped_func(x: DNDarray):
        if not isinstance(x, DNDarray):
            raise TypeError("The input to the vmapped-version of your function must be a DNDarray.")
        if x.split is None:
            # if the input DNDarray is not split
            new_is_split = False
            new_split = None
        else:
            # if input DNDarray is split, check if split axis is in in_dims and determine split axis of output DNDarray
            if x.split not in in_dims:
                raise ValueError(
                    'The split axis of the input DNDarray to your vmapped function must be in the dimensions mapped over by vmap ("`in_dims`").'
                )
            new_is_split = True
            new_split = out_dims[in_dims.index(x.split)]
        # apply Torch vmap to the input function and the result to the local arrays of the input DNDarray
        torch_vmap_func = torch.vmap(
            func, in_dims, out_dims, randomness=randomness, chunk_size=chunk_size
        )
        y_larray = torch_vmap_func(x.larray)
        y = array(y_larray, is_split=new_is_split, split=new_split)
        return y

    return vmapped_func
