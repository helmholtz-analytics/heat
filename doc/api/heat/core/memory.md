Module heat.core.memory
=======================
Utilities to manage the internal memory of an array.

Functions
---------

`copy(x: heat.core.dndarray.DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Return a deep copy of the given object.

    Parameters
    ----------
    x : DNDarray
        Input array to be copied.

    Examples
    --------
    >>> a = ht.array([1, 2, 3])
    >>> b = ht.copy(a)
    >>> b
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> a[0] = 4
    >>> a
    DNDarray([4, 2, 3], dtype=ht.int64, device=cpu:0, split=None)
    >>> b
    DNDarray([1, 2, 3], dtype=ht.int64, device=cpu:0, split=None)

`sanitize_memory_layout(x: torch.Tensor, order: str = 'C') ‑> torch.Tensor`
:   Return the given object with memory layout as defined below. The default memory distribution is assumed.

    Parameters
    ----------
    x: torch.Tensor
        Input data
    order: str, optional.
        Default is ``'C'`` as in C-like (row-major) memory layout. The array is stored first dimension first (rows first if ``ndim=2``).
        Alternative is ``'F'``, as in Fortran-like (column-major) memory layout. The array is stored last dimension first (columns first if ``ndim=2``).
