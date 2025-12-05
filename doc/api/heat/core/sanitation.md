Module heat.core.sanitation
===========================
Collection of validation/sanitation routines.

Functions
---------

`sanitize_distribution(*args: DNDarray, target: DNDarray, diff_map: torch.Tensor = None) ‑> Union[DNDarray, Tuple(DNDarray)]`
:   Distribute every `arg` according to `target.lshape_map` or, if provided, `diff_map`.
    After this sanitation, the lshapes are compatible along the split dimension.
    `Args` can contain non-distributed DNDarrays, they will be split afterwards, if `target` is split.

    Parameters
    ----------
    args : DNDarray
        Dndarrays to be distributed

    target : DNDarray
        Dndarray used to sanitize the metadata and to, if diff_map is not given, determine the resulting distribution.

    diff_map : torch.Tensor (optional)
        Different lshape_map. Overwrites the distribution of the target array.
        Used in cases when the target array does not correspond to the actually wanted distribution,
        e.g. because it only contains a single element along the split axis and gets broadcast.

    Raises
    ------
    TypeError
        When an argument is not a ``DNDarray`` or ``None``.
    ValueError
        When the split-axes or sizes along the split-axis do not match.

    See Also
    --------
    :func:`~heat.core.dndarray.create_lshape_map`
        Function to create the lshape_map.

`sanitize_in(x: Any)`
:   Verify that input object is ``DNDarray``.

    Parameters
    ----------
    x : Any
        Input object

    Raises
    ------
    TypeError
        When ``x`` is not a ``DNDarray``.

`sanitize_in_tensor(x: Any)`
:   Verify that input object is ``torch.Tensor``.

    Parameters
    ----------
    x : Any
        Input object.

    Raises
    ------
    TypeError
        When ``x`` is not a ``torch.Tensor``.

`sanitize_infinity(x: Union[DNDarray, torch.Tensor]) ‑> int | float`
:   Returns largest possible value for the ``dtype`` of the input array.

    Parameters
    ----------
    x: Union[DNDarray, torch.Tensor]
        Input object.

`sanitize_lshape(array: DNDarray, tensor: torch.Tensor)`
:   Verify shape consistency when manipulating process-local arrays.

    Parameters
    ----------
    array : DNDarray
        the original, potentially distributed ``DNDarray``
    tensor : torch.Tensor
        process-local data meant to replace ``array.larray``

    Raises
    ------
    ValueError
        if shape of local ``torch.Tensor`` is inconsistent with global ``DNDarray``.

`sanitize_out(out: DNDarray, output_shape: Tuple, output_split: int, output_device: str, output_comm: Communication = None)`
:   Validate output buffer ``out``.

    Parameters
    ----------
    out : DNDarray
          the `out` buffer where the result of some operation will be stored

    output_shape : Tuple
                   the calculated shape returned by the operation

    output_split : Int
                   the calculated split axis returned by the operation

    output_device : Str
                    "cpu" or "gpu" as per location of data

    output_comm : Communication
                    Communication object of the result of the operation

    Raises
    ------
    TypeError
        if ``out`` is not a ``DNDarray``.
    ValueError
        if shape, split direction, or device of the output buffer ``out`` do not match the operation result.

`sanitize_sequence(seq: Union[Sequence[int, ...], Sequence[float, ...], DNDarray, torch.Tensor]) ‑> List`
:   Check if sequence is valid, return list.

    Parameters
    ----------
    seq : Union[Sequence[int, ...], Sequence[float, ...], DNDarray, torch.Tensor]
        Input sequence.

    Raises
    ------
    TypeError
        if ``seq`` is neither a list nor a tuple

`scalar_to_1d(x: DNDarray) ‑> heat.core.dndarray.DNDarray`
:   Turn a scalar ``DNDarray`` into a 1-D ``DNDarray`` with 1 element.

    Parameters
    ----------
    x : DNDarray
        with `x.ndim = 0`
