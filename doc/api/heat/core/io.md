Module heat.core.io
===================
Enables parallel I/O with data on disk.

Functions
---------

`load(path: str, *args: Optional[List[object]], **kwargs: Optional[Dict[str, object]]) ‑> heat.core.dndarray.DNDarray`
:   Attempts to load data from a file stored on disk. Attempts to auto-detect the file format by determining the
    extension. Supports at least CSV files, HDF5 and netCDF4 are additionally possible if the corresponding libraries
    are installed.

    Parameters
    ----------
    path : str
        Path to the file to be read.
    args : list, optional
        Additional options passed to the particular functions.
    kwargs : dict, optional
        Additional options passed to the particular functions.

    Raises
    ------
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    Examples
    --------
    >>> ht.load("data.h5", dataset="DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("data.nc", variable="DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=None)
    >>> ht.load("my_data.zarr", variable="RECEIVER_1/DATA")
    DNDarray([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981], dtype=ht.float32, device=cpu:0, split=0)
    >>> ht.load("my_data.zarr", variable="RECEIVER_*/DATA")
    DNDarray([[ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981],
                [ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981],
                [ 1.0000,  2.7183,  7.3891, 20.0855, 54.5981]], dtype=ht.float32, device=cpu:0, split=0)

    See Also
    --------
    :func:`load_csv` : Loads data from a CSV file.
    :func:`load_csv_from_folder` : Loads multiple .csv files into one DNDarray which will be returned.
    :func:`load_hdf5` : Loads data from an HDF5 file.
    :func:`load_netcdf` : Loads data from a NetCDF4 file.
    :func:`load_npy_from_path` : Loads multiple .npy files into one DNDarray which will be returned.
    :func:`load_zarr` : Loads zarr-Format into DNDarray which will be returned.

`load_csv(path: str, header_lines: int = 0, sep: str = ',', dtype: datatype = heat.core.types.float32, encoding: str = 'utf-8', split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Loads data from a CSV file. The data will be distributed along the axis 0.

    Parameters
    ----------
    path : str
        Path to the CSV file to be read.
    header_lines : int, optional
        The number of columns at the beginning of the file that should not be considered as data.
    sep : str, optional
        The single ``char`` or ``str`` that separates the values in each row.
    dtype : datatype, optional
        Data type of the resulting array.
    encoding : str, optional
        The type of encoding which will be used to interpret the lines of the csv file as strings.
    split : int or None : optional
        Along which axis the resulting array should be split.
        Default is ``None`` which means each node will have the full array.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, defaults to global default

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.

    Examples
    --------
    >>> import heat as ht
    >>> a = ht.load_csv("data.csv")
    >>> a.shape
    [0/3] (150, 4)
    [1/3] (150, 4)
    [2/3] (150, 4)
    [3/3] (150, 4)
    >>> a.lshape
    [0/3] (38, 4)
    [1/3] (38, 4)
    [2/3] (37, 4)
    [3/3] (37, 4)
    >>> b = ht.load_csv("data.csv", header_lines=10)
    >>> b.shape
    [0/3] (140, 4)
    [1/3] (140, 4)
    [2/3] (140, 4)
    [3/3] (140, 4)
    >>> b.lshape
    [0/3] (35, 4)
    [1/3] (35, 4)
    [2/3] (35, 4)
    [3/3] (35, 4)

`load_hdf5(path: str, dataset: str, dtype: datatype = heat.core.types.float32, slices: Optional[Tuple[Optional[slice], ...]] = None, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Loads data from an HDF5 file. The data may be distributed among multiple processing nodes via the split flag.

    Parameters
    ----------
    path : str
        Path to the HDF5 file to be read.
    dataset : str
        Name of the dataset to be read.
    dtype : datatype, optional
        Data type of the resulting array.
    slices : tuple of slice objects, optional
        Load only the specified slices of the dataset.
    split : int or None, optional
        The axis along which the data is distributed among the processing cores.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type

    Examples
    --------
    >>> a = ht.load_hdf5("data.h5", dataset="DATA")
    >>> a.shape
    [0/2] (5,)
    [1/2] (5,)
    >>> a.lshape
    [0/2] (5,)
    [1/2] (5,)
    >>> b = ht.load_hdf5("data.h5", dataset="DATA", split=0)
    >>> b.shape
    [0/2] (5,)
    [1/2] (5,)
    >>> b.lshape
    [0/2] (3,)
    [1/2] (2,)

    Using the slicing argument:
    >>> not_sliced = ht.load_hdf5("other_data.h5", dataset="DATA", split=0)
    >>> not_sliced.shape
    [0/2] (10,2)
    [1/2] (10,2)
    >>> not_sliced.lshape
    [0/2] (5,2)
    [1/2] (5,2)
    >>> not_sliced.larray
    [0/2] [[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 8,  9]]
    [1/2] [[10, 11],
           [12, 13],
           [14, 15],
           [16, 17],
           [18, 19]]

    >>> sliced = ht.load_hdf5("other_data.h5", dataset="DATA", split=0, slices=slice(8))
    >>> sliced.shape
    [0/2] (8,2)
    [1/2] (8,2)
    >>> sliced.lshape
    [0/2] (4,2)
    [1/2] (4,2)
    >>> sliced.larray
    [0/2] [[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7]]
    [1/2] [[ 8,  9],
           [10, 11],
           [12, 13],
           [14, 15],
           [16, 17]]

    >>> sliced = ht.load_hdf5('other_data.h5', dataset='DATA', split=0, slices=(slice(2,8), slice(0,1))
    >>> sliced.shape
    [0/2] (6,1)
    [1/2] (6,1)
    >>> sliced.lshape
    [0/2] (3,1)
    [1/2] (3,1)
    >>> sliced.larray
    [0/2] [[ 4, ],
           [ 6, ],
           [ 8, ]]
    [1/2] [[10, ],
           [12, ],
           [14, ]]

`load_netcdf(path: str, variable: str, dtype: datatype = heat.core.types.float32, split: Optional[int] = None, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Loads data from a NetCDF4 file. The data may be distributed among multiple processing nodes via the split flag.

    Parameters
    ----------
    path : str
        Path to the NetCDF4 file to be read.
    variable : str
        Name of the variable to be read.
    dtype : datatype, optional
        Data type of the resulting array
    split : int or None, optional
        The axis along which the data is distributed among the processing cores.
    comm : Communication, optional
        The communication to use for the data distribution. Defaults to MPI_COMM_WORLD.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.

    Examples
    --------
    >>> a = ht.load_netcdf("data.nc", variable="DATA")
    >>> a.shape
    [0/2] (5,)
    [1/2] (5,)
    >>> a.lshape
    [0/2] (5,)
    [1/2] (5,)
    >>> b = ht.load_netcdf("data.nc", variable="DATA", split=0)
    >>> b.shape
    [0/2] (5,)
    [1/2] (5,)
    >>> b.lshape
    [0/2] (3,)
    [1/2] (2,)

`load_npy_from_path(path: str, dtype: datatype = heat.core.types.int32, split: int = 0, device: Optional[str] = None, comm: Optional[Communication] = None) ‑> heat.core.dndarray.DNDarray`
:   Loads multiple .npy files into one DNDarray which will be returned. The data will be concatenated along the split axis provided as input.

    Parameters
    ----------
    path : str
        Path to the directory in which .npy-files are located.
    dtype : datatype, optional
        Data type of the resulting array.
    split : int
        Along which axis the loaded arrays should be concatenated.
    device : str, optional
        The device id on which to place the data, defaults to globally set default device.
    comm : Communication, optional
        The communication to use for the data distribution, default is 'heat.MPI_WORLD'

`save(data: DNDarray, path: str, *args: Optional[List[object]], **kwargs: Optional[Dict[str, object]])`
:   Attempts to save data from a :class:`~heat.core.dndarray.DNDarray` to disk. An auto-detection based on the file
    format extension is performed.

    Parameters
    ----------
    data : DNDarray
        The array holding the data to be stored
    path : str
        Path to the file to be stored.
    args : list, optional
        Additional options passed to the particular functions.
    kwargs : dict, optional
        Additional options passed to the particular functions.

    Raises
    ------
    ValueError
        If the file extension is not understood or known.
    RuntimeError
        If the optional dependency for a file extension is not available.

    Examples
    --------
    >>> x = ht.arange(100, split=0)
    >>> ht.save(x, "data.h5", "DATA", mode="a")

`save_csv(data: DNDarray, path: str, header_lines: Iterable[str] = None, sep: str = ',', decimals: int = -1, encoding: str = 'utf-8', comm: Optional[Communication] = None, truncate: bool = True)`
:   Saves data to CSV files. Only 2D data, all split axes.

    Parameters
    ----------
    data : DNDarray
        The DNDarray to be saved to CSV.
    path : str
        The path as a string.
    header_lines : Iterable[str]
        Optional iterable of str to prepend at the beginning of the file. No
        pound sign or any other comment marker will be inserted.
    sep : str
        The separator character used in this CSV.
    decimals: int
        Number of digits after decimal point.
    encoding : str
        The encoding to be used in this CSV.
    comm : Optional[Communication]
        An optional object of type Communication to be used.
    truncate : bool
        Whether to truncate an existing file before writing, i.e. fully overwrite it.
        The sane default is True. Setting it to False will not shorten files if
        needed and thus may leave garbage at the end of existing files.

`save_hdf5(data: DNDarray, path: str, dataset: str, mode: str = 'w', **kwargs: Dict[str, object])`
:   Saves ``data`` to an HDF5 file. Attempts to utilize parallel I/O if possible.

    Parameters
    ----------
    data : DNDarray
        The data to be saved on disk.
    path : str
        Path to the HDF5 file to be written.
    dataset : str
        Name of the dataset the data is saved to.
    mode : str, optional
        File access mode, one of ``'w', 'a', 'r+'``
    kwargs : dict, optional
        Additional arguments passed to the created dataset.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.
    ValueError
        If the access mode is not understood.

    Examples
    --------
    >>> x = ht.arange(100, split=0)
    >>> ht.save_hdf5(x, "data.h5", dataset="DATA")

`save_netcdf(data: DNDarray, path: str, variable: str, mode: str = 'w', dimension_names: Union[list, tuple, str] = None, is_unlimited: bool = False, file_slices: Union[Iterable[int], slice, bool] = slice(None, None, None), **kwargs: Dict[str, object])`
:   Saves data to a netCDF4 file. Attempts to utilize parallel I/O if possible.

    Parameters
    ----------
    data : DNDarray
        The data to be saved on disk.
    path : str
        Path to the netCDF4 file to be written.
    variable : str
        Name of the variable the data is saved to.
    mode : str, optional
        File access mode, one of ``'w', 'a', 'r+'``.
    dimension_names : list or tuple or string
        Specifies the netCDF Dimensions used by the variable. Ignored if Variable already exists.
    is_unlimited : bool, optional
        If True, every dimension created for this variable (i.e. doesn't already exist) is unlimited. Already
        existing limited dimensions cannot be changed to unlimited and vice versa.
    file_slices : integer iterable, slice, ellipsis or bool
        Keys used to slice the netCDF Variable, as given in the nc.utils._StartCountStride method.
    kwargs : dict, optional
        additional arguments passed to the created dataset.

    Raises
    ------
    TypeError
        If any of the input parameters are not of correct type.
    ValueError
        If the access mode is not understood or if the number of dimension names does not match the number of
        dimensions.

    Examples
    --------
    >>> x = ht.arange(100, split=0)
    >>> ht.save_netcdf(x, "data.nc", dataset="DATA")

`supports_hdf5() ‑> bool`
:   Returns ``True`` if Heat supports reading from and writing to HDF5 files, ``False`` otherwise.

`supports_netcdf() ‑> bool`
:   Returns ``True`` if Heat supports reading from and writing to netCDF4 files, ``False`` otherwise.

`supports_zarr() ‑> bool`
:   Returns ``True`` if zarr is installed, ``False`` otherwise.
