Module heat.fft.fft
===================
Provides a collection of Discrete Fast Fourier Transforms (DFFT) and their inverses.

Functions
---------

`fft(x: heat.core.dndarray.DNDarray, n: int = None, axis: int = -1, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the one-dimensional discrete Fourier Transform over the specified axis in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, the last axis is transformed, while the remaining
    axes are left unchanged.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex. WARNING: If x is 1-D and distributed, the entire array is copied on each MPI process. See Notes.
    n : int, optional
        Length of the transformed axis of the output. If not given, the length is assumed to be the length of the input
        along the axis specified by `axis`. If `n` is smaller than the length of the input, the input is truncated. If `n` is
        larger, the input is padded with zeros. Default: None.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used, or the only axis if `x` has only one
        dimension. Default: -1.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`ifft` : inverse 1-dimensional FFT
    :func:`fft2` : 2-dimensional FFT
    :func:`fftn` : N-dimensional FFT
    :func:`rfft` : 1-dimensional FFT of a real signal
    :func:`hfft` : 1-dimensional FFT of a Hermitian symmetric sequence
    :func:`fftfreq` : frequency bins for given FFT parameters
    :func:`rfftfreq` : frequency bins for real FFT

    Notes
    -----
    This function requires MPI communication if the input array is transformed along the distribution axis.
    If the input array is 1-D and distributed, this function copies the entire array on each MPI process! i.e. if the array is very large, you might run out of memory.
    Hint: if you are looping through a batch of 1-D arrays to transform them, consider stacking them into a 2-D DNDarray and transforming them in one go (see :func:`fft2`).

`fft2(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the 2-dimensional discrete Fourier Transform over the specified axes in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, the last two axes are transformed, while the
    remaining axes are left unchanged.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, int], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, int], optional
        Axes over which to compute the FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is also
        not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet.
        (default is (-2, -1))
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`ifft2` : inverse 2-dimensional FFT
    :func:`fft` : 1-dimensional FFT
    :func:`fftn` : N-dimensional FFT
    :func:`rfft2` : 2-dimensional FFT of a real signal
    :func:`hfft2` : 2-dimensional FFT of a Hermitian symmetric sequence

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`fftfreq(n: int, d: int | float = 1.0, dtype: Type | None = None, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: mpi4py.MPI.Comm | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return the Discrete Fourier Transform sample frequencies for a signal of size ``n``.

    The returned ``DNDarray`` contains the frequency bin centers in cycles per unit of the sample spacing (with zero
    at the start). For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.

    Parameters
    ----------
    n : int
        Window length.
    d : Union[int, float], optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    dtype : Type, optional
        The desired data type of the output. Defaults to `ht.float32`.
    split : int, optional
        The axis along which to split the result. Can be None or 0, as the output is 1-dimensional. Defaults to None, i.e. non-distributed output.
    device : str or Device, optional
        The device on which to place the output. If not given, the output is placed on the current device.
    comm : MPI.Comm, optional
        The MPI communicator to use for distributing the output. If not given, the default communicator is used.

    See Also
    --------
    :func:`rfftfreq` : frequency bins for :func:`rfft`

`fftn(x: heat.core.dndarray.DNDarray, s: Tuple[int, ...] = None, axes: Tuple[int, ...] = None, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the N-dimensional discrete Fourier Transform.

    This function computes the N-dimensional discrete Fourier Transform over any number of axes in an M-dimensional
    array by means of the Fast Fourier Transform (FFT).

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, ...], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, ...], optional
        Axes over which to compute the FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is also
        not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet.
        (default is None)
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`ifftn` : inverse N-dimensional FFT
    :func:`fft` : 1-dimensional FFT
    :func:`fft2` : 2-dimensional FFT
    :func:`rfftn` : N-dimensional FFT of a real signal
    :func:`hfftn` : N-dimensional FFT of a Hermitian symmetric sequence

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`fftshift(x: heat.core.dndarray.DNDarray, axes: int | Iterable[int] | None = None) ‑> heat.core.dndarray.DNDarray`
:   Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all). Note that ``y[0]`` is the Nyquist component
    only if ``len(x)`` is even.

    Parameters
    ----------
    x : DNDarray
        Input array
    axes : int or Iterable[int], optional
        Axes over which to shift. Default is None, which shifts all axes.

    See Also
    --------
    :func:`ifftshift` : The inverse of `fftshift`.

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is shifted.

`hfft(x: heat.core.dndarray.DNDarray, n: int = None, axis: int = -1, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the one-dimensional discrete Fourier Transform of a Hermitian symmetric signal.

    This function computes the one-dimensional discrete Fourier Transform over the specified axis in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, the last axis is transformed, while the remaining
    axes are left unchanged. The input signal is assumed to be Hermitian-symmetric, i.e. `x[..., i] = x[..., -i].conj()`.

    Parameters
    ----------
    x : DNDarray
        Input array
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is not None, the input array is either zero-padded or trimmed to length `n` before the transform.
        Default: `2 * (x.shape[axis] - 1)`.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used, or the only axis if x has only one
        dimension. Default: -1.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`ihfft` : inverse 1-dimensional FFT of a Hermitian-symmetric sequence
    :func:`hfft2` : 2-dimensional FFT of a Hermitian-symmetric sequence
    :func:`hfftn` : N-dimensional FFT of a Hermitian-symmetric sequence
    :func:`fft` : 1-dimensional FFT
    :func:`rfft` : 1-dimensional FFT of a real signal

    Notes
    -----
    This function requires MPI communication if the input array is transformed along the distribution axis.

`hfft2(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the 2-dimensional discrete Fourier Transform of a Hermitian symmetric signal.

    This function computes the 2-dimensional discrete Fourier Transform over the specified axes in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, the last two axes are transformed, while the
    remaining axes are left unchanged. The input signal is assumed to be Hermitian-symmetric, i.e. `x[..., i] = x[..., -i].conj()`.

    Parameters
    ----------
    x : DNDarray
        Input array
    s : Tuple[int, int], optional
        Shape of the signal along the transformed axes. If `s` is specified, the input array is either zero-padded or trimmed to length `s` before the transform.
        If `s` is not given, the last dimension defaults to even output: `s[-1] = 2 * (x.shape[-1] - 1)`.
    axes : Tuple[int, int], optional
        Axes over which to compute the FFT. If not given, the last two dimensions are transformed. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default: (-2, -1).
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`ihfft2` : inverse 2-dimensional FFT of a Hermitian-symmetric sequence
    :func:`hfft` : 1-dimensional FFT of a Hermitian-symmetric sequence
    :func:`hfftn` : N-dimensional FFT of a Hermitian-symmetric sequence
    :func:`fft2` : 2-dimensional FFT
    :func:`rfft2` : 2-dimensional FFT of a real signal

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`hfftn(x: heat.core.dndarray.DNDarray, s: Tuple[int, ...] = None, axes: Tuple[int, ...] = None, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the N-dimensional discrete Fourier Transform of a Hermitian symmetric signal.

    This function computes the N-dimensional discrete Fourier Transform over any number of axes in an M-dimensional
    array by means of the Fast Fourier Transform (FFT). By default, all axes are transformed.

    Parameters
    ----------
    x : DNDarray
        Input array
    s : Tuple[int, ...], optional
        Shape of the signal along the transformed axes. If `s` is specified, the input array is either zero-padded or trimmed to length `s` before the transform.
        If `s` is not given, the last dimension defaults to even output: `s[-1] = 2 * (x.shape[-1] - 1)`.
    axes : Tuple[int, ...], optional
        Axes over which to compute the FFT. If not given, all dimensions are transformed. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default: None.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`ihfftn` : inverse N-dimensional FFT of a Hermitian-symmetric sequence
    :func:`hfft` : 1-dimensional FFT of a Hermitian-symmetric sequence
    :func:`hfft2` : 2-dimensional FFT of a Hermitian-symmetric sequence
    :func:`fftn` : N-dimensional FFT
    :func:`rfftn` : N-dimensional FFT of a real signal

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`ifft(x: heat.core.dndarray.DNDarray, n: int = None, axis: int = -1, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the one-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    n : int, optional
        Length of the transformed axis of the output. If not given, the length is taken to be the length of the input
        along the axis specified by `axis`. If `n` is smaller than the length of the input, the input is cropped. If `n` is
        larger, the input is padded with zeros. Default: None.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last axis is used, or the only axis if x has only one dimension. Default: -1.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`fft` : forward 1-dimensional FFT
    :func:`ifft2` : inverse 2-dimensional FFT
    :func:`ifftn` : inverse N-dimensional FFT
    :func:`irfft` : inverse 1-dimensional FFT of a real sequence
    :func:`ihfft` : inverse 1-dimensional FFT of a Hermitian symmetric sequence

    Notes
    -----
    This function requires MPI communication if the input array is transformed along the distribution axis.
    If the input array is 1-D and distributed, this function copies the entire array on each MPI process! i.e. if the array is very large, you might run out of memory.
    Hint: if you are looping through a batch of 1-D arrays to transform them, consider stacking them into a 2-D DNDarray and transforming them all at once (see :func:`ifft2`).

`ifft2(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the 2-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, int], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, int], optional
        Axes over which to compute the inverse FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default: (-2, -1).
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`fft2` : forward 2-dimensional FFT
    :func:`ifft` : inverse 1-dimensional FFT
    :func:`ifftn` : inverse N-dimensional FFT
    :func:`irfft2` : inverse 2-dimensional FFT of a real sequence
    :func:`ihfft2` : inverse 2-dimensional FFT of a Hermitian symmetric sequence

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`ifftn(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, ...] = None, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the N-dimensional inverse discrete Fourier Transform.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, ...], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, ...], optional
        Axes over which to compute the inverse FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default: None.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`fftn` : forward N-dimensional FFT
    :func:`ifft` : inverse 1-dimensional FFT
    :func:`ifft2` : inverse 2-dimensional FFT
    :func:`irfftn` : inverse N-dimensional FFT of a real sequence
    :func:`ihfftn` : inverse N-dimensional FFT of a Hermitian symmetric sequence

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`ifftshift(x: heat.core.dndarray.DNDarray, axes: int | Iterable[int] | None = None) ‑> heat.core.dndarray.DNDarray`
:   The inverse of fftshift.

    Parameters
    ----------
    x : DNDarray
        Input array
    axes : int or Iterable[int], optional
        Axes over which to shift. Default is None, which shifts all axes.

    See Also
    --------
    :func:`fftshift` : Shift the zero-frequency component to the center of the spectrum.

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is shifted.

`ihfft(x: heat.core.dndarray.DNDarray, n: int = None, axis: int = -1, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the one-dimensional inverse discrete Fourier Transform of a real signal. The output is Hermitian-symmetric.

    Parameters
    ----------
    x : DNDarray
        Input array, must be real
    n : int, optional
        Length of the transformed axis of the output. If not given, the length is taken to be the length of the input
        along the axis specified by `axis`. If `n` is smaller than the length of the input, the input is cropped. If `n` is
        larger, the input is padded with zeros. Default: None.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last axis is used, or the only axis if x has only one dimension. Default: -1.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`hfft` : 1-dimensional FFT of a Hermitian-symmetric sequence
    :func:`ihfft2` : inverse 2-dimensional FFT of a Hermitian-symmetric sequence
    :func:`ihfftn` : inverse N-dimensional FFT of a Hermitian-symmetric sequence
    :func:`rfft` : 1-dimensional FFT of a real signal
    :func:`irfft` : inverse 1-dimensional FFT of a real sequence

    Notes
    -----
    This function requires MPI communication if the input array is transformed along the distribution axis.

`ihfft2(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse of a 2-dimensional discrete Fourier Transform of a Hermitian-symmetric signal. The output is Hermitian-symmetric. Requires torch >= 1.11.0.

    Parameters
    ----------
    x : DNDarray
        Input array, must be real
    s : Tuple[int, int], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, int], optional
        Axes over which to compute the inverse FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default is (-2, -1).
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`hfft2` : 2-dimensional FFT of a Hermitian-symmetric sequence
    :func:`ihfft` : inverse 1-dimensional FFT of a Hermitian-symmetric sequence
    :func:`ihfftn` : inverse N-dimensional FFT of a Hermitian-symmetric sequence
    :func:`rfft2` : 2-dimensional FFT of a real signal
    :func:`irfft2` : inverse 2-dimensional FFT of a real sequence

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`ihfftn(x: heat.core.dndarray.DNDarray, s: Tuple[int, ...] = None, axes: Tuple[int, ...] = None, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse of a N-dimensional discrete Fourier Transform of Hermitian-symmetric signal. The output is Hermitian-symmetric. Requires torch >= 1.11.0.

    Parameters
    ----------
    x : DNDarray
        Input array, must be real
    s : Tuple[int, ...], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, ...], optional
        Axes over which to compute the inverse FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default: None.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`hfftn` : N-dimensional FFT of a Hermitian-symmetric sequence
    :func:`ihfft` : inverse 1-dimensional FFT of a Hermitian-symmetric sequence
    :func:`ihfft2` : inverse 2-dimensional FFT of a Hermitian-symmetric sequence
    :func:`rfftn` : N-dimensional FFT of a real signal
    :func:`irfftn` : inverse N-dimensional FFT of a real sequence

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`irfft(x: heat.core.dndarray.DNDarray, n: int = None, axis: int = -1, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse of a one-dimensional discrete Fourier Transform of real signal. The output is real.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    n : int, optional
        Length of the transformed axis of the output. If not given, the length is taken to be the length of the input
        along the axis specified by `axis`. If `n` is smaller than the length of the input, the input is cropped. If `n` is
        larger, the input is padded with zeros. Default: None.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last axis is used, or the only axis if x has only one dimension. Default: -1.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`irfft2` : inverse 2-dimensional FFT
    :func:`irfftn` : inverse N-dimensional FFT
    :func:`rfft` : 1-dimensional FFT of a real signal
    :func:`hfft` : 1-dimensional FFT of a Hermitian symmetric sequence
    :func:`fft` : 1-dimensional FFT

    Notes
    -----
    This function requires MPI communication if the input array is transformed along the distribution axis.
    If the input array is 1-D and distributed, this function copies the entire array on each MPI process! i.e. if the array is very large, you might run out of memory.
    Hint: if you are looping through a batch of 1-D arrays to transform them, consider stacking them into a 2-D DNDarray and transforming them all at once (see :func:`irfft2`).

`irfft2(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse of a 2-dimensional discrete real Fourier Transform. The output is real.

    Parameters
    ----------
    x : DNDarray
        Input array, can be complex
    s : Tuple[int, int], optional
        Shape of the output along the transformed axes.
    axes : Tuple[int, int], optional
        Axes over which to compute the inverse FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. Default is (-2, -1))
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    See Also
    --------
    :func:`irfft` : inverse 1-dimensional FFT
    :func:`irfftn` : inverse N-dimensional FFT
    :func:`rfft2` : 2-dimensional FFT of a real signal
    :func:`hfft2` : 2-dimensional FFT of a Hermitian symmetric sequence
    :func:`fft2` : 2-dimensional FFT

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`irfftn(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, ...] = None, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the inverse of an N-dimensional discrete Fourier Transform of real signal.
    The output is real.

    Parameters
    ----------
    x : DNDarray
        Input array, assumed to be Hermitian-symmetric along the transformed axes, with the last transformed axis only containing the positive half of the frequencies.
    s : Tuple[int, ...], optional
        Shape of the output along the transformed axes. If ``s`` is not specified, the last transposed axis is reconstructued in full, i.e. `s[-1] = 2 * (x.shape[axes[-1]] - 1)`.
    axes : Tuple[int, ...], optional
        Axes over which to compute the inverse FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet.
        (default is None)
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`rfft(x: heat.core.dndarray.DNDarray, n: int = None, axis: int = -1, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the one-dimensional discrete Fourier Transform of real input. The output is Hermitian-symmetric.

    Parameters
    ----------
    x : DNDarray
        Input array, must be real.
    n : int, optional
        Length of the transformed axis of the output. If not given, the length is taken to be the length of the input
        along the axis specified by `axis`. If `n` is smaller than the length of the input, the input is cropped. If `n` is
        larger, the input is padded with zeros. Default: None.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is used, or the only axis if x has only one dimension. Default: -1.
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    Notes
    -----
    This function requires MPI communication if the input array is transformed along the distribution axis.
    If the input array is 1-D and distributed, this function copies the entire array on each MPI process! i.e. if the array is very large, you might run out of memory.
    Hint: if you are looping through a batch of 1-D arrays to transform them, consider stacking them into a 2-D DNDarray and transforming them all at once (see :func:`rfft2`).

`rfft2(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, int] = (-2, -1), norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the 2-dimensional discrete Fourier Transform of real input. The output is Hermitian-symmetric.

    Parameters
    ----------
    x : DNDarray
        Input array, must be real.
    s : Tuple[int, int], optional
        Shape of the output along the transformed axes. (default is x.shape)
    axes : Tuple[int, int], optional
        Axes over which to compute the FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. (default is (-2, -1))
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.

`rfftfreq(n: int, d: int | float = 1.0, dtype: Type | None = None, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: mpi4py.MPI.Comm | None = None) ‑> heat.core.dndarray.DNDarray`
:   Return the Discrete Fourier Transform sample frequencies.

    The returned float DNDarray contains the frequency bin centers in cycles per unit of the sample spacing (with zero
    at the start). For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.

    Parameters
    ----------
    n : int
        Window length.
    d : Union[int, float], optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.
    dtype : Type, optional
        The desired data type of the output. Defaults to `float32`.
    split : int, optional
        The axis along which to split the result. If not given, the result is not split.
    device : str or Device, optional
        The device on which to place the output. If not given, the output is placed on the current device.
    comm : MPI.Comm, optional
        The MPI communicator to use for distributing the output. If not given, the default communicator is used.

`rfftn(x: heat.core.dndarray.DNDarray, s: Tuple[int, int] = None, axes: Tuple[int, ...] = None, norm: str = None) ‑> heat.core.dndarray.DNDarray`
:   Compute the N-dimensional discrete Fourier Transform of real input. By default, all axes are transformed, with the real transform
    performed over the last axis, while the remaining transforms are complex. The output is Hermitian-symmetric, with the last transformed axis having length `s[-1] // 2 + 1` (the positive part of the spectrum).

    Parameters
    ----------
    x : DNDarray
        Input array, must be real.
    s : Tuple[int, ...], optional
        Shape of the output along the transformed axes.
    axes : Tuple[int, ...], optional
        Axes over which to compute the FFT. If not given, the last `len(s)` axes are used, or all axes if `s` is
        also not specified. Repeated transforms over an axis, i.e. repeated indices in ``axes``, are not supported yet. (default is None)
    norm : str, optional
        Normalization mode: 'forward', 'backward', or 'ortho'. Indicates in what direction the forward/backward pair of transforms is normalized. Default is "backward".

    Notes
    -----
    This function requires MPI communication if the input array is distributed and the split axis is transformed.
