Module heat.utils.data.matrixgallery
====================================
Generate matrices for specific tests and functions

Functions
---------

`hermitian(n: int, dtype: Type[heat.core.types.datatype] = heat.core.types.complex64, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, positive_definite: bool = False) ‑> heat.core.dndarray.DNDarray`
:   Generates a random Hermitian matrix of size `(n,n)`. A Hermitian matrix is a complex square matrix that is equal to its conjugate transpose; for real data-types this routine
    returns a random symmetric matrix of size `(n,n)`.

    If `positive_definite=True`, the output is given by :math:`\frac{1}{n} R R^H` with :math:`R\in\mathbb{K}^{n\times n}` having entries distributed according to the standard normal distribution.
    This corresponds to sampling a random matrix according to the so-called Wishart distribution; see, e.g., [2], and also [3] for additional information regarding the asymptotic distribution of
    the singular values. The output matrix will be positive definite with probability 1.

    If `positive_definite=False`, the output is :math:`R+R^H` with :math:`R` generated as above.

    Parameters
    ----------
    n : int
        size of the resulting square matrix
    dtype: Type[datatype], optional
        The desired data-type for the array, defaults to ht.complex64; only floating-point data-types allowed.
        For real data-types, i.e. float32 and float64, a matrix with real entries (i.e. a symmetric one) is returned.
    split: None or int, optional
        The axis along which the array content is split and distributed in memory.
    device: None or str or Device, optional
        Specifies the device the tensor shall be allocated on, defaults globally set default device.
    comm : Communication, optional
        Handle to the nodes holding distributed parts or copies of this array.
    positive_definite : bool, optional
        If True, the resulting matrix is positive definite, defaults to False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Hermitian_matrix
    [2] https://en.wikipedia.org/wiki/Wishart_distribution
    [3] https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution

`parter(n: int, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, dtype: Type[heat.core.types.datatype] = heat.core.types.float32) ‑> heat.core.dndarray.DNDarray`
:   Generates the Parter matrix, a Toeplitz matrix that has the interesting property of having its singular values cluster at
    pi. The matrix has been named so by Cleve Moler in recognition of Seymour Parter's proof of this fact.

    Parameters
    ----------
    n : int
        size of the resulting square matrix
    split: None or int, optional
        The axis along which the array content is split and distributed in memory.
    device: None or str or Device, optional
        Specifies the device the tensor shall be allocated on, defaults globally set default device.
    comm: None or Communication, optional
        Handle to the nodes holding distributed tensor chunks.
    dtype: Type[datatype], optional
        The desired data-type for the array, defaults to ht.float64.

    References
    ----------
    [1] https://blogs.mathworks.com/cleve/2019/06/24/bohemian-matrices-in-the-matlab-gallery/

    [2] https://blogs.mathworks.com/cleve/2014/02/03/surprising-svd-square-waves-and-pi/

    [3] Seymour V. Parter, On the distribution of the singular values of Toeplitz matrices, Linear Algebra and its
    Applications 80, 1986, 115-130, http://www.sciencedirect.com/science/article/pii/0024379586902806

`random_known_rank(m: int, n: int, r: int, quantile_function: Callable = <function <lambda>>, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, dtype: Type[heat.core.types.datatype] = heat.core.types.float32) ‑> Tuple[heat.core.dndarray.DNDarray, Tuple[heat.core.dndarray.DNDarray]]`
:   Creates a random m x n matrix with rank r.
    This routine uses :func:`random_known_singularvalues` with r singular values randomly chosen
    w.r.t. the distribution with quantile function given by the input quantile_function. Default yields exponential distibution with parameter lambda=1.
    Unlike in :func:`random_known_singularvalues`, here the singular values of the output are sorted in descending order.

`random_known_singularvalues(m: int, n: int, singular_values: heat.core.dndarray.DNDarray, split: int | None = None, device: str | heat.core.devices.Device | None = None, comm: heat.core.communication.Communication | None = None, dtype: Type[heat.core.types.datatype] = heat.core.types.float32) ‑> Tuple[heat.core.dndarray.DNDarray, Tuple[heat.core.dndarray.DNDarray]]`
:   Creates an m x n matrix with singular values given by the entries of the input array singular_values.
    Caveat: if the entries of `singular_values` are not sorted, the singular value decomposition of A (returned as second output) is so as well.
    The singular vectors are chosen randomly using :func:`random_orthogonal`.
