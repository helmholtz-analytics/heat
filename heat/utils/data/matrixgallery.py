"""
Generate matrices for specific tests and functions
"""

from heat import core
from ...core.dndarray import DNDarray
from ...core.communication import Communication
from ...core.devices import Device
from ...core.types import datatype, heat_type_is_complexfloating, heat_type_is_exact
from ...core.random import randn, rand
from ...core.linalg import qr, matmul
from ...core.manipulations import diag, sort
from ...core.exponential import log
from typing import Type, Union, Tuple, Callable

__all__ = ["hermitian", "parter", "random_known_singularvalues", "random_known_rank"]


def hermitian(
    n: int,
    dtype: Type[datatype] = core.complex64,
    split: Union[None, int] = None,
    device: Union[None, str, Device] = None,
    comm: Union[None, Communication] = None,
    positive_definite: bool = False,
) -> DNDarray:
    r"""
    Generates a random Hermitian matrix of size `(n,n)`. A Hermitian matrix is a complex square matrix that is equal to its conjugate transpose; for real data-types this routine
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
    """
    if heat_type_is_complexfloating(dtype):
        real_dtype = core.float32 if dtype is core.complex64 else core.float64
        matrix = randn(n, n, dtype=real_dtype, split=split, device=device, comm=comm) + 1j * randn(
            n, n, dtype=real_dtype, split=split, device=device, comm=comm
        )
    elif not heat_type_is_exact(dtype):
        matrix = randn(n, n, dtype=dtype, split=split, device=device, comm=comm)
    else:
        raise ValueError("dtype must be floating-point data-type but is ", dtype, ".")
    if positive_definite:
        return 1 / n * matrix @ core.conj(matrix).T

    return matrix + core.conj(matrix).T.resplit_(split)


def parter(
    n: int,
    split: Union[None, int] = None,
    device: Union[None, str, Device] = None,
    comm: Union[None, Communication] = None,
    dtype: Type[datatype] = core.float32,
) -> DNDarray:
    """
    Generates the Parter matrix, a Toeplitz matrix that has the interesting property of having its singular values cluster at
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
    """
    if split is None:
        a = core.arange(n, dtype=dtype, device=device, comm=comm)
        II = a.expand_dims(0)
        JJ = a.expand_dims(1)
    elif split == 0:
        II = core.arange(n, dtype=dtype, device=device, comm=comm).expand_dims(0)
        JJ = core.arange(n, dtype=dtype, split=split, device=device, comm=comm).expand_dims(1)
    elif split == 1:
        II = core.arange(n, dtype=dtype, split=0, device=device, comm=comm).expand_dims(0)
        JJ = core.arange(n, dtype=dtype, device=device, comm=comm).expand_dims(1)
    else:
        raise ValueError(f"expected split value to be either {{None,0,1}}, but was {split}")

    return 1.0 / (II - JJ + 0.5)


def random_orthogonal(
    m: int,
    n: int,
    split: Union[None, int] = None,
    device: Union[None, str, Device] = None,
    comm: Union[None, Communication] = None,
    dtype: Type[datatype] = core.float32,
) -> DNDarray:
    """Auxiliary routine: creates a random mxn matrix with orthonormal columns
    Caveat: this is done by QR of mxn matrices with i.i.d. normal entries, so this does not produce the uniform distribution on the orthogonal matrices...
    """
    if n > m:
        raise RuntimeError("No orthogonal matrix of shape %d x %d possible." % (m, n))

    # TODO: if QR does not make problems anymore, replace split=None by split=split
    U = randn(m, n, split=None, dtype=dtype, comm=comm, device=device)
    Q, _ = qr(U)

    return Q[:, :n].resplit_(split)


def random_known_singularvalues(
    m: int,
    n: int,
    singular_values: DNDarray,
    split: Union[None, int] = None,
    device: Union[None, str, Device] = None,
    comm: Union[None, Communication] = None,
    dtype: Type[datatype] = core.float32,
) -> Tuple[DNDarray, Tuple[DNDarray]]:
    """
    Creates an m x n matrix with singular values given by the entries of the input array singular_values.
    Caveat: if the entries of `singular_values` are not sorted, the singular value decomposition of A (returned as second output) is so as well.
    The singular vectors are chosen randomly using :func:`random_orthogonal`.
    """
    if not isinstance(singular_values, DNDarray):
        raise RuntimeError(
            f"Argument singular_values needs to be a DNDarray but is {type(singular_values)}."
        )
    if singular_values.ndim != 1:
        raise RuntimeError(
            f"Argument singular_values needs to be a 1D array, but dimension is {singular_values.ndim}."
        )
    if singular_values.shape[0] > min(m, n):
        raise RuntimeError(
            f"Number of given singular values must not exceed matrix dimensions. Got {singular_values.shape[0]} singular values for matrix size ({m}, {n})."
        )

    r = singular_values.shape[0]
    U = random_orthogonal(m, r, split=split, device=device, comm=comm, dtype=dtype)
    V = random_orthogonal(n, r, split=split, device=device, comm=comm, dtype=dtype)

    A = matmul(U, matmul(diag(singular_values), V.T))

    return A.resplit_(split), (U, singular_values, V)


def random_known_rank(
    m: int,
    n: int,
    r: int,
    quantile_function: Callable = lambda x: -log(x),
    split: Union[None, int] = None,
    device: Union[None, str, Device] = None,
    comm: Union[None, Communication] = None,
    dtype: Type[datatype] = core.float32,
) -> Tuple[DNDarray, Tuple[DNDarray]]:
    """
    Creates a random m x n matrix with rank r.
    This routine uses :func:`random_known_singularvalues` with r singular values randomly chosen
    w.r.t. the distribution with quantile function given by the input quantile_function. Default yields exponential distibution with parameter lambda=1.
    Unlike in :func:`random_known_singularvalues`, here the singular values of the output are sorted in descending order.
    """
    if r > min(m, n):
        raise RuntimeError("rank must not exceed matrix dimensions.")

    singular_values = rand(r, dtype=dtype, comm=comm, device=device)
    singular_values = sort(quantile_function(singular_values), descending=True)[0]

    return random_known_singularvalues(
        m, n, singular_values, split=split, device=device, comm=comm, dtype=dtype
    )
