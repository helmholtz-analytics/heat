"""
Generate matrices for specific tests and functions
"""

from heat import core
from ...core.dndarray import DNDarray
from ...core.communication import Communication
from ...core.devices import Device
from ...core.types import datatype
from typing import Type, Union

__all__ = ["parter"]


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
        raise ValueError("expected split value to be either {{None,0,1}}, but was {}".format(split))

    return 1.0 / (II - JJ + 0.5)
