"""
Script containing various linear algebra functions
"""

import torch

import heat as ht
from ..dndarray import DNDarray
from ..sanitation import sanitize_in

__all__ = ["matrix_exp", "expm"]


def matrix_exp(A: DNDarray) -> DNDarray:
    r"""
    Computes the matrix exponential of a square matrix.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
    this function computes the **matrix exponential** of :math:`A \in \mathbb{K}^{n \times n}`, which is defined as

    .. math::
        \mathrm{matrix\_exp}(A) = \sum_{k=0}^\infty \frac{1}{k!}A^k \in \mathbb{K}^{n \times n}.

    If the matrix :math:`A` has eigenvalues :math:`\lambda_i \in \mathbb{C}`,
    the matrix :math:`\mathrm{matrix\_exp}(A)` has eigenvalues :math:`e^{\lambda_i} \in \mathbb{C}`.

    Supports input of bfloat16, float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
    the output has the same batch dimensions.

    .. note::
         A may only be distributed in the batch dimensions.

    .. seealso::
             :func:`torch.linalg.matrix_exp` is called under the hood on the local data.

    Args:
        A (DNDarray): DNDarray of shape `(*, n, n)` where `*` is zero or more batch dimensions.

    Example::

        >>> A = ht.empty((2, 2, 2), split=0)
        >>> A[0, :, :] = ht.eye((2, 2))
        >>> A[1, :, :] = 2 * ht.eye((2, 2))
        >>> ht.linalg.matrix_exp(A)
        DNDarray([[[2.7183, 0.0000],
           [0.0000, 2.7183]],

          [[7.3891, 0.0000],
           [0.0000, 7.3891]]], dtype=ht.float32, device=cpu:0, split=0)
    """
    sanitize_in(A)

    if A.is_distributed() and A.split >= A.ndim - 2:
        raise ValueError(
            f"A of shape {A.shape} may only be distributed in batched dimensions but is distributed in {A.split}"
        )
    out = ht.empty_like(A)
    out.larray[...] = torch.linalg.matrix_exp(A.larray)
    return out


expm = matrix_exp  # provide alias with name of scipy equivalent
"""Alias for :py:func:matrix_exp"""
