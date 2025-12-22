Module heat.sparse.arithmetics
==============================
Arithmetic functions for Dcsr_matrices

Functions
---------

`add(t1: DCSR_matrix, t2: DCSR_matrix, orientation: str = 'row') ‑> heat.sparse.dcsx_matrix.DCSR_matrix`
:   Element-wise addition of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.sparse.DCSR_matrix`) whose elements are to be added
    as argument and returns a ``DCSR_matrix`` containing the results of element-wise addition of ``t1`` and ``t2``.

    Parameters
    ----------
    t1: DCSR_matrix
        The first operand involved in the addition
    t2: DCSR_matrix
        The second operand involved in the addition
    orientation: str, optional
        The orientation of the operation. Options: 'row' or 'col'
        Default: 'row'

    Examples
    --------
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3]), indices: tensor([0, 2, 2]), data: tensor([1., 2., 3.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> heat_sparse_csr.todense()
    DNDarray([[1., 0., 2.],
              [0., 0., 3.]], dtype=ht.float32, device=cpu:0, split=0)
    >>> sum_sparse = heat_sparse_csr + heat_sparse_csr
        (or)
    >>> sum_sparse = ht.sparse.sparse_add(heat_sparse_csr, heat_sparse_csr)
    >>> sum_sparse
    (indptr: tensor([0, 2, 3], dtype=torch.int32), indices: tensor([0, 2, 2], dtype=torch.int32), data: tensor([2., 4., 6.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> sum_sparse.todense()
    DNDarray([[2., 0., 4.],
              [0., 0., 6.]], dtype=ht.float32, device=cpu:0, split=0)

`mul(t1: DCSR_matrix, t2: DCSR_matrix, orientation: str = 'row') ‑> heat.sparse.dcsx_matrix.DCSR_matrix`
:   Element-wise multiplication (NOT matrix multiplication) of values from two operands, commutative.
    Takes the first and second operand (scalar or :class:`~heat.sparse.DCSR_matrix`) whose elements are to be
    multiplied as argument.

    Parameters
    ----------
    t1: DCSR_matrix
        The first operand involved in the multiplication
    t2: DCSR_matrix
        The second operand involved in the multiplication
    orientation: str, optional
        The orientation of the operation. Options: 'row' or 'col'
        Default: 'row'

    Examples
    --------
    >>> heat_sparse_csr
    (indptr: tensor([0, 2, 3]), indices: tensor([0, 2, 2]), data: tensor([1., 2., 3.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> heat_sparse_csr.todense()
    DNDarray([[1., 0., 2.],
              [0., 0., 3.]], dtype=ht.float32, device=cpu:0, split=0)
    >>> pdt_sparse = heat_sparse_csr * heat_sparse_csr
        (or)
    >>> pdt_sparse = ht.sparse.sparse_mul(heat_sparse_csr, heat_sparse_csr)
    >>> pdt_sparse
    (indptr: tensor([0, 2, 3]), indices: tensor([0, 2, 2]), data: tensor([1., 4., 9.]), dtype=ht.float32, device=cpu:0, split=0)
    >>> pdt_sparse.todense()
    DNDarray([[1., 0., 4.],
              [0., 0., 9.]], dtype=ht.float32, device=cpu:0, split=0)
