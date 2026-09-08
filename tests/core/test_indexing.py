import pytest

import heat as ht
from heat.testing.basic_test import TestCase

import torch
import numpy as np

def compare_ht_where_to_numpy_where(ht_res, np_res):
    if isinstance(np_res, tuple):
        assert isinstance(ht_res, tuple)

        assert len(ht_res) == len(np_res)
        for _ht_res, _np_res in zip(ht_res, np_res):
            assert ht.equal(_ht_res, ht.array(_np_res))

        assert ht_res[0].dtype == ht.int64

    else:
        assert np.allclose(ht_res.shape, np_res.shape)
        assert ht.equal(ht_res, ht.array(np_res))
        assert ht_res[0].dtype == ht.types.canonical_heat_type(np_res.dtype)

@pytest.mark.parametrize('split', [None, 0, 1])
@pytest.mark.parametrize('cond_type', ['mean', 'max'])
def test_nonzero(split, cond_type):
    a = ht.random.random((2*ht.comm.size, 3*ht.comm.size, 4*ht.comm.size))
    if cond_type == 'mean':
        cond = a > a.mean() / 2
    elif cond_type == 'max':
        cond = a == a.max()
    else:
        raise NotImplementedError

    nz_as_tuple = ht.nonzero(cond, as_tuple=True)
    nz_as_tuple_ref = np.nonzero(cond.numpy())
    for i in range(len(nz_as_tuple)):
        assert nz_as_tuple[i].dtype == ht.int64
        assert np.allclose(nz_as_tuple[i].numpy(), nz_as_tuple_ref[i])

    nz_no_tuple = ht.nonzero(cond, as_tuple=False)
    nz_no_tuple_ref = torch.nonzero(cond.resplit(None), as_tuple=False)
    assert nz_no_tuple.dtype == ht.int64
    assert np.allclose(nz_no_tuple.numpy(), nz_no_tuple_ref.numpy())

    if cond_type == 'max':
        assert len(cond[cond]) == 1
        for me in nz_as_tuple:
            assert me.shape == (1,)
        assert nz_no_tuple.shape == (1, a.ndim)

@pytest.mark.parametrize('split', [None, 0, 1])
def test_where_against_numpy(split):
    # no x and y
    a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=split)
    cond = a > 3
    ht_res = ht.where(cond)
    np_res = np.where(cond.numpy())
    compare_ht_where_to_numpy_where(ht_res, np_res)
    if split is None or not cond.is_distributed():
        assert ht_res[0].split == None
    else:
        assert ht_res[0].split == 0

    # x and y DNDarray
    a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=split, dtype=ht.float32)
    b = -a
    cond = a > 3
    ht_res = ht.where(cond, a, b)
    np_res = np.where(cond.numpy(), a.numpy(), b.numpy())
    compare_ht_where_to_numpy_where(ht_res, np_res)
    assert ht_res.split == split

    # x and y float
    a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=split, dtype=ht.float32)
    cond = a > 3
    ht_res = ht.where(cond, 1., -1.)
    np_res = np.where(cond.numpy(), 1., -1.)
    compare_ht_where_to_numpy_where(ht_res, np_res.astype(np.float32))
    assert ht_res.split == split

    # x and y int
    a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=split, dtype=ht.float32)
    cond = a > 3
    ht_res = ht.where(cond, 1, -1)
    np_res = np.where(cond.numpy(), 1, -1)
    compare_ht_where_to_numpy_where(ht_res, np_res.astype(np.int32))
    assert ht_res.split == split



class TestIndexing(TestCase):
    def test_nonzero_special_cases(self):
        # edge case: single non-zero element
        for split in [None, 0, 1]:
            a = ht.zeros((4, 3), dtype=ht.bool, split=split)
            a[1, 2] = True
            nz = ht.indexing.nonzero(a, as_tuple=False)
            self.assertTrue(ht.allclose(a[nz], a[a]))
            a.comm.Barrier()

        # as_tuple = False (torch-style output)
        a = ht.array([[1, 0, 0], [0, 4, 1], [0, 6, 0]], split=1)
        nz = ht.nonzero(a, as_tuple=False)
        self.assertEqual(nz.gshape, (4, 2))
        self.assertEqual(nz.dtype, ht.int64)
        if a.is_distributed():
            self.assertEqual(nz.split, 0)
        else:
            self.assertEqual(nz.split, None)
        t_a =  a.resplit_(None).larray
        t_nz = torch.nonzero(t_a, as_tuple=False)
        self.assertTrue(ht.equal(nz, ht.array(t_nz)))

        # attribute error
        a = a.numpy()
        with self.assertRaises(TypeError):
            ht.nonzero(a)

    def test_where_special_cases(self):
        # not split cond
        a = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]], split=None)
        res = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 3.0, -1.0]], split=None)
        ht_res = ht.where(a < 4.0, a, -1)
        self.assertTrue(
            ht.equal(a[ht.nonzero(a < 4)], ht.array([0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 3.0]))
        )
        self.assertTrue(ht.equal(ht_res, res))
        self.assertEqual(ht_res.gshape, (3, 3))
        self.assertEqual(ht_res.dtype, ht.float32)

        # split cond
        a = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]], split=0)
        res = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 3.0, -1.0]], split=0)
        ht_res = ht.where(a < 4.0, a, -1)
        self.assertTrue(ht.all(ht_res[ht.nonzero(a >= 4)] == -1))
        self.assertTrue(ht.equal(ht_res, res))
        self.assertEqual(ht_res.gshape, (3, 3))
        self.assertEqual(ht_res.dtype, ht.float32)
        self.assertEqual(ht_res.split, 0)

        a = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]], split=1)
        res = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 3.0, -1.0]], split=1)
        ht_res = ht.where(a < 4.0, a, -1.0)
        self.assertTrue(ht.equal(ht_res, res))
        self.assertEqual(ht_res.gshape, (3, 3))
        self.assertEqual(ht_res.dtype, ht.float)
        self.assertEqual(ht_res.split, 1)

        with self.assertRaises(TypeError):
            ht.where(a < 3, a)

        with self.assertRaises(NotImplementedError):
            ht.where(a < 3, ht.ones((3, 3), split=0), ht.ones((3, 3), split=1))
