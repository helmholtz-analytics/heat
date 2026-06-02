import heat as ht
from heat.testing.basic_test import TestCase

import torch
import numpy as np

class TestIndexing(TestCase):
    def test_nonzero(self):
        for split in [None, 0, 1]:
            for cond_type in ['mean', 'max']:
                a = ht.random.random((2*self.comm.size, 3*self.comm.size, 4*self.comm.size))
                if cond_type == 'mean':
                    cond = a > a.mean() / 2
                elif cond_type == 'max':
                    cond = a == a.max()
                else:
                    raise NotImplementedError

                nz_as_tuple = ht.nonzero(cond, as_tuple=True)
                nz_as_tuple_ref = np.nonzero(cond.numpy())
                for i in range(len(nz_as_tuple)):
                    self.assertEqual(nz_as_tuple[i].dtype, ht.int64)
                    self.assertTrue(np.allclose(nz_as_tuple[i].numpy(), nz_as_tuple_ref[i]))

                nz_no_tuple = ht.nonzero(cond, as_tuple=False)
                nz_no_tuple_ref = torch.nonzero(cond.resplit(None), as_tuple=False)
                self.assertEqual(nz_no_tuple.dtype, ht.int64)
                self.assertTrue(np.allclose(nz_no_tuple.numpy(), nz_no_tuple_ref.numpy()))

                if cond_type == 'max':
                    self.assertEqual(len(cond[cond]), 1)
                    for me in nz_as_tuple:
                        self.assertEqual(me.shape, (1,))
                    self.assertEqual(nz_no_tuple.shape, (1, a.ndim))


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

    def test_where(self):
        # cases to test
        # no x and y
        a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=None)
        cond = a > 3
        wh = ht.where(cond)
        self.assertEqual(len(wh), 2)
        self.assertEqual(wh[0].gshape[0], 6)
        self.assertEqual(wh[0].dtype, ht.int64)
        self.assertEqual(wh[0].split, None)
        # split
        a = ht.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split=1)
        cond = a > 3
        wh = ht.where(cond)
        self.assertEqual(wh.gshape, (6, 2))
        self.assertEqual(wh.dtype, ht.int64)
        self.assertEqual(wh.split, 0)

        # not split cond
        a = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]], split=None)
        res = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 3.0, -1.0]], split=None)
        wh = ht.where(a < 4.0, a, -1)
        self.assertTrue(
            ht.equal(a[ht.nonzero(a < 4)], ht.array([0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 3.0]))
        )
        self.assertTrue(ht.equal(wh, res))
        self.assertEqual(wh.gshape, (3, 3))
        self.assertEqual(wh.dtype, ht.float32)

        # split cond
        a = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]], split=0)
        res = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 3.0, -1.0]], split=0)
        wh = ht.where(a < 4.0, a, -1)
        self.assertTrue(ht.all(wh[ht.nonzero(a >= 4)] == -1))
        self.assertTrue(ht.equal(wh, res))
        self.assertEqual(wh.gshape, (3, 3))
        self.assertEqual(wh.dtype, ht.float32)
        self.assertEqual(wh.split, 0)

        a = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]], split=1)
        res = ht.array([[0.0, 1.0, 2.0], [0.0, 2.0, -1.0], [0.0, 3.0, -1.0]], split=1)
        wh = ht.where(a < 4.0, a, -1.0)
        self.assertTrue(ht.equal(wh, res))
        self.assertEqual(wh.gshape, (3, 3))
        self.assertEqual(wh.dtype, ht.float)
        self.assertEqual(wh.split, 1)

        with self.assertRaises(TypeError):
            ht.where(cond, a)

        with self.assertRaises(NotImplementedError):
            ht.where(cond, ht.ones((3, 3), split=0), ht.ones((3, 3), split=1))
