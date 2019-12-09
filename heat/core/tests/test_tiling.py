import torch
import unittest

import heat as ht


class TestTiling(unittest.TestCase):

    # arrs = (m_eq_n_s0, m_eq_n_s1, m_gr_n_s0, m_gr_n_s1, m_ls_n_s0, m_ls_n_s1)

    def test_init_raises(self):
        # need to test the raises here
        with self.assertRaises(TypeError):
            ht.tiling.SquareDiagTiles("sdkd", tiles_per_proc=1)
        with self.assertRaises(ValueError):
            ht.tiling.SquareDiagTiles(ht.arange(2), tiles_per_proc=0)
        with self.assertRaises(TypeError):
            ht.tiling.SquareDiagTiles(ht.arange(2), tiles_per_proc="sdf")
        with self.assertRaises(ValueError):
            ht.tiling.SquareDiagTiles(ht.arange(2), tiles_per_proc=1)

    def test_properties_global_set_get(self):
        m_eq_n_s0 = ht.random.randn(47, 47, split=0)
        m_eq_n_s0_t1 = ht.tiling.SquareDiagTiles(m_eq_n_s0, tiles_per_proc=1)
        m_eq_n_s0_t2 = ht.tiling.SquareDiagTiles(m_eq_n_s0, tiles_per_proc=2)
        # ----------------- properties ------ s0 -----------
        # arr
        self.assertTrue(ht.equal(m_eq_n_s0_t1.arr, m_eq_n_s0))
        self.assertTrue(ht.equal(m_eq_n_s0_t2.arr, m_eq_n_s0))
        # lshape_map
        self.assertTrue(torch.equal(m_eq_n_s0_t1.lshape_map, m_eq_n_s0.create_lshape_map()))
        self.assertTrue(torch.equal(m_eq_n_s0_t2.lshape_map, m_eq_n_s0.create_lshape_map()))

        if m_eq_n_s0.comm.size == 3:
            # col_inds
            self.assertEqual(m_eq_n_s0_t1.col_indices, [0, 16, 32])
            self.assertEqual(m_eq_n_s0_t2.col_indices, [0, 8, 16, 24, 32, 40])
            # row inds
            self.assertEqual(m_eq_n_s0_t1.row_indices, [0, 16, 32])
            self.assertEqual(m_eq_n_s0_t2.row_indices, [0, 8, 16, 24, 32, 40])
            # tile cols per proc
            self.assertEqual(m_eq_n_s0_t1.tile_columns_per_process, [3, 3, 3])
            self.assertEqual(m_eq_n_s0_t2.tile_columns_per_process, [6, 6, 6])
            # tile rows per proc
            self.assertEqual(m_eq_n_s0_t1.tile_rows_per_process, [1, 1, 1])
            self.assertEqual(m_eq_n_s0_t2.tile_rows_per_process, [2, 2, 2])

        # last diag pr
        self.assertEqual(m_eq_n_s0_t1.last_diagonal_process, m_eq_n_s0.comm.size - 1)
        self.assertEqual(m_eq_n_s0_t2.last_diagonal_process, m_eq_n_s0.comm.size - 1)
        # tile cols
        self.assertEqual(m_eq_n_s0_t1.tile_columns, m_eq_n_s0.comm.size)
        self.assertEqual(m_eq_n_s0_t2.tile_columns, m_eq_n_s0.comm.size * 2)
        # tile rows
        self.assertEqual(m_eq_n_s0_t1.tile_rows, m_eq_n_s0.comm.size)
        self.assertEqual(m_eq_n_s0_t2.tile_rows, m_eq_n_s0.comm.size * 2)

        # ----------------- properties ------ s1 -----------
        m_eq_n_s1 = ht.random.randn(47, 47, split=1)
        m_eq_n_s1_t1 = ht.tiling.SquareDiagTiles(m_eq_n_s1, tiles_per_proc=1)
        m_eq_n_s1_t2 = ht.tiling.SquareDiagTiles(m_eq_n_s1, tiles_per_proc=2)
        # lshape_map
        self.assertTrue(torch.equal(m_eq_n_s1_t1.lshape_map, m_eq_n_s1.create_lshape_map()))
        self.assertTrue(torch.equal(m_eq_n_s1_t2.lshape_map, m_eq_n_s1.create_lshape_map()))

        if m_eq_n_s1.comm.size == 3:
            # col_inds
            self.assertEqual(m_eq_n_s1_t1.col_indices, [0, 16, 32])
            self.assertEqual(m_eq_n_s1_t2.col_indices, [0, 8, 16, 24, 32, 40])
            # row inds
            self.assertEqual(m_eq_n_s1_t1.row_indices, [0, 16, 32])
            self.assertEqual(m_eq_n_s1_t2.row_indices, [0, 8, 16, 24, 32, 40])
            # tile cols per proc
            self.assertEqual(m_eq_n_s1_t1.tile_columns_per_process, [1, 1, 1])
            self.assertEqual(m_eq_n_s1_t2.tile_columns_per_process, [2, 2, 2])
            # tile rows per proc
            self.assertEqual(m_eq_n_s1_t1.tile_rows_per_process, [3, 3, 3])
            self.assertEqual(m_eq_n_s1_t2.tile_rows_per_process, [6, 6, 6])

        # last diag pr
        self.assertEqual(m_eq_n_s1_t1.last_diagonal_process, m_eq_n_s1.comm.size - 1)
        self.assertEqual(m_eq_n_s1_t2.last_diagonal_process, m_eq_n_s1.comm.size - 1)
        # tile cols
        self.assertEqual(m_eq_n_s1_t1.tile_columns, m_eq_n_s1.comm.size)
        self.assertEqual(m_eq_n_s1_t2.tile_columns, m_eq_n_s1.comm.size * 2)
        # tile rows
        self.assertEqual(m_eq_n_s1_t1.tile_rows, m_eq_n_s1.comm.size)
        self.assertEqual(m_eq_n_s1_t2.tile_rows, m_eq_n_s1.comm.size * 2)

        # m_gr_n_s0 = ht.random.randn(38, 128, split=0)
        # m_gr_n_s0_t1 = ht.tiling.SquareDiagTiles(m_gr_n_s0, tiles_per_proc=1)
        # m_gr_n_s0_t2 = ht.tiling.SquareDiagTiles(m_gr_n_s0, tiles_per_proc=2)
        #
        # m_gr_n_s1 = ht.random.randn(38, 128, split=1)
        # m_gr_n_s1_t1 = ht.tiling.SquareDiagTiles(m_gr_n_s1, tiles_per_proc=1)
        # m_gr_n_s1_t2 = ht.tiling.SquareDiagTiles(m_gr_n_s1, tiles_per_proc=2)
        #
        # m_ls_n_s0 = ht.random.randn(323, 49, split=0)
        # m_ls_n_s0_t1 = ht.tiling.SquareDiagTiles(m_ls_n_s0, tiles_per_proc=1)
        # m_ls_n_s0_t2 = ht.tiling.SquareDiagTiles(m_ls_n_s0, tiles_per_proc=2)
        #
        # m_ls_n_s1 = ht.random.randn(323, 49, split=1)
        # m_ls_n_s1_t1 = ht.tiling.SquareDiagTiles(m_ls_n_s1, tiles_per_proc=1)
        # m_ls_n_s1_t2 = ht.tiling.SquareDiagTiles(m_ls_n_s1, tiles_per_proc=2)

        # last diag pr
        # tile cols per pr
        # tile rows per pr
        # lshape map
        #
        pass
