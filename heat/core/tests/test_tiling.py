import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestSplitTiles(TestCase):
    # most of the cases are covered by the resplit tests
    def test_raises(self):
        length = torch.tensor([i + 20 for i in range(2)])
        test = torch.arange(torch.prod(length), dtype=torch.float64).reshape(
            [i + 20 for i in range(2)]
        )
        a = ht.array(test, split=1)
        tiles = ht.tiling.SplitTiles(a)
        with self.assertRaises(TypeError):
            tiles["p"]
        with self.assertRaises(TypeError):
            tiles[0] = "p"
        with self.assertRaises(TypeError):
            tiles["p"] = "p"

    def test_misc_coverage(self):
        length = torch.tensor([i + 5 for i in range(3)])
        test = torch.arange(torch.prod(length), dtype=torch.float64).reshape(
            [i + 5 for i in range(3)]
        )
        a = ht.array(test, split=None)
        tiles = ht.tiling.SplitTiles(a)
        self.assertTrue(torch.all(tiles.tile_locations == a.comm.rank))
        a = ht.resplit(a, 0)
        tiles = ht.tiling.SplitTiles(a)
        if a.comm.size == 3:
            # definition of adjusting tests is he same logic as the code itself,
            #   therefore, fixed tests are issued for one process confic
            tile_dims = torch.tensor([[2.0, 2.0, 1.0], [2.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
            res = tiles.tile_dimensions
            self.assertTrue(torch.equal(tile_dims, res))
            testing_tensor = torch.tensor(
                [
                    [
                        [168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0],
                        [175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0],
                        [182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0],
                        [189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0],
                        [196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0],
                        [203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
                    ]
                ],
                dtype=torch.float64,
            )
            if a.comm.rank == 2:
                self.assertTrue(torch.equal(tiles[2], testing_tensor))
            tiles[2] = 1000
            sl = tiles[2]
            if a.comm.rank == 2:
                self.assertEqual(torch.Size([1, 6, 7]), sl.shape)
                self.assertTrue(torch.all(sl == 1000))
            else:
                self.assertTrue(sl is None)


class TestSquareDiagTiles(TestCase):
    # arrs = (m_eq_n_s0, m_eq_n_s1, m_gr_n_s0, m_gr_n_s1, m_ls_n_s0, m_ls_n_s1)
    if ht.MPI_WORLD.size > 1:

        def test_init_raises(self):
            # need to test the raises here
            with self.assertRaises(TypeError):
                ht.core.tiling.SquareDiagTiles("sdkd", tiles_per_proc=1)
            with self.assertRaises(TypeError):
                ht.core.tiling.SquareDiagTiles(ht.arange(2), tiles_per_proc="sdf")
            with self.assertRaises(ValueError):
                ht.core.tiling.SquareDiagTiles(ht.arange(2), tiles_per_proc=0)
            with self.assertRaises(ValueError):
                ht.core.tiling.SquareDiagTiles(ht.arange(2), tiles_per_proc=1)

        def test_properties(self):
            # ---- m = n ------------- properties ------ s0 -----------
            m_eq_n_s0 = ht.random.randn(47, 47, split=0)
            # m_eq_n_s0.create_square_diag_tiles(tiles_per_proc=1)
            m_eq_n_s0_t1 = ht.tiling.SquareDiagTiles(m_eq_n_s0, tiles_per_proc=1)
            m_eq_n_s0_t2 = ht.tiling.SquareDiagTiles(m_eq_n_s0, tiles_per_proc=2)
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

            # ---- m = n ------------- properties ------ s1 -----------
            m_eq_n_s1 = ht.random.randn(47, 47, split=1)
            m_eq_n_s1_t1 = ht.core.tiling.SquareDiagTiles(m_eq_n_s1, tiles_per_proc=1)
            m_eq_n_s1_t2 = ht.core.tiling.SquareDiagTiles(m_eq_n_s1, tiles_per_proc=2)
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

            # ---- m > n ------------- properties ------ s0 -----------
            m_gr_n_s0 = ht.random.randn(38, 128, split=0)
            m_gr_n_s0_t1 = ht.core.tiling.SquareDiagTiles(m_gr_n_s0, tiles_per_proc=1)
            m_gr_n_s0_t2 = ht.core.tiling.SquareDiagTiles(m_gr_n_s0, tiles_per_proc=2)
            if m_eq_n_s1.comm.size == 3:
                # col_inds
                self.assertEqual(m_gr_n_s0_t1.col_indices, [0, 13, 26])
                self.assertEqual(m_gr_n_s0_t2.col_indices, [0, 7, 13, 20, 26, 32])
                # row inds
                self.assertEqual(m_gr_n_s0_t1.row_indices, [0, 13, 26])
                self.assertEqual(m_gr_n_s0_t2.row_indices, [0, 7, 13, 20, 26, 32])
                # tile cols per proc
                self.assertEqual(m_gr_n_s0_t1.tile_columns_per_process, [3, 3, 3])
                self.assertEqual(m_gr_n_s0_t2.tile_columns_per_process, [6, 6, 6])
                # tile rows per proc
                self.assertEqual(m_gr_n_s0_t1.tile_rows_per_process, [1, 1, 1])
                self.assertEqual(m_gr_n_s0_t2.tile_rows_per_process, [2, 2, 2])
            # last diag pr
            self.assertEqual(m_gr_n_s0_t1.last_diagonal_process, m_eq_n_s1.comm.size - 1)
            self.assertEqual(m_gr_n_s0_t2.last_diagonal_process, m_eq_n_s1.comm.size - 1)
            # tile cols
            self.assertEqual(m_gr_n_s0_t1.tile_columns, m_eq_n_s1.comm.size)
            self.assertEqual(m_gr_n_s0_t2.tile_columns, m_eq_n_s1.comm.size * 2)
            # tile rows
            self.assertEqual(m_gr_n_s0_t1.tile_rows, m_eq_n_s1.comm.size)
            self.assertEqual(m_gr_n_s0_t2.tile_rows, m_eq_n_s1.comm.size * 2)

            # ---- m > n ------------- properties ------ s1 -----------
            m_gr_n_s1 = ht.random.randn(38, 128, split=1)
            m_gr_n_s1_t1 = ht.core.tiling.SquareDiagTiles(m_gr_n_s1, tiles_per_proc=1)
            m_gr_n_s1_t2 = ht.core.tiling.SquareDiagTiles(m_gr_n_s1, tiles_per_proc=2)
            if m_eq_n_s1.comm.size == 3:
                # col_inds
                self.assertEqual(m_gr_n_s1_t1.col_indices, [0, 38, 43, 86, 128, 171])
                self.assertEqual(m_gr_n_s1_t2.col_indices, [0, 19, 38, 43, 86, 128, 171])
                # row inds
                self.assertEqual(m_gr_n_s1_t1.row_indices, [0])
                self.assertEqual(m_gr_n_s1_t2.row_indices, [0, 19])
                # tile cols per proc
                self.assertEqual(m_gr_n_s1_t1.tile_columns_per_process, [2, 1, 1])
                self.assertEqual(m_gr_n_s1_t2.tile_columns_per_process, [3, 1, 1])
                # tile rows per proc
                self.assertEqual(m_gr_n_s1_t1.tile_rows_per_process, [1, 1, 1])
                self.assertEqual(m_gr_n_s1_t2.tile_rows_per_process, [2, 2, 2])
                # last diag pr
                self.assertEqual(m_gr_n_s1_t1.last_diagonal_process, 0)
                self.assertEqual(m_gr_n_s1_t2.last_diagonal_process, 0)
                # tile cols
                self.assertEqual(m_gr_n_s1_t1.tile_columns, 6)
                self.assertEqual(m_gr_n_s1_t2.tile_columns, 7)
                # tile rows
                self.assertEqual(m_gr_n_s1_t1.tile_rows, 1)
                self.assertEqual(m_gr_n_s1_t2.tile_rows, 2)

            # ---- m < n ------------- properties ------ s0 -----------
            m_ls_n_s0 = ht.random.randn(323, 49, split=0)
            m_ls_n_s0_t1 = ht.core.tiling.SquareDiagTiles(m_ls_n_s0, tiles_per_proc=1)
            m_ls_n_s0_t2 = ht.core.tiling.SquareDiagTiles(m_ls_n_s0, tiles_per_proc=2)
            if m_eq_n_s1.comm.size == 3:
                # col_inds
                self.assertEqual(m_ls_n_s0_t1.col_indices, [0])
                self.assertEqual(m_ls_n_s0_t2.col_indices, [0, 25])
                # row inds
                self.assertEqual(m_ls_n_s0_t1.row_indices, [0, 49, 109, 216])
                self.assertEqual(m_ls_n_s0_t2.row_indices, [0, 25, 49, 110, 163, 216, 270])
                # tile cols per proc
                self.assertEqual(m_ls_n_s0_t1.tile_columns_per_process, [1])
                self.assertEqual(m_ls_n_s0_t2.tile_columns_per_process, [2])
                # tile rows per proc
                self.assertEqual(m_ls_n_s0_t1.tile_rows_per_process, [2, 1, 1])
                self.assertEqual(m_ls_n_s0_t2.tile_rows_per_process, [3, 2, 2])
                # last diag pr
                self.assertEqual(m_ls_n_s0_t1.last_diagonal_process, 0)
                self.assertEqual(m_ls_n_s0_t2.last_diagonal_process, 0)
                # tile cols
                self.assertEqual(m_ls_n_s0_t1.tile_columns, 1)
                self.assertEqual(m_ls_n_s0_t2.tile_columns, 2)
                # tile rows
                self.assertEqual(m_ls_n_s0_t1.tile_rows, 4)
                self.assertEqual(m_ls_n_s0_t2.tile_rows, 7)

            # ---- m < n ------------- properties ------ s1 -----------
            m_ls_n_s1 = ht.random.randn(323, 49, split=1)
            m_ls_n_s1_t1 = ht.core.tiling.SquareDiagTiles(m_ls_n_s1, tiles_per_proc=1)
            m_ls_n_s1_t2 = ht.core.tiling.SquareDiagTiles(m_ls_n_s1, tiles_per_proc=2)
            if m_eq_n_s1.comm.size == 3:
                # col_inds
                self.assertEqual(m_ls_n_s1_t1.col_indices, [0, 17, 33])
                self.assertEqual(m_ls_n_s1_t2.col_indices, [0, 9, 17, 25, 33, 41])
                # row inds
                self.assertEqual(m_ls_n_s1_t1.row_indices, [0, 17, 33, 49])
                self.assertEqual(m_ls_n_s1_t2.row_indices, [0, 9, 17, 25, 33, 41, 49])
                # tile cols per proc
                self.assertEqual(m_ls_n_s1_t1.tile_columns_per_process, [1, 1, 1])
                self.assertEqual(m_ls_n_s1_t2.tile_columns_per_process, [2, 2, 2])
                # tile rows per proc
                self.assertEqual(m_ls_n_s1_t1.tile_rows_per_process, [4, 4, 4])
                self.assertEqual(m_ls_n_s1_t2.tile_rows_per_process, [7, 7, 7])
                # last diag pr
                self.assertEqual(m_ls_n_s1_t1.last_diagonal_process, 2)
                self.assertEqual(m_ls_n_s1_t2.last_diagonal_process, 2)
                # tile cols
                self.assertEqual(m_ls_n_s1_t1.tile_columns, 3)
                self.assertEqual(m_ls_n_s1_t2.tile_columns, 6)
                # tile rows
                self.assertEqual(m_ls_n_s1_t1.tile_rows, 4)
                self.assertEqual(m_ls_n_s1_t2.tile_rows, 7)

        def test_local_set_get(self):
            # this function will also test all of the start_stop functions and the
            # --------------------- local ----------- s0 ----------------
            # (int), (int, int), (slice, int), (slice, slice), (int, slice)
            m_eq_n_s0 = ht.zeros((25, 25), split=0)
            m_eq_n_s0_t2 = ht.core.tiling.SquareDiagTiles(m_eq_n_s0, tiles_per_proc=2)
            k = (slice(0, 10), slice(2, None))
            m_eq_n_s0_t2.local_set(key=k, value=1)
            lcl_key = m_eq_n_s0_t2.local_to_global(key=k, rank=m_eq_n_s0.comm.rank)
            st_sp = m_eq_n_s0_t2.get_start_stop(key=lcl_key)
            sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            lcl_slice = m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
            lcl_shape = m_eq_n_s0_t2.local_get(key=(slice(None), slice(None))).shape
            self.assertEqual(lcl_shape, m_eq_n_s0.lshape)
            self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
            # reset base
            m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            k = (1, 1)
            m_eq_n_s0_t2.local_set(key=k, value=1)
            lcl_key = m_eq_n_s0_t2.local_to_global(key=k, rank=m_eq_n_s0.comm.rank)
            st_sp = m_eq_n_s0_t2.get_start_stop(key=lcl_key)
            sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            lcl_slice = m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
            self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
            # reset base
            m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            k = 1
            m_eq_n_s0_t2.local_set(key=k, value=1)
            lcl_key = m_eq_n_s0_t2.local_to_global(key=k, rank=m_eq_n_s0.comm.rank)
            st_sp = m_eq_n_s0_t2.get_start_stop(key=lcl_key)
            sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            lcl_slice = m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
            self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
            # reset base
            m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            # --------------------- local ----------- s1 ----------------
            m_eq_n_s1 = ht.zeros((25, 25), split=1)
            m_eq_n_s1_t2 = ht.core.tiling.SquareDiagTiles(m_eq_n_s1, tiles_per_proc=2)
            k = (slice(0, 2), slice(0, None))
            m_eq_n_s1_t2.local_set(key=k, value=1)
            lcl_key = m_eq_n_s1_t2.local_to_global(key=k, rank=m_eq_n_s1.comm.rank)
            st_sp = m_eq_n_s1_t2.get_start_stop(key=lcl_key)
            sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            lcl_slice = m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
            lcl_shape = m_eq_n_s1_t2.local_get(key=(slice(None), slice(None))).shape
            self.assertEqual(lcl_shape, m_eq_n_s1.lshape)
            self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
            # reset base
            m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0
            if ht.MPI_WORLD.size > 2:
                k = (5, 1)
                m_eq_n_s1_t2.local_set(key=k, value=1)
                lcl_key = m_eq_n_s1_t2.local_to_global(key=k, rank=m_eq_n_s1.comm.rank)
                st_sp = m_eq_n_s1_t2.get_start_stop(key=lcl_key)
                sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                lcl_slice = m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                # reset base
                m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            k = 2
            m_eq_n_s1_t2.local_set(key=k, value=1)
            lcl_key = m_eq_n_s1_t2.local_to_global(key=k, rank=m_eq_n_s1.comm.rank)
            st_sp = m_eq_n_s1_t2.get_start_stop(key=lcl_key)
            sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
            lcl_slice = m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
            self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
            # reset base
            m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            # --------------------- global ---------- s0 ----------------
            m_eq_n_s0 = ht.zeros((25, 25), split=0)
            m_eq_n_s0_t2 = ht.core.tiling.SquareDiagTiles(m_eq_n_s0, tiles_per_proc=2)
            k = 2
            m_eq_n_s0_t2[k] = 1
            if m_eq_n_s0_t2[k] is not None:
                st_sp = m_eq_n_s0_t2.get_start_stop(key=k)
                sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                lcl_slice = m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                # reset base
                m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0
            if ht.MPI_WORLD.size > 2:
                k = (5, 5)
                m_eq_n_s0_t2[k] = 1
                if m_eq_n_s0_t2[k] is not None:
                    st_sp = m_eq_n_s0_t2.get_start_stop(key=k)
                    sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                    lcl_slice = m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                    self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                    # reset base
                    m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

                k = (slice(0, 2), slice(1, 5))
                m_eq_n_s0_t2[k] = 1
                if m_eq_n_s0_t2[k] is not None:
                    st_sp = m_eq_n_s0_t2.get_start_stop(key=k)
                    sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                    lcl_slice = m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                    self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                    # reset base
                    m_eq_n_s0.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            # --------------------- global ---------- s1 ----------------
            m_eq_n_s1 = ht.zeros((25, 25), split=1)
            m_eq_n_s1_t2 = ht.core.tiling.SquareDiagTiles(m_eq_n_s1, tiles_per_proc=2)
            k = (slice(0, 3), slice(0, 2))
            m_eq_n_s1_t2[k] = 1
            if m_eq_n_s1_t2[k] is not None:
                st_sp = m_eq_n_s1_t2.get_start_stop(key=k)
                sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                lcl_slice = m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                # reset base
                m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            # k = (slice(0, 3), slice(0, 2))
            if ht.MPI_WORLD.size > 2:
                k = (5, 5)
                m_eq_n_s1_t2[k] = 1
                if m_eq_n_s1_t2[k] is not None:
                    st_sp = m_eq_n_s1_t2.get_start_stop(key=k)
                    sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                    lcl_slice = m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                    self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                    # reset base
                    m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0

            k = (slice(0, 3), 3)
            m_eq_n_s1_t2[k] = 1
            if m_eq_n_s1_t2[k] is not None:
                st_sp = m_eq_n_s1_t2.get_start_stop(key=k)
                sz = st_sp[1] - st_sp[0], st_sp[3] - st_sp[2]
                lcl_slice = m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]]
                self.assertTrue(torch.all(lcl_slice - torch.ones(sz) == 0))
                # reset base
                m_eq_n_s1.larray[st_sp[0] : st_sp[1], st_sp[2] : st_sp[3]] = 0
            with self.assertRaises(ValueError):
                m_eq_n_s1_t2[1, :]
            with self.assertRaises(TypeError):
                m_eq_n_s1_t2["asdf"]
            with self.assertRaises(TypeError):
                m_eq_n_s1_t2[1, "asdf"]
            with self.assertRaises(ValueError):
                m_eq_n_s1_t2[1, :] = 2
            with self.assertRaises(ValueError):
                m_eq_n_s1_t2.get_start_stop(key=(1, slice(None)))
            with self.assertRaises(ValueError):
                m_eq_n_s1_t2.get_start_stop(key=(1, slice(None)))
