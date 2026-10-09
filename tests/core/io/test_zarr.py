"""Tests for ``heat.core.io.zarr`` behaviour that is specific to this module."""

import os
import shutil
import unittest
from typing import Iterable

import numpy as np
import torch

import heat as ht
from heat.testing.basic_test import TestCase

from ._base import IOTestCase


@unittest.skipUnless(ht.io.supports("zarr"), "Requires zarr")
class TestZarr(TestCase):
    """Tests for the zarr loader's split handling."""

    SHAPE = (20, 10)

    def setUp(self):
        self.path = os.path.join(os.getcwd(), "test_zarr_split.zarr")
        self.data = np.arange(self.SHAPE[0] * self.SHAPE[1], dtype=np.float32).reshape(self.SHAPE)
        if ht.MPI_WORLD.rank == 0:
            import zarr

            try:
                array = zarr.create_array(self.path, shape=self.SHAPE, dtype=np.float32)
            except AttributeError:
                array = zarr.create(store=self.path, shape=self.SHAPE, dtype=np.float32)
            array[:] = self.data
        ht.MPI_WORLD.Barrier()

    def tearDown(self):
        ht.MPI_WORLD.Barrier()
        if ht.MPI_WORLD.rank == 0:
            import shutil

            shutil.rmtree(self.path, ignore_errors=True)
        ht.MPI_WORLD.Barrier()

    def test_split_defaults_to_none(self):
        """`load_zarr` used to default to split=0, unlike every other loader."""
        loaded = ht.load_zarr(self.path)
        self.assertIsNone(loaded.split)
        self.assertTrue(np.array_equal(loaded.numpy(), self.data))

    def test_explicit_split_is_honoured(self):
        for split in (0, 1):
            with self.subTest(split=split):
                loaded = ht.load_zarr(self.path, split=split)
                self.assertEqual(loaded.split, split)
                self.assertTrue(np.array_equal(loaded.numpy(), self.data))

    def test_slices_with_default_split(self):
        loaded = ht.load_zarr(self.path, slices=[slice(2, 9), slice(1, 6)])
        self.assertTrue(np.array_equal(loaded.numpy(), self.data[2:9, 1:6]))

    def test_wildcard_without_split_is_rejected(self):
        """Concatenating across files needs an axis, so the None default is an error."""
        with self.assertRaises(ValueError):
            ht.load_zarr(self.path, variable="RECEIVER_*/DATA")



@unittest.skipUnless(ht.io.supports("zarr"), "Requires zarr")
class TestZarrFormat(IOTestCase):
    """Tests moved from the former monolithic tests/core/test_io.py."""

    def test_load_zarr(self):

        import zarr

        test_data = np.arange(self.ZARR_SHAPE[0] * self.ZARR_SHAPE[1]).reshape(self.ZARR_SHAPE)
        dtype = np.float32
        if ht.MPI_WORLD.rank == 0:
            try:
                arr = zarr.create_array(
                    self.ZARR_TEMP_PATH, shape=self.ZARR_SHAPE, dtype=dtype
                )
            except AttributeError:
                arr = zarr.create(
                    store=self.ZARR_TEMP_PATH, shape=self.ZARR_SHAPE, dtype=dtype
                )
            arr[:] = test_data

        ht.MPI_WORLD.handle.Barrier()

        dndarray = ht.load_zarr(self.ZARR_TEMP_PATH)
        dndnumpy = dndarray.numpy()

        if ht.MPI_WORLD.rank == 0:
            self.assertTrue((dndnumpy == test_data).all())

        ht.MPI_WORLD.Barrier()

    def test_load_zarr_group(self):

        import zarr

        ht.MPI_WORLD.Barrier()

        # Write out a nested Zarr store
        original_data = np.arange(np.prod(self.ZARR_SHAPE)).reshape(self.ZARR_SHAPE)
        nested_group_name = "MAIN_0"
        array_name = "DATA"
        variable_path = f"{nested_group_name}/{array_name}"

        if ht.MPI_WORLD.rank == 0:
            root = zarr.open_group(self.ZARR_NESTED_PATH, mode="w")
            main_0 = root.create_group(nested_group_name)
            main_0.create_dataset(
                array_name,
                shape=original_data.shape,
                dtype=original_data.dtype,
                data=original_data,
            )

        ht.MPI_WORLD.Barrier()

        # Test loading using both positional and keyword arguments for different splits
        for split in [None, 0, 1]:
            # Test with positional argument
            with self.subTest(split=split, arg_type="positional"):
                ht_tensor_pos = ht.load(self.ZARR_NESTED_PATH, variable_path, split=split)
                self.assertIsInstance(ht_tensor_pos, ht.DNDarray)
                self.assertEqual(ht_tensor_pos.gshape, original_data.shape)
                self.assertTrue(np.array_equal(ht_tensor_pos.numpy(), original_data))

            # Test with keyword argument
            with self.subTest(split=split, arg_type="keyword"):
                ht_tensor_kw = ht.load(
                    self.ZARR_NESTED_PATH, variable=variable_path, split=split
                )
                self.assertIsInstance(ht_tensor_kw, ht.DNDarray)
                self.assertEqual(ht_tensor_kw.gshape, original_data.shape)
                self.assertTrue(np.array_equal(ht_tensor_kw.numpy(), original_data))

        ht.MPI_WORLD.Barrier()
        # test loading with wildcard
        num_chunks = self.comm.size * 2 + 1
        if self.comm.size > 3:
            # test empty ranks
            num_chunks = self.comm.size - 1

        np_testing_types = [np.int32, np.int64, np.float32, np.complex64]
        if not self.is_mps:
            np_testing_types.extend([np.float64, np.complex128])

        ht.MPI_WORLD.Barrier()
        for dtype in np_testing_types:
            global_data_shape = (num_chunks * 10, num_chunks * 5, 7)
            global_data = np.arange(np.prod(global_data_shape), dtype=dtype).reshape(global_data_shape)
            if self.comm.rank == 0:
                # create zarr store for split=0 and split=1
                chunk_shape_split0 = (10, global_data_shape[1], global_data_shape[2])
                chunk_shape_split1 = (global_data_shape[0], 5, global_data_shape[2])

                root_zarr = zarr.open_group(self.ZARR_OUT_PATH, mode="w")

                for i in range(num_chunks):
                    chunk_data_split0 = global_data[i * chunk_shape_split0[0] : (i + 1) * chunk_shape_split0[0], :, :]
                    chunk_group_split0 = root_zarr.create_group(f"CHUNK_{i}_SPLIT0")
                    chunk_group_split0.create_dataset(
                        "DATA",
                        shape=chunk_data_split0.shape,
                        dtype=chunk_data_split0.dtype,
                        data=chunk_data_split0
                    )

                    chunk_data_split1 = global_data[:, i * chunk_shape_split1[1] : (i + 1) * chunk_shape_split1[1], :]
                    chunk_group_split1 = root_zarr.create_group(f"CHUNK_{i}_SPLIT1")
                    chunk_group_split1.create_dataset(
                        "DATA",
                        shape=chunk_data_split1.shape,
                        dtype=chunk_data_split1.dtype,
                        data=chunk_data_split1
                    )
            ht.MPI_WORLD.Barrier()

            # test wildcard loading for split=0
            with self.subTest(dtype=dtype, split=0):
                ht_array_split0 = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT0/DATA", split=0, device=self.device)
                self.assertIsInstance(ht_array_split0, ht.DNDarray)
                self.assertEqual(ht_array_split0.gshape, global_data_shape)
                ht_array_split0.balance_()
                self.assertTrue((ht_array_split0.numpy() == global_data).all())
                self.assertTrue(ht_array_split0.dtype == ht.types.canonical_heat_type(dtype))

            # test wildcard loading for split=1
            with self.subTest(dtype=dtype, split=1):
                ht_array_split1 = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT1/DATA", split=1, device=self.device)
                self.assertIsInstance(ht_array_split1, ht.DNDarray)
                self.assertEqual(ht_array_split1.gshape, global_data_shape)
                self.assertTrue((ht_array_split1.numpy() == global_data).all())
                self.assertTrue(ht_array_split1.dtype == ht.types.canonical_heat_type(dtype))

            # test wildcard loading with dtype conversion
            with self.subTest(dtype=dtype, split="dtype_conversion"):
                # only for non-complex dtypes
                if not np.issubdtype(dtype, np.complexfloating):
                    ht_array_split0 = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT0/DATA", split=0, device=self.device, dtype=ht.float32)
                    self.assertIsInstance(ht_array_split0, ht.DNDarray)
                    self.assertEqual(ht_array_split0.gshape, global_data_shape)
                    self.assertTrue((ht_array_split0.numpy() == global_data).all())
                    self.assertTrue(ht_array_split0.dtype == ht.float32)

            ht.MPI_WORLD.Barrier()

            # Test data misconstruction when using the wrong split axis
            with self.subTest(split="split_mismatch_0", dtype=dtype):
                with self.assertRaises(ValueError):
                    test = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT1/DATA", split=0, device=self.device)
                    self.assertTrue((test.numpy() == global_data).all())

            with self.subTest(split="split_mismatch_1", dtype=dtype):
                with self.assertRaises(ValueError):
                    test = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT0/DATA", split=1, device=self.device)
                    self.assertFalse((test.numpy() == global_data).all())

            # test exceptions
            with self.subTest(split="split_exception", dtype=dtype):
                with self.assertRaises(ValueError):
                    test = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT0/DATA", split=3)
            with self.assertRaises(NotImplementedError):
                test = ht.load(self.ZARR_OUT_PATH, variable="CHUNK_*_SPLIT0/DATA", slices=slice(0,10))
            with self.assertRaises(FileNotFoundError):
                test = ht.load(self.ZARR_OUT_PATH, variable="NONEXSISTENT_CHUNK_*_SPLIT0/DATA", split=0)

            ht.MPI_WORLD.Barrier()

    def test_load_zarr_slice(self):

        import zarr

        test_data = np.arange(25).reshape(5, 5)

        if ht.MPI_WORLD.rank == 0:
            try:
                arr = zarr.create_array(
                    self.ZARR_TEMP_PATH, shape=test_data.shape, dtype=test_data.dtype
                )
            except AttributeError:
                arr = zarr.create(
                    store=self.ZARR_TEMP_PATH, shape=test_data.shape, dtype=test_data.dtype
                )
            arr[:] = test_data

        ht.MPI_WORLD.Barrier()

        slices_to_test = [
            None,
            slice(None),
            slice(1, -1),
            [None],
            [None, slice(None)],
            [None, slice(1, -1)],
            [slice(1, -1)],
            [slice(1, -1), None],
        ]

        for slices in slices_to_test:
            with self.subTest(silces=slices):
                dndarray = ht.load_zarr(self.ZARR_TEMP_PATH, slices=slices)
                dndnumpy = dndarray.numpy()

                if not isinstance(slices, Iterable):
                    slices = [slices]

                slices = tuple(
                    slice(elem) if not isinstance(elem, slice) else elem for elem in slices
                )

                if ht.MPI_WORLD.rank == 0:
                    self.assertTrue((dndnumpy == test_data[slices]).all())

                ht.MPI_WORLD.Barrier()

    def test_save_zarr_2d_split0(self):

        import zarr

        for type in self.testing_types:
            for dims in [(i, self.ZARR_SHAPE[1]) for i in range(1, max(10, ht.MPI_WORLD.size + 1))]:
                with self.subTest(type=type, dims=dims):
                    n = dims[0] * dims[1]
                    dndarray = ht.arange(0, n, dtype=type, split=0).reshape(dims)
                    ht.save_zarr(dndarray, self.ZARR_OUT_PATH, overwrite=True)
                    dndnumpy = dndarray.numpy()
                    zarr_array = zarr.open_array(self.ZARR_OUT_PATH)

                    if ht.MPI_WORLD.rank == 0:
                        self.assertTrue((dndnumpy == zarr_array).all())

                    ht.MPI_WORLD.handle.Barrier()

    def test_save_zarr_2d_split1(self):

        import zarr

        for type in self.testing_types:
            for dims in [(self.ZARR_SHAPE[0], i) for i in range(1, max(10, ht.MPI_WORLD.size + 1))]:
                with self.subTest(type=type, dims=dims):
                    n = dims[0] * dims[1]
                    dndarray = ht.arange(0, n, dtype=type).reshape(dims).resplit(axis=1)
                    ht.save_zarr(dndarray, self.ZARR_OUT_PATH, overwrite=True)
                    dndnumpy = dndarray.numpy()
                    zarr_array = zarr.open_array(self.ZARR_OUT_PATH)

                    if ht.MPI_WORLD.rank == 0:
                        self.assertTrue((dndnumpy == zarr_array).all())

                    ht.MPI_WORLD.handle.Barrier()

    def test_save_zarr_split_none(self):

        import zarr

        for type in self.testing_types:
            for n in [10, 100, 1000]:
                with self.subTest(type=type, n=n):
                    dndarray = ht.arange(n, dtype=type, split=None)
                    ht.save_zarr(dndarray, self.ZARR_OUT_PATH, overwrite=True)
                    arr = zarr.open_array(self.ZARR_OUT_PATH)
                    dndnumpy = dndarray.numpy()
                    if ht.MPI_WORLD.rank == 0:
                        self.assertTrue((dndnumpy == arr).all())

                    ht.MPI_WORLD.handle.Barrier()

    def test_save_zarr_1d_split_0(self):

        import zarr

        for type in self.testing_types:
            for n in [10, 100, 1000]:
                with self.subTest(type=type, n=n):
                    dndarray = ht.arange(n, dtype=type, split=0)
                    ht.save_zarr(dndarray, self.ZARR_OUT_PATH, overwrite=True)
                    arr = zarr.open_array(self.ZARR_OUT_PATH)
                    dndnumpy = dndarray.numpy()
                    if ht.MPI_WORLD.rank == 0:
                        self.assertTrue((dndnumpy == arr).all())

                    ht.MPI_WORLD.handle.Barrier()

    def test_load_zarr_arguments(self):

        with self.assertRaises(TypeError):
            ht.load_zarr(None)
        with self.assertRaises(ValueError):
            ht.load_zarr("data.npy")
        with self.assertRaises(ValueError):
            ht.load_zarr("", "")
        with self.assertRaises(ValueError):
            ht.load_zarr("", device=1)
        with self.assertRaises(TypeError):
            ht.load_zarr("", slices=0)
        with self.assertRaises(TypeError):
            ht.load_zarr("", slices=[0])

    def test_save_zarr_arguments(self):

        import zarr

        with self.assertRaises(TypeError):
            ht.save_zarr(None, None)
        with self.assertRaises(ValueError):
            ht.save_zarr(None, "data.npy")

        comm = ht.MPI_WORLD
        if comm.rank == 0:
            zarr.create(
                store=self.ZARR_TEMP_PATH,
                shape=(4, 4),
                dtype=ht.types.int.char(),
                overwrite=True,
            )
        comm.Barrier()

        with self.assertRaises(RuntimeError):
            ht.save_zarr(ht.arange(16).reshape((4, 4)), self.ZARR_TEMP_PATH)


if __name__ == "__main__":
    unittest.main()
