"""Tests for ``heat.core.io.zarr`` behaviour that is specific to this module."""

import os
import unittest

import numpy as np

import heat as ht
from heat.testing.basic_test import TestCase


@unittest.skipUnless(ht.io.supports_zarr(), "Requires zarr")
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


if __name__ == "__main__":
    unittest.main()
