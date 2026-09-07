"""Tests for the public surface and generic dispatch of ``heat.core.io``."""

import subprocess
import sys
import unittest

import heat as ht
from heat.testing.basic_test import TestCase


class TestIOSurface(TestCase):
    """Guards the names ``heat.core.io`` exports and what importing it costs."""

    @staticmethod
    def _expected_exports():
        expected = {
            "load",
            "save",
            "load_csv",
            "save_csv",
            "supports_pandas",
            "load_npy_from_path",
            "supports_hdf5",
            "supports_netcdf",
            "supports_zarr",
        }
        if ht.io.supports_pandas():
            expected.add("load_csv_from_folder")
        if ht.io.supports_hdf5():
            expected |= {"load_hdf5", "save_hdf5", "load_multiple_hdf5"}
        if ht.io.supports_netcdf():
            expected |= {"load_netcdf", "save_netcdf"}
        if ht.io.supports_zarr():
            expected |= {"load_zarr", "save_zarr"}
        return expected

    def test_all_is_exactly_the_expected_set(self):
        self.assertEqual(set(ht.io.__all__), self._expected_exports())
        self.assertEqual(len(ht.io.__all__), len(set(ht.io.__all__)), "duplicate in __all__")

    def test_every_export_reaches_the_heat_namespace(self):
        for name in ht.io.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(ht, name), f"ht.{name} is missing")

    def test_non_exported_names_remain_reachable(self):
        # never in __all__, but part of the module's de facto surface
        self.assertTrue(callable(ht.io.size_from_slice))

    def test_dndarray_save_methods(self):
        self.assertTrue(hasattr(ht.DNDarray, "save"))
        if ht.io.supports_hdf5():
            self.assertTrue(hasattr(ht.DNDarray, "save_hdf5"))
        if ht.io.supports_netcdf():
            self.assertTrue(hasattr(ht.DNDarray, "save_netcdf"))

    def test_import_heat_does_not_import_optional_dependencies(self):
        """`import heat` must not pay for h5py, netCDF4, zarr or pandas.

        These are only needed by one file format each, and are resolved on first use.
        """
        if ht.MPI_WORLD.size > 1:
            # spawning a nested MPI process from inside an mpirun job deadlocks
            self.skipTest("runs on a single process only")
        script = (
            "import heat, sys; "
            "print([m for m in ('h5py', 'netCDF4', 'zarr', 'pandas') if m in sys.modules])"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
