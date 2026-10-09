"""Tests for the public surface and generic dispatch of ``heat.core.io``."""

import os
import subprocess
import sys
import unittest

import heat as ht
from heat.testing.basic_test import TestCase


_NO_OPTIONAL_DEPS_PROBE = """
import sys

BLOCKED = {"h5py", "netCDF4", "zarr", "pandas"}


class Blocker:
    # Hide the optional dependencies, but only from Heat's own availability probe:
    # torch calls find_spec("pandas") while importing, and blocking that breaks the
    # import for reasons unrelated to what is being tested.
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in BLOCKED:
            return None
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename.endswith("heat/core/_config.py"):
                raise ModuleNotFoundError(name, name=name)
            frame = frame.f_back
        return None


sys.meta_path.insert(0, Blocker())

import heat as ht

assert ht.io.available_formats() == {
    "csv": True, "hdf5": False, "netcdf": False, "zarr": False
}, ht.io.available_formats()

for name in ("load_hdf5", "save_hdf5", "load_multiple_hdf5", "load_netcdf", "save_netcdf",
             "load_zarr", "save_zarr", "load_csv_from_folder"):
    assert not hasattr(ht, name), f"ht.{name} should not exist without its dependency"

for method in ("save_hdf5", "save_netcdf", "save_zarr"):
    assert not hasattr(ht.DNDarray, method), method

for path, extra in (("x.h5", "hdf5"), ("x.nc", "netcdf"), ("x.zarr", "zarr")):
    try:
        ht.load(path)
    except RuntimeError as error:
        assert f"pip install heat[{extra}]" in str(error), str(error)
    else:
        raise AssertionError(f"load({path}) should have raised RuntimeError")

try:
    ht.load("x.json")
except ValueError:
    pass
else:
    raise AssertionError("unknown extension should raise ValueError")

# csv needs no optional dependency and must still work end to end
import os, tempfile
with tempfile.TemporaryDirectory() as directory:
    target = os.path.join(directory, "roundtrip.csv")
    data = ht.arange(12, dtype=ht.float32).reshape((3, 4))
    ht.save(data, target)
    assert ht.equal(data, ht.load(target))

print("ok")
"""


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
        if ht.io.supports("pandas"):
            expected.add("load_csv_from_folder")
        if ht.io.supports("hdf5"):
            expected |= {"load_hdf5", "save_hdf5", "load_multiple_hdf5"}
        if ht.io.supports("netcdf"):
            expected |= {"load_netcdf", "save_netcdf"}
        if ht.io.supports("zarr"):
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
        if ht.io.supports("hdf5"):
            self.assertTrue(hasattr(ht.DNDarray, "save_hdf5"))
        if ht.io.supports("netcdf"):
            self.assertTrue(hasattr(ht.DNDarray, "save_netcdf"))

    def test_dndarray_save_hdf5_forwards_dtype(self):
        """`DNDarray.save_hdf5` used to accept `dtype` and silently drop it."""
        if not ht.io.supports("hdf5"):
            self.skipTest("Requires HDF5")
        import h5py

        path = os.path.join(os.getcwd(), "test_save_hdf5_dtype.h5")
        data = ht.arange(8, split=0, dtype=ht.int32)
        data.save_hdf5(path, "data", dtype=ht.float64)
        ht.MPI_WORLD.Barrier()
        with h5py.File(path, "r") as handle:
            self.assertEqual(handle["data"].dtype, "float64")
        ht.MPI_WORLD.Barrier()
        if ht.MPI_WORLD.rank == 0:
            os.remove(path)
        ht.MPI_WORLD.Barrier()

    def test_dndarray_save_methods_cover_every_writable_format(self):
        """Every format with a saver gets a `DNDarray.save_<format>` method."""
        for name, spec in ht.io.registered_formats().items():
            if spec.saver is not None:
                with self.subTest(format=name):
                    self.assertTrue(hasattr(ht.DNDarray, f"save_{name}"))

    def test_behaviour_without_any_optional_dependency(self):
        """Heat must degrade cleanly when h5py/netCDF4/zarr/pandas are all absent.

        CI installs `.[dev]`, which pulls in every optional dependency, so this is the
        only coverage these fallbacks get.
        """
        if ht.MPI_WORLD.size > 1:
            # spawning a nested MPI process from inside an mpirun job deadlocks
            self.skipTest("runs on a single process only")
        result = subprocess.run(
            [sys.executable, "-c", _NO_OPTIONAL_DEPS_PROBE], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ok", result.stdout)

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
