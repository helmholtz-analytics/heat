"""Tests for the configuration and optional-dependency handling in ``heat.core._config``."""

import os
import sys
import tempfile
import unittest

from heat.core import _config

from heat.testing.basic_test import TestCase


class TestConfig(TestCase):
    """Tests for the optional-dependency probes."""

    def setUp(self):
        self._clear_caches()

    def tearDown(self):
        self._clear_caches()

    @staticmethod
    def _clear_caches():
        _config.is_installed.cache_clear()
        _config._import_optional.cache_clear()

    def test_is_installed(self):
        self.assertTrue(_config.is_installed("torch"))
        self.assertFalse(_config.is_installed("heat_no_such_module_xyz"))

    def test_has_dependency(self):
        self.assertTrue(_config.has_dependency("torch"))
        self.assertFalse(_config.has_dependency("heat_no_such_module_xyz"))

    def test_optional_import(self):
        import torch

        self.assertIs(_config.optional_import("torch"), torch)
        self.assertIsNone(_config.optional_import("heat_no_such_module_xyz"))

    def test_optional_import_is_cached(self):
        first = _config.optional_import("torch")
        self.assertIs(first, _config.optional_import("torch"))
        self.assertGreater(_config._import_optional.cache_info().hits, 0)

    def test_require_dependency(self):
        import torch

        self.assertIs(_config.require_dependency("torch"), torch)

    def test_require_dependency_raises_with_pip_hint(self):
        # every optional dependency Heat knows about must name its extra in the message
        for module, extra in _config.OPTIONAL_DEPENDENCIES.items():
            if _config.has_dependency(module):
                continue
            with self.assertRaises(RuntimeError) as caught:
                _config.require_dependency(module)
            self.assertIn(f"pip install heat[{extra}]", str(caught.exception))

    def test_require_dependency_raises_without_extra(self):
        with self.assertRaises(RuntimeError) as caught:
            _config.require_dependency("heat_no_such_module_xyz")
        self.assertIn("heat_no_such_module_xyz", str(caught.exception))
        self.assertNotIn("pip install", str(caught.exception))

    def test_broken_install_does_not_propagate(self):
        """A module that is present but raises on import must degrade, not explode.

        Regression test: ``zarr`` used to be guarded by ``except ModuleNotFoundError``,
        so a broken-but-installed ``zarr`` made ``import heat`` fail outright.
        """
        name = "heat_broken_module_xyz"
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, f"{name}.py"), "w") as handle:
                handle.write("raise OSError('broken shared library')\n")
            sys.path.insert(0, directory)
            try:
                self._clear_caches()
                self.assertTrue(_config.is_installed(name))
                self.assertIsNone(_config.optional_import(name))
                self.assertFalse(_config.has_dependency(name))
                with self.assertRaises(RuntimeError) as caught:
                    _config.require_dependency(name)
                self.assertIn("OSError", str(caught.exception))
            finally:
                sys.path.remove(directory)
                sys.modules.pop(name, None)
                self._clear_caches()

    def test_capability_flags(self):
        self.assertIsInstance(_config.hdf5_has_mpi(), bool)
        self.assertIsInstance(_config.netcdf_has_parallel(), bool)
        if not _config.has_dependency("h5py"):
            self.assertFalse(_config.hdf5_has_mpi())
        if not _config.has_dependency("netCDF4"):
            self.assertFalse(_config.netcdf_has_parallel())


if __name__ == "__main__":
    unittest.main()
