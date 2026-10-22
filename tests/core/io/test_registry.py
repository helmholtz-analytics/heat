"""Tests for the file-format registry backing ``ht.load`` / ``ht.save`` dispatch."""

import unittest
import warnings

import heat as ht
from heat.core.io import _registry
from heat.testing.basic_test import TestCase


class TestRegistry(TestCase):
    """Tests for :mod:`heat.core.io._registry`."""

    def setUp(self):
        self._saved = dict(_registry._REGISTRY)

    def tearDown(self):
        _registry._REGISTRY.clear()
        _registry._REGISTRY.update(self._saved)

    def test_builtin_formats_are_registered(self):
        formats = _registry.registered_formats()
        for name in ("csv", "hdf5", "netcdf", "zarr"):
            self.assertIn(name, formats)
        self.assertIn(".csv", _registry.registered_extensions())
        self.assertIn(".h5", _registry.registered_extensions())

    def test_no_extension_is_claimed_twice(self):
        seen = {}
        for spec in _registry.registered_formats().values():
            for extension in spec.extensions:
                self.assertNotIn(
                    extension,
                    seen,
                    f"{extension} claimed by both {seen.get(extension)} and {spec.name}",
                )
                seen[extension] = spec.name

    def test_register_rejects_conflicting_extension(self):
        with self.assertRaises(ValueError):
            _registry.register_format("conflicting", {".csv"})

    def test_register_allows_reregistering_same_format(self):
        spec = _registry.register_format("csv", {".csv"}, loader=lambda path: None)
        self.assertEqual(spec.name, "csv")

    def test_supports(self):
        for name in ("csv", "hdf5", "netcdf", "zarr"):
            with self.subTest(name=name):
                self.assertIsInstance(_registry.supports(name), bool)
        # optional dependencies are addressable by import name too
        self.assertIsInstance(_registry.supports("pandas"), bool)
        with self.assertRaises(ValueError):
            _registry.supports("not_a_format_or_dependency")

    def test_available_formats(self):
        formats = ht.io.available_formats()
        self.assertEqual(set(formats), set(_registry.registered_formats()))
        for name, usable in formats.items():
            self.assertEqual(usable, ht.io.supports(name), name)

    def test_supports_agrees_with_the_deprecated_helpers(self):
        pairs = [
            ("hdf5", ht.io.supports_hdf5),
            ("netcdf", ht.io.supports_netcdf),
            ("zarr", ht.io.supports_zarr),
            ("pandas", ht.io.supports_pandas),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for name, deprecated in pairs:
                with self.subTest(name=name):
                    self.assertEqual(ht.io.supports(name), deprecated())

    def test_deprecated_helpers_warn(self):
        for deprecated in (
            ht.io.supports_hdf5,
            ht.io.supports_netcdf,
            ht.io.supports_zarr,
            ht.io.supports_pandas,
        ):
            with self.subTest(function=deprecated.__name__):
                with self.assertWarns(DeprecationWarning):
                    deprecated()

    def test_unknown_extension_raises_value_error(self):
        with self.assertRaises(ValueError):
            _registry.loader_for_path("data.unknownext")
        with self.assertRaises(ValueError):
            _registry.saver_for_path("data.unknownext")
        with self.assertRaises(ValueError):
            ht.load("data.unknownext")

    def test_missing_dependency_raises_runtime_error(self):
        """A known extension whose library is absent must report the dependency."""
        _registry.register_format(
            "heat_fake_format", {".heatfake"}, dependency="heat_no_such_module_xyz"
        )
        for resolve in (_registry.loader_for_path, _registry.saver_for_path):
            with self.assertRaises(RuntimeError) as caught:
                resolve("data.heatfake")
            self.assertIn("heat_no_such_module_xyz", str(caught.exception))

    def test_unsupported_direction_raises_value_error(self):
        """A format that is readable but not writable must say so."""
        _registry.register_format(
            "heat_readonly_format", {".heatreadonly"}, loader=lambda path: None
        )
        self.assertIsNotNone(_registry.loader_for_path("data.heatreadonly"))
        with self.assertRaises(ValueError) as caught:
            _registry.saver_for_path("data.heatreadonly")
        self.assertIn("does not support saving", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
