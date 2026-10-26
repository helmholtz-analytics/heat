from unittest import mock
import pytest

import heat as ht
from heat.testing.basic_test import TestCase
from heat._array_api_inspection import __array_namespace_info__


class TestArrayApiInspection(TestCase):
    def setUp(self):
        self.info = __array_namespace_info__()

    def test_capabilities(self):
        caps = self.info.capabilities()
        self.assertIsInstance(caps, dict)
        self.assertTrue(caps["boolean indexing"])
        self.assertTrue(caps["data-dependent shapes"])
        self.assertEqual(caps["max dimensions"], 64)

    def test_default_device(self):
        dev = self.info.default_device()
        self.assertEqual(dev, ht.get_device())

    def test_default_dtypes(self):
        # Default device
        dtypes = self.info.default_dtypes()
        self.assertIn("real floating", dtypes)
        self.assertIn("complex floating", dtypes)
        self.assertIn("integral", dtypes)
        self.assertIn("indexing", dtypes)
        self.assertEqual(dtypes["real floating"], ht.float32)
        self.assertEqual(dtypes["integral"], ht.int64)

        # Explicit CPU device
        dtypes_cpu = self.info.default_dtypes(device=ht.cpu)
        self.assertEqual(dtypes_cpu, dtypes)

        # Invalid device type
        with self.assertRaises(ValueError):
            self.info.default_dtypes(device="cpu")

        # Mocked MPS device branch
        mock_mps = mock.MagicMock(spec=ht.Device)
        mock_mps.torch_device = "mps:0"
        dtypes_mps = self.info.default_dtypes(device=mock_mps)
        self.assertEqual(dtypes_mps["integral"], ht.int32)
        self.assertEqual(dtypes_mps["indexing"], ht.int32)

    def test_devices(self):
        devs = self.info.devices()
        self.assertIsInstance(devs, tuple)
        self.assertIn(ht.cpu, devs)

        # Check branch when ht_devices has gpu attribute
        if hasattr(ht.core.devices, "gpu"):
            self.assertIn(ht.core.devices.gpu, devs)

    def test_dtypes(self):
        # Default device, kind=None
        all_dtypes = self.info.dtypes()
        expected_kinds = [
            "bool",
            "signed integer",
            "unsigned integer",
            "integral",
            "real floating",
            "complex floating",
            "numeric",
        ]
        for kind in expected_kinds:
            res = self.info.dtypes(kind=kind)
            self.assertIsInstance(res, dict)
            for k, val in res.items():
                self.assertIn(k, all_dtypes)
                self.assertEqual(val, all_dtypes[k])

        # Tuple of kinds
        res_tuple = self.info.dtypes(kind=("bool", "real floating"))
        self.assertIn("bool", res_tuple)
        self.assertIn("float32", res_tuple)
        self.assertIn("float64", res_tuple)

        # Invalid device
        with self.assertRaises(ValueError):
            self.info.dtypes(device="invalid_device")

        # Unsupported kind
        with self.assertRaises(ValueError):
            self.info.dtypes(kind="unsupported_kind")

        # Mocked MPS device branch for all kinds
        mock_mps = mock.MagicMock(spec=ht.Device)
        mock_mps.torch_device = "mps:0"

        mps_all = self.info.dtypes(device=mock_mps)
        self.assertNotIn("float64", mps_all)
        self.assertNotIn("int64", mps_all)

        for kind in expected_kinds:
            res_mps = self.info.dtypes(device=mock_mps, kind=kind)
            self.assertIsInstance(res_mps, dict)
