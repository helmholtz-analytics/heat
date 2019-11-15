import unittest
import os
import heat as ht

ht.use_device(os.environ.get("DEVICE"))


class TestDevices(unittest.TestCase):
    def test_get_default_device(self):
        if os.environ.get("DEVICE") == "gpu":
            ht.use_device(os.environ.get("DEVICE"))
            self.assertIs(ht.get_device(), ht.gpu)
        else:
            self.assertIs(ht.get_device(), ht.cpu)

    def test_sanitize_device(self):
        if os.environ.get("DEVICE") == "gpu":
            ht.use_device(os.environ.get("DEVICE"))
            self.assertIs(ht.sanitize_device("gpu"), ht.gpu)
            self.assertIs(ht.sanitize_device("gPu"), ht.gpu)
            self.assertIs(ht.sanitize_device("  GPU  "), ht.gpu)
            self.assertIs(ht.sanitize_device(ht.gpu), ht.gpu)
            self.assertIs(ht.sanitize_device(None), ht.gpu)
        else:
            self.assertIs(ht.sanitize_device("cpu"), ht.cpu)
            self.assertIs(ht.sanitize_device("cPu"), ht.cpu)
            self.assertIs(ht.sanitize_device("  CPU  "), ht.cpu)
            self.assertIs(ht.sanitize_device(ht.cpu), ht.cpu)
            self.assertIs(ht.sanitize_device(None), ht.cpu)

        with self.assertRaises(ValueError):
            self.assertIs(ht.sanitize_device("fpu"), ht.cpu)
        with self.assertRaises(ValueError):
            self.assertIs(ht.sanitize_device(1), ht.cpu)

    def test_set_default_device(self):
        if os.environ.get("DEVICE") == "gpu":
            ht.use_device("gpu")
            self.assertIs(ht.get_device(), ht.gpu)
            ht.use_device(ht.gpu)
            self.assertIs(ht.get_device(), ht.gpu)
            ht.use_device(None)
            self.assertIs(ht.get_device(), ht.gpu)
        else:
            ht.use_device("cpu")
            self.assertIs(ht.get_device(), ht.cpu)
            ht.use_device(ht.cpu)
            self.assertIs(ht.get_device(), ht.cpu)
            ht.use_device(None)
            self.assertIs(ht.get_device(), ht.cpu)

        with self.assertRaises(ValueError):
            ht.use_device("fpu")
        with self.assertRaises(ValueError):
            ht.use_device(1)
