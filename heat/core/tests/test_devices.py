import unittest
import os
import heat as ht

ht_device, torch_device, envar = ht.use_envar_device()


class TestDevices(unittest.TestCase):
    @unittest.skipIf(envar not in ["cpu", "lgpu"], "only supported for cpu")
    def test_get_default_device_cpu(self):
        self.assertIs(ht.get_device(), ht.cpu)

    @unittest.skipIf(envar not in ["gpu", "lcpu"], "only supported for gpu")
    def test_get_default_device_gpu(self):
        if ht.torch.cuda.is_available():
            self.assertIs(ht.get_device(), ht.gpu)

    @unittest.skipIf(envar not in ["cpu", "lgpu"], "only supported for cpu")
    def test_sanitize_device_cpu(self):
        self.assertIs(ht.sanitize_device("cpu"), ht.cpu)
        self.assertIs(ht.sanitize_device("cPu"), ht.cpu)
        self.assertIs(ht.sanitize_device("  CPU  "), ht.cpu)
        self.assertIs(ht.sanitize_device(ht.cpu), ht.cpu)
        self.assertIs(ht.sanitize_device(None), ht.cpu)

        with self.assertRaises(ValueError):
            self.assertIs(ht.sanitize_device("fpu"), ht.cpu)
        with self.assertRaises(ValueError):
            self.assertIs(ht.sanitize_device(1), ht.cpu)

    @unittest.skipIf(envar not in ["gpu", "lcpu"], "only supported for gpu")
    def test_sanitize_device_gpu(self):
        if ht.torch.cuda.is_available():
            self.assertIs(ht.sanitize_device("gpu"), ht.gpu)
            self.assertIs(ht.sanitize_device("gPu"), ht.gpu)
            self.assertIs(ht.sanitize_device("  GPU  "), ht.gpu)
            self.assertIs(ht.sanitize_device(ht.gpu), ht.gpu)
            self.assertIs(ht.sanitize_device(None), ht.gpu)

            with self.assertRaises(ValueError):
                self.assertIs(ht.sanitize_device("fpu"), ht.gpu)
            with self.assertRaises(ValueError):
                self.assertIs(ht.sanitize_device(1), ht.gpu)

    @unittest.skipIf(envar not in ["cpu", "lgpu"], "only supported for cpu")
    def test_set_default_device_cpu(self):
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

    @unittest.skipIf(envar not in ["gpu", "lcpu"], "only supported for gpu")
    def test_set_default_device_gpu(self):
        if ht.torch.cuda.is_available():
            ht.use_device("gpu")
            self.assertIs(ht.get_device(), ht.gpu)
            ht.use_device(ht.gpu)
            self.assertIs(ht.get_device(), ht.gpu)
            ht.use_device(None)
            self.assertIs(ht.get_device(), ht.gpu)

        with self.assertRaises(ValueError):
            ht.use_device("fpu")
        with self.assertRaises(ValueError):
            ht.use_device(1)
