import unittest

import heat as ht


class TestDevices(unittest.TestCase):
    def test_get_default_device(self):
        self.assertIs(ht.get_backend(), ht.cpu)

    def test_sanitize_device(self):
        self.assertIs(ht.sanitize_device('cpu'), ht.cpu)
        self.assertIs(ht.sanitize_device('cPu'), ht.cpu)
        self.assertIs(ht.sanitize_device('  CPU  '), ht.cpu)
        self.assertIs(ht.sanitize_device(ht.cpu), ht.cpu)
        self.assertIs(ht.sanitize_device(None), ht.cpu)

        with self.assertRaises(ValueError):
            self.assertIs(ht.sanitize_device('fpu'), ht.cpu)
        with self.assertRaises(ValueError):
            self.assertIs(ht.sanitize_device(1), ht.cpu)

    def test_set_default_device(self):
        ht.use_backend('cpu')
        self.assertIs(ht.get_backend(), ht.cpu)
        ht.use_backend(ht.cpu)
        self.assertIs(ht.get_backend(), ht.cpu)
        ht.use_backend(None)
        self.assertIs(ht.get_backend(), ht.cpu)

        with self.assertRaises(ValueError):
            ht.use_backend('fpu')
        with self.assertRaises(ValueError):
            ht.use_backend(1)
