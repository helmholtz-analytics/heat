import unittest

import heat as ht


class TestDevices(unittest.TestCase):
    def test_get_default_device(self):
        self.assertIs(ht.get_default_device(), ht.cpu)

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
        ht.set_default_device('cpu')
        self.assertIs(ht.get_default_device(), ht.cpu)
        ht.set_default_device(ht.cpu)
        self.assertIs(ht.get_default_device(), ht.cpu)
        ht.set_default_device(None)
        self.assertIs(ht.get_default_device(), ht.cpu)

        with self.assertRaises(ValueError):
            ht.set_default_device('fpu')
        with self.assertRaises(ValueError):
            ht.set_default_device(1)
