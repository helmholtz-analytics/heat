import heat as ht
import unittest


class TestNN(unittest.TestCase):
    def test_nn_getattr(self):
        with self.assertRaises(AttributeError):
            ht.nn.asdf()


class TestFunctional(unittest.TestCase):
    def test_functional_getattr(self):
        with self.assertRaises(AttributeError):
            ht.nn.functional.asdf()
