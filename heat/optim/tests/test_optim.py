import heat as ht
import unittest


class TestOptim(unittest.TestCase):
    def test_optim_getattr(self):
        with self.assertRaises(AttributeError):
            ht.optim.asdf()
