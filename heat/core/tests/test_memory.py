import unittest

import heat as ht


class TestMemory(unittest.TestCase):
    def test_copy(self):
        tensor = ht.ones(5)
        copied = tensor.copy()

        # test identity inequality and value equality
        self.assertIsNot(tensor, copied)
        self.assertIsNot(tensor._Tensor__array, copied._Tensor__array)
        self.assertTrue((tensor == copied)._Tensor__array.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.copy('hello world')
