import unittest

import heat as ht


class TestTensor(unittest.TestCase):
    def test_randn(self):
        # scalar input
        simple_randn_float = ht.random.randn(3)
        self.assertIsInstance(simple_randn_float, ht.tensor)
        self.assertEqual(simple_randn_float.shape, (3,))
        self.assertEqual(simple_randn_float.lshape, (3,))
        self.assertEqual(simple_randn_float.split, None)
        self.assertEqual(simple_randn_float.dtype, ht.float32)

        # multi-dimensional
        elaborate_randn_float = ht.random.randn(2, 3)
        self.assertIsInstance(elaborate_randn_float, ht.tensor)
        self.assertEqual(elaborate_randn_float.shape, (2, 3))
        self.assertEqual(elaborate_randn_float.lshape, (2, 3))
        self.assertEqual(elaborate_randn_float.split, None)
        self.assertEqual(elaborate_randn_float.dtype, ht.float32)

        # exceptions
        with self.assertRaises(TypeError):
            ht.random.randn('(2, 3,)')
        with self.assertRaises(ValueError):
            ht.random.randn(-1, 3)
