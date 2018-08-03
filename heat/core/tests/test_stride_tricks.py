import unittest

import heat as ht


class TestStrideTricks(unittest.TestCase):
    def test_sanitize_shape(self):
        # valid integers and iterables
        self.assertEqual(ht.core.stride_tricks.sanitize_shape(1), (1,))
        self.assertEqual(ht.core.stride_tricks.sanitize_shape([1, 2]), (1, 2,))
        self.assertEqual(ht.core.stride_tricks.sanitize_shape((1, 2,)), (1, 2,))

        # invalid value ranges
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_shape(0)
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_shape(-1)
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_shape((2, -1,))

        # invalid types
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_shape('shape')
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_shape(1.0)
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_shape((1, 1.0,))
