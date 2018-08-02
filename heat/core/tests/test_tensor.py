import torch
import unittest

import heat as ht


class TestTensor(unittest.TestCase):
    def test_astype(self):
        data = ht.float32([
            [1, 2, 3],
            [4, 5, 6]
        ])

        # check starting invariant
        self.assertEqual(data.dtype, ht.float32)

        # check the copy case for uint8
        as_uint8 = data.astype(ht.uint8)
        self.assertIsInstance(as_uint8, ht.tensor)
        self.assertEqual(as_uint8.dtype, ht.uint8)
        self.assertEqual(as_uint8._tensor__array.dtype, torch.uint8)
        self.assertIsNot(as_uint8, data)

        # check the copy case for uint8
        as_float64 = data.astype(ht.float64, copy=False)
        self.assertIsInstance(as_float64, ht.tensor)
        self.assertEqual(as_float64.dtype, ht.float64)
        self.assertEqual(as_float64._tensor__array.dtype, torch.float64)
        self.assertIs(as_float64, data)
