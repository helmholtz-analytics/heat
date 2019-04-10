import unittest
import torch

import heat as ht

FLOAT_EPSILON = 1e-4


class TestOperations(unittest.TestCase):
    def test___binary_op_broadcast(self):
        left_tensor = ht.ones((4, 1), split=0) 
        right_tensor = ht.ones((1, 2), split=0)
        result = left_tensor + right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor + left_tensor
        self.assertEqual(result.shape, (4, 2))

        left_tensor = ht.ones((4, 1), split=1) 
        right_tensor = ht.ones((1, 2), split=1)
        result = left_tensor + right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor + left_tensor
        self.assertEqual(result.shape, (4, 2))

        left_tensor = ht.ones((4, 1, 3, 1, 2), split=0, dtype=torch.uint8) 
        right_tensor = ht.ones((1, 2, 1, 3, 1), split=0, dtype=torch.uint8)
        result = left_tensor + right_tensor
        self.assertEqual(result.shape, (4, 2, 3, 3, 2))
        result = right_tensor + left_tensor
        self.assertEqual(result.shape, (4, 2, 3, 3, 2))
