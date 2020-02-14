import unittest
import torch
import os
import heat as ht

if os.environ.get("HEAT_USE_DEVICE") == 'cpu':
    ht.use_device("cpu")
    torch_device = ht.get_device().torch_device
    heat_device = None
elif os.environ.get("HEAT_USE_DEVICE") == 'gpu' and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
    torch_device = ht.get_device().torch_device
    heat_device = None
elif os.environ.get("HEAT_USE_DEVICE") == 'lcpu' and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
    torch_device = ht.cpu.torch_device
    heat_device = ht.cpu
elif os.environ.get("HEAT_USE_DEVICE") == 'lgpu' and torch.cuda.is_available():
    ht.use_device("cpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
    torch_device = ht.cpu.torch_device
    heat_device = ht.cpu


class TestOperations(unittest.TestCase):
    def test___binary_bit_op_broadcast(self):

        # broadcast without split
        left_tensor = ht.ones((4, 1), dtype=ht.int32, device=heat_device)
        right_tensor = ht.ones((1, 2), dtype=ht.int32, device=heat_device)
        result = left_tensor & right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor & left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=0 for both operants
        left_tensor = ht.ones((4, 1), split=0, dtype=ht.int32, device=heat_device)
        right_tensor = ht.ones((1, 2), split=0, dtype=ht.int32, device=heat_device)
        result = left_tensor | right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor | left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=1 for both operants
        left_tensor = ht.ones((4, 1), split=1, dtype=ht.int32, device=heat_device)
        right_tensor = ht.ones((1, 2), split=1, dtype=ht.int32, device=heat_device)
        result = left_tensor ^ right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor ^ left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=1 for second operant
        left_tensor = ht.ones((4, 1), dtype=ht.int32, device=heat_device)
        right_tensor = ht.ones((1, 2), split=1, dtype=ht.int32, device=heat_device)
        result = left_tensor & right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor & left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=0 for first operant
        left_tensor = ht.ones((4, 1), split=0, dtype=ht.int32, device=heat_device)
        right_tensor = ht.ones((1, 2), dtype=ht.int32, device=heat_device)
        result = left_tensor | right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor | left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with unequal dimensions and one splitted tensor
        left_tensor = ht.ones((2, 4, 1), split=0, dtype=ht.int32, device=heat_device)
        right_tensor = ht.ones((1, 2), dtype=ht.int32, device=heat_device)
        result = left_tensor ^ right_tensor
        self.assertEqual(result.shape, (2, 4, 2))
        result = right_tensor ^ left_tensor
        self.assertEqual(result.shape, (2, 4, 2))

        # broadcast with unequal dimensions, a scalar, and one splitted tensor
        left_scalar = ht.np.int32(1)
        right_tensor = ht.ones((1, 2), split=0, dtype=ht.int32, device=heat_device)
        result = ht.bitwise_or(left_scalar, right_tensor)
        self.assertEqual(result.shape, (1, 2))
        result = right_tensor | left_scalar
        self.assertEqual(result.shape, (1, 2))

        # broadcast with unequal dimensions and two splitted tensors
        left_tensor = ht.ones((4, 1, 3, 1, 2), split=0, dtype=torch.uint8, device=heat_device)
        right_tensor = ht.ones((1, 3, 1), split=0, dtype=torch.uint8, device=heat_device)
        result = left_tensor & right_tensor
        self.assertEqual(result.shape, (4, 1, 3, 3, 2))
        result = right_tensor & left_tensor
        self.assertEqual(result.shape, (4, 1, 3, 3, 2))

        with self.assertRaises(TypeError):
            ht.bitwise_and(ht.ones((1, 2), device=heat_device), "wrong type")
        with self.assertRaises(NotImplementedError):
            ht.bitwise_or(
                ht.ones((1, 2), dtype=ht.int32, split=0, device=heat_device),
                ht.ones((1, 2), dtype=ht.int32, split=1, device=heat_device),
            )
