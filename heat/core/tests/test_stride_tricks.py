import unittest
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
    heat_device = ht.gpu


class TestStrideTricks(unittest.TestCase):
    def test_broadcast_shape(self):
        self.assertEqual(ht.core.stride_tricks.broadcast_shape((5, 4), (4,)), (5, 4))
        self.assertEqual(
            ht.core.stride_tricks.broadcast_shape((1, 100, 1), (10, 1, 5)), (10, 100, 5)
        )
        self.assertEqual(
            ht.core.stride_tricks.broadcast_shape((8, 1, 6, 1), (7, 1, 5)), (8, 7, 6, 5)
        )

        # invalid value ranges
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.broadcast_shape((5, 4), (5,))
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.broadcast_shape((5, 4), (2, 3))
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.broadcast_shape((5, 2), (5, 2, 3))
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.broadcast_shape((2, 1), (8, 4, 3))

    def test_sanitize_axis(self):
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4, 4), 1), 1)
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4, 4), -1), 2)
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4, 4), 2), 2)
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4, 4), (0, 1)), (0, 1))
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4, 4), (-2, -3)), (1, 0))
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4), 0), 0)
        self.assertEqual(ht.core.stride_tricks.sanitize_axis((5, 4), None), None)

        # invalid types
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_axis((5, 4), 1.0)
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_axis((5, 4), "axis")

        # invalid value ranges
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_axis((5, 4), 2)
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_axis((5, 4), -3)
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_axis((5, 4, 4), (-4, 1))

    def test_sanitize_shape(self):
        # valid integers and iterables
        self.assertEqual(ht.core.stride_tricks.sanitize_shape(1), (1,))
        self.assertEqual(ht.core.stride_tricks.sanitize_shape([1, 2]), (1, 2))
        self.assertEqual(ht.core.stride_tricks.sanitize_shape((1, 2)), (1, 2))

        # invalid value ranges
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_shape(-1)
        with self.assertRaises(ValueError):
            ht.core.stride_tricks.sanitize_shape((2, -1))

        # invalid types
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_shape("shape")
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_shape(1.0)
        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_shape((1, 1.0))
