import unittest
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu" and ht.torch.cuda.is_available():
    ht.use_device("gpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and ht.torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    ht.torch.cuda.set_device(device)


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
        self.assertEqual(ht.core.stride_tricks.sanitize_axis(tuple(), 0), None)

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

    def test_sanitize_slice(self):
        test_slice = slice(None, None, None)
        ret_slice = ht.core.stride_tricks.sanitize_slice(test_slice, 100)
        self.assertEqual(ret_slice.start, 0)
        self.assertEqual(ret_slice.stop, 100)
        self.assertEqual(ret_slice.step, 1)
        test_slice = slice(-50, -5, 2)
        ret_slice = ht.core.stride_tricks.sanitize_slice(test_slice, 100)
        self.assertEqual(ret_slice.start, 50)
        self.assertEqual(ret_slice.stop, 95)
        self.assertEqual(ret_slice.step, 2)

        with self.assertRaises(TypeError):
            ht.core.stride_tricks.sanitize_slice("test_slice", 100)
