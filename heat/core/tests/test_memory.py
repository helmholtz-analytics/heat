import unittest
import os
import heat as ht

if os.environ.get("DEVICE") == "gpu":
    ht.use_device("gpu" if ht.torch.cuda.is_available() else "cpu")
    ht.torch.cuda.set_device(ht.torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device


class TestMemory(unittest.TestCase):
    def test_copy(self):
        tensor = ht.ones(5)
        copied = tensor.copy()

        # test identity inequality and value equality
        self.assertIsNot(tensor, copied)
        self.assertIsNot(tensor._DNDarray__array, copied._DNDarray__array)
        self.assertTrue((tensor == copied)._DNDarray__array.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.copy("hello world")
