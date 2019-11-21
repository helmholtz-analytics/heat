import os
import heat as ht
import numpy as np
import torch

if os.environ.get("DEVICE") == "gpu":
    ht.use_device("gpu" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device

from heat.core.tests.test_suites.basic_test import BasicTest


class TestBasicTest(BasicTest):
    def test_assert_array_equal(self):
        heat_array = ht.ones((self.get_size(), 10, 10), dtype=ht.int32, split=1)
        np_array = np.ones((self.get_size(), 10, 10), dtype=np.int32)
        self.assert_array_equal(heat_array, np_array)

        np_array[0, 1, 1] = 0
        with self.assertRaises(AssertionError):
            self.assert_array_equal(heat_array, np_array)

        heat_array = ht.zeros((25, 13, self.get_size(), 20), dtype=ht.float32, split=2)
        expected_array = torch.zeros(
            (25, 13, self.get_size(), 20),
            dtype=torch.float32,
            device=heat_array.device.torch_device,
        )
        self.assert_array_equal(heat_array, expected_array)

    def test_assert_func_equal(self):
        array = np.ones((self.get_size(), 20), dtype=np.int8)
        ht_func = ht.any
        np_func = np.any
        self.assert_func_equal(array, ht_func, np_func, distributed_result=False)

        array = np.array([[1, 2, 4, 1, 3], [1, 4, 7, 5, 1]], dtype=np.int8)
        ht_func = ht.expand_dims
        np_func = np.expand_dims
        ht_args = {"axis": 1}
        np_args = {"axis": 1}
        self.assert_func_equal(array, ht_func, np_func, heat_args=ht_args, numpy_args=np_args)

        # Testing with random values
        shape = (5, 3, 2, 9)
        ht_func = ht.exp
        np_func = np.exp
        self.assert_func_equal(shape, heat_func=ht_func, numpy_func=np_func)

        array = torch.randn(15, 15)
        ht_func = ht.exp
        np_func = np.exp
        self.assert_func_equal(array, heat_func=ht_func, numpy_func=np_func)

        array = ht.ones((15, 15))
        with self.assertRaises(TypeError):
            self.assert_func_equal(array, heat_func=ht_func, numpy_func=np_func)
