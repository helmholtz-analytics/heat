import numpy as np
import torch

import heat as ht
from .basic_test import TestCase


class TestBasicTest(TestCase):
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

        if self.get_rank() == 0:
            data = torch.arange(self.get_size(), dtype=torch.int32)
        else:
            data = torch.empty((0,), dtype=torch.int32)

        ht_array = ht.array(data, is_split=0)
        np_array = np.arange(self.get_size(), dtype=np.int32)
        self.assert_array_equal(ht_array, np_array)

    def test_assert_func_equal(self):
        # Testing with random values
        shape = (5, 3, 2, 9)

        self.assert_func_equal(shape, heat_func=ht.exp, numpy_func=np.exp, low=-100, high=100)

        self.assert_func_equal(shape, heat_func=ht.exp2, numpy_func=np.exp2, low=-100, high=100)

        # np.random.randn eventually creates values < 0 which will result in math.nan.
        # Because math.nan != math.nan this would always produce an exception.
        self.assert_func_equal(
            shape, heat_func=ht.log, numpy_func=np.log, data_types=[np.int32, np.int64], low=1
        )

        with self.assertRaises(AssertionError):
            self.assert_func_equal(shape, heat_func=ht.exp, numpy_func=np.exp2, low=-100, high=100)

        with self.assertRaises(ValueError):
            self.assert_func_equal(np.ones(shape), heat_func=np.exp, numpy_func=np.exp)

        with self.assertRaises(ValueError):
            self.assert_func_equal(
                shape,
                heat_func=ht.exp,
                numpy_func=np.exp,
                low=-100,
                high=100,
                data_types=[np.object],
            )

    def test_assert_func_equal_for_tensor(self):
        array = np.ones((self.get_size(), 20), dtype=np.int8)
        ht_func = ht.any
        np_func = np.any
        self.assert_func_equal_for_tensor(array, ht_func, np_func, distributed_result=False)

        array = np.array([[1, 2, 4, 1, 3], [1, 4, 7, 5, 1]], dtype=np.int8)
        ht_func = ht.expand_dims
        np_func = np.expand_dims
        ht_args = {"axis": 1}
        np_args = {"axis": 1}
        self.assert_func_equal_for_tensor(
            array, ht_func, np_func, heat_args=ht_args, numpy_args=np_args
        )

        array = torch.randn(15, 15)
        ht_func = ht.exp
        np_func = np.exp
        self.assert_func_equal_for_tensor(array, heat_func=ht_func, numpy_func=np_func)

        array = ht.ones((15, 15))
        with self.assertRaises(TypeError):
            self.assert_func_equal_for_tensor(array, heat_func=ht_func, numpy_func=np_func)

    def test_assertTrue_memory_layout(self):
        data = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
        data_F = ht.array(data, order="F")
        with self.assertRaises(ValueError):
            self.assertTrue_memory_layout(data_F, order="K")
