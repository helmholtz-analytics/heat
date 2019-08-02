import heat as ht
import numpy as np

from heat.core.tests.test_suites.basic_test import BasicTest


class TestBasicTest(BasicTest):

    def test_assert_array_equal(self):
        heat_array = ht.ones((self.get_size(), 10, 10), dtype=ht.int32, split=1)
        np_array = np.ones((self.get_size(), 10, 10), dtype=np.int32)
        self.assert_array_equal(heat_array, np_array)

        np_array[0, 1, 1] = 0
        with self.assertRaises(AssertionError):
            self.assert_array_equal(heat_array, np_array)

    def test_assert_func_equal(self):
        # array = np.ones((self.get_size(), 20), dtype=np.int8)
        # ht_func = ht.any
        # np_func = np.any
        # self.assert_func_equal(array, ht_func, np_func, distributed_result=False)
        #
        # array = np.array([[1, 2, 4, 1, 3], [1, 4, 7, 5, 1]], dtype=np.int8)
        # ht_func = ht.unique
        # np_func = np.unique
        # ht_args = {'sorted': True, 'axis': 0}
        # np_args = {'axis': 0}
        # self.assert_func_equal(array, ht_func, np_func, heat_args=ht_args, numpy_args=np_args)

        # Testing with random values
        a = ht.ones((4, ), split=0)
        a = a.exp()
        print('shape', a.gshape)
        shape = (5, )
        ht_func = ht.exp
        np_func = np.exp
        self.assert_func_equal(shape, heat_func=ht_func, numpy_func=np_func)
