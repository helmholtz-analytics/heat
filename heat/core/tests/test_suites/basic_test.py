from unittest import TestCase
from heat.core import dndarray, MPI

import numpy as np


class BasicTest(TestCase):

    def test_function(self, tensor, heat_func, numpy_func, heat_args=None, numpy_args=None, device=None, comm=MPI.COMM_WORLD):
        for i in range(len(tensor.shape)):
            ht_array = dndarray.DNDarray(tensor, split=i, dtype=tensor.dtype, gshape=tensor.shape, device=device, comm=comm)
            ht_res = heat_func(ht_array, **heat_args)

    def compare_results(self, heat_array: dndarray.DNDarray, numpy_array):
        split = heat_array.split
        offset, local_shape, slices = heat_array.comm.chunk(heat_array.gshape, split)
        print('offset', offset, 'local_shape', local_shape, 'slices', slices)
        print('data', heat_array)
        self.assertTrue(np.array_equal(heat_array._DNDarray__array.numpy(), numpy_array[slices]))
        heat_array.resplit(axis=None)
        print('data', heat_array)
        self.assertTrue(np.array_equal(heat_array._DNDarray__array.numpy(), numpy_array))
