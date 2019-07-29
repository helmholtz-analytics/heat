from unittest import TestCase
from heat.core import dndarray, MPICommunication, MPI, types, factories

import numpy as np
import torch


class BasicTest(TestCase):

    __comm = MPICommunication()
    __device = None

    @property
    def device(self):
        return BasicTest.__device

    @property
    def comm(self):
        return BasicTest.__comm

    def get_rank(self):
        return self.comm.rank

    def get_size(self):
        return self.comm.size

    def assert_func_equal(self, tensor, heat_func, numpy_func, distributed_result=True, heat_args=None, numpy_args=None):
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            tensor = np.random.randn(tensor)
        print('rank', self.get_rank())
        if heat_args is None:
            heat_args = {}
        if numpy_args is None:
            numpy_args = {}
        if isinstance(tensor, np.ndarray):
            torch_tensor = torch.tensor(tensor)
            print('tensor', torch_tensor.dtype)
            np_array = tensor
        if isinstance(tensor, torch.Tensor):
            torch_tensor = tensor
            np_array = tensor.numpy()
        np_res = numpy_func(np_array, **numpy_args)
        if not isinstance(np_res, np.ndarray):
            np_res = np.array([np_res])
        print('np_res', np_res, np_res.dtype)

        for i in range(len(tensor.shape)):
            dtype = types.canonical_heat_type(torch_tensor.dtype)
            print('dtype', dtype)
            ht_array = factories.array(torch_tensor, split=i, dtype=dtype, device=self.device, comm=self.comm)
            print('heat dtype', ht_array.dtype)
            ht_res = heat_func(ht_array, **heat_args)
            print('ht_res', ht_res, 'gshape', ht_res.gshape)
            if distributed_result:
                self.assert_array_equal(ht_res, np_res)
            else:
                self.assertTrue(np.array_equal(ht_res._DNDarray__array.numpy(), np_res))

    def assert_array_equal(self, heat_array: dndarray.DNDarray, numpy_array):
        split = heat_array.split
        offset, local_shape, slices = heat_array.comm.chunk(heat_array.gshape, split)

        # Array is distributed correctly
        equal_res = np.array(np.array_equal(heat_array._DNDarray__array.numpy(), numpy_array[slices]))
        self.comm.Allreduce(MPI.IN_PLACE, equal_res, MPI.LAND)
        self.assertTrue(equal_res)

        # Combined array is correct
        combined = heat_array.numpy()
        print('numpy', combined, combined.dtype, heat_array.dtype)
        self.assertTrue(np.array_equal(heat_array.numpy(), numpy_array))
