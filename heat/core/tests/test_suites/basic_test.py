from unittest import TestCase
from heat.core import dndarray, MPICommunication, MPI, types, factories, random

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
        print('rank', self.get_rank())
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            tensor = self._create_random_array(tensor)

        if heat_args is None:
            heat_args = {}
        if numpy_args is None:
            numpy_args = {}

        if isinstance(tensor, np.ndarray):
            torch_tensor = torch.from_numpy(tensor.copy())
            np_array = tensor
        elif isinstance(tensor, torch.Tensor):
            torch_tensor = tensor
            np_array = tensor.numpy().copy()
        else:
            raise TypeError('The input tensors type must be one of [tuple, list, ' +
                            'numpy.ndarray, torch.tensor] but is {}'.format(type(tensor)))

        print('tensor', tensor)
        np_res = numpy_func(np_array, **numpy_args)
        if not isinstance(np_res, np.ndarray):
            np_res = np.array([np_res])
        print('np_res', np_res, np_res.dtype)

        dtype = types.canonical_heat_type(torch_tensor.dtype)
        print('dtype', dtype)

        for i in range(len(tensor.shape)):
            print('before', torch_tensor)
            ht_array = factories.array(torch_tensor, split=i, dtype=dtype, device=self.device, comm=self.comm)
            ht_res = heat_func(ht_array, **heat_args)
            print('ht_res', ht_res, 'gshape', ht_res.gshape)
            if distributed_result:
                self.assert_array_equal(ht_res, np_res)
            else:
                self.assertTrue(np.array_equal(ht_res._DNDarray__array.numpy(), np_res))

    def assert_array_equal(self, heat_array: dndarray.DNDarray, numpy_array):
        self.assertIsInstance(heat_array, dndarray.DNDarray,
                              'The array to test was not a instance of ht.DNDarray. '
                              'Instead got {}.'.format(type(heat_array)))
        self.assertIsInstance(numpy_array, np.ndarray,
                              'The array to test against was not a instance of numpy.ndarray. '
                              'Instead got {}.'.format(type(numpy_array)))

        numpy_array = numpy_array.astype(np.float64)
        split = heat_array.split
        offset, local_shape, slices = heat_array.comm.chunk(heat_array.gshape, split)

        local_numpy = heat_array._DNDarray__array.numpy()

        # Array is distributed correctly
        equal_res = np.array(np.array_equal(local_numpy, numpy_array[slices]))
        print('heat', local_numpy, '\nnumpy', numpy_array[slices], '\ndiff', heat_array._DNDarray__array.numpy() - numpy_array[slices])
        self.comm.Allreduce(MPI.IN_PLACE, equal_res, MPI.LAND)
        self.assertTrue(equal_res, 'Local tensors do not match the corresponding numpy slices.')
        self.assertEqual(local_numpy.dtype, numpy_array.dtype,
                         'Resulting types do not match heat: {} numpy: {}.'.format(heat_array.dtype, numpy_array.dtype))

        # Combined array is correct
        combined = heat_array.numpy()
        print('combined', combined)
        self.assertTrue(np.array_equal(heat_array.numpy(), numpy_array), 'Combined tensors do not match.')

    def _create_random_array(self, shape):
        # Ensure all processes have the same initial array
        if self.get_rank() == 0:
            array = np.random.randn(*shape)
        else:
            array = np.empty(shape)
        self.comm.Bcast(array, root=0)
        return array
