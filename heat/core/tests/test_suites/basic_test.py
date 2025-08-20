import os
import platform
import unittest

import numpy as np
import torch
from typing import Optional, Callable, Any, Union

import heat as ht
from heat.core import MPI, MPICommunication, dndarray, factories, types, Device
from heat.core.random import seed

# TODO adapt for GPU once this is working properly
class TestCase(unittest.TestCase):
    __comm = MPICommunication()
    device: Device = ht.cpu
    _hostnames: Optional[list[str]] = None
    other_device: Optional[Device] = None
    envar: Optional[str] = None


    @classmethod
    def setUpClass(cls) -> None:
        """
        Read the environment variable 'HEAT_TEST_USE_DEVICE' and return the requested devices.
        Supported values
            - cpu: Use CPU only (default)
            - gpu: Use GPU only

        Raises
        ------
        RuntimeError if value of 'HEAT_TEST_USE_DEVICE' is not recognized

        """
        envar = os.getenv("HEAT_TEST_USE_DEVICE", "cpu")
        is_mps = False

        if envar == "cpu":
            ht.use_device("cpu")
            ht_device = ht.cpu
            other_device = ht.cpu
            if torch.cuda.is_available():
                torch.cuda.set_device(torch.device(ht.gpu.torch_device))
                other_device = ht.gpu
            elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
                other_device = ht.gpu
        elif envar == "gpu":
            if torch.cuda.is_available():
                ht.use_device("gpu")
                torch.cuda.set_device(torch.device(ht.gpu.torch_device))
                ht_device = ht.gpu
                other_device = ht.cpu
            elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
                ht.use_device("gpu")
                ht_device = ht.gpu
                other_device = ht.cpu
                is_mps = True
        else:
            raise RuntimeError(
                f"Value '{envar}' of environment variable 'HEAT_TEST_USE_DEVICE' is unsupported"
            )

        cls.device, cls.other_device, cls.envar, cls.is_mps = ht_device, other_device, envar, is_mps
        seed(42)

    @property
    def comm(self) -> MPICommunication:
        return self.__comm


    def get_rank(self) -> Optional[int]:
        return self.comm.rank

    def get_size(self) -> Optional[int]:
        return self.comm.size

    @classmethod
    def get_hostnames(cls) -> list[str]:
        if not cls._hostnames:
            if platform.system() == "Windows":
                host = platform.uname().node
            else:
                host = os.uname()[1]
            cls._hostnames = list(set(cls.__comm.handle.allgather(host)))
        return cls._hostnames

    def assert_array_equal(self, heat_array: ht.DNDarray, expected_array: Union[np.ndarray,torch.Tensor], rtol:float=1e-5, atol:float=1e-08) -> None:
        """
        Check if the heat_array is equivalent to the expected_array. Therefore first the split heat_array is compared to
        the corresponding expected_array slice locally and second the heat_array is combined and fully compared with the
        expected_array.
        Note if the heat array is split it also needs to be balanced.

        Parameters
        ----------
        heat_array: heat.DNDarray
            The heat array which should be checked.
        expected_array: numpy.ndarray or torch.Tensor
            The array against which the heat_array should be checked.

        Raises
        ------
        AssertionError if the arrays do not equal.

        Examples
        --------
        >>> import numpy as np
        >>> import heat as ht
        >>> a = ht.ones((5, 5), split=1, dtype=ht.int32)
        >>> b = np.ones((5, 5), dtype=np.int32)
        >>> self.assert_array_equal(a, b)

        >>> c = np.ones((5, 5), dtype=np.int64)
        >>> self.assert_array_equal(a, c)
        AssertionError: [...]
        >>> c = np.zeros((5, 5), dtype=np.int32)
        >>> self.assert_array_equal(a, c)
        AssertionError: [...]
        """
        self._comm = heat_array.comm
        if isinstance(expected_array, torch.Tensor):
            # Does not work because heat sets an index while torch does not
            # self.assertEqual(expected_array.device, torch.device(heat_array.device.torch_device))
            expected_array = expected_array.cpu().numpy()

        self.assertIsInstance(
            heat_array,
            dndarray.DNDarray,
            f"The array to test was not a instance of ht.DNDarray. Instead got {type(heat_array)}.",
        )
        self.assertIsInstance(
            expected_array,
            np.ndarray,
            f"The array to test against was not a instance of numpy.ndarray or torch.Tensor Instead got {type(expected_array)}.",
        )
        self.assertEqual(
            heat_array.shape,
            expected_array.shape,
            f"Global shapes do not match. Got {heat_array.shape} expected {expected_array.shape}",
        )

        if not heat_array.is_balanced():
            # Array is not distributed correctly
            heat_array.balance_()

        split = heat_array.split
        offset, local_shape, slices = heat_array.comm.chunk(heat_array.gshape, split)
        self.assertEqual(
            heat_array.lshape,
            expected_array[slices].shape,
            f"Local shapes do not match. Got {heat_array.lshape} expected {expected_array[slices].shape}",
        )
        # compare local tensors to corresponding slice of expected_array
        is_allclose = torch.tensor(
            np.allclose(heat_array.larray.cpu(), expected_array[slices], atol=atol, rtol=rtol),
            dtype=torch.int32,
        )
        heat_array.comm.Allreduce(MPI.IN_PLACE, is_allclose, MPI.SUM)
        self.assertTrue(is_allclose == heat_array.comm.size)

    def assert_func_equal(
        self,
        shape: Union[tuple[Any, ...],list[Any]],
        heat_func: Callable[..., Any],
        numpy_func: Callable[..., Any],
        distributed_result: bool=True,
        heat_args: Optional[dict[str, Any]]=None,
        numpy_args:Optional[dict[str, Any]]=None,
        data_types: tuple[type,...]=(np.int32, np.int64, np.float32, np.float64),
        low:int=-10000,
        high:int=10000,
    ) -> None:
        """
        This function will create random tensors of the given shape with different data types.
        All of these tensors will be tested with `ht.assert_func_equal_for_tensor`.

        Parameters
        ----------
        shape: tuple or list
            The shape of which a random tensors will be created and tested against
        heat_func: function
            The function that is to be tested
        numpy_func: function
            The numpy implementation of an equivalent function to test against
        heat_args: dictionary, optional
            The keyword arguments that will be passed to the heat function. Array and split function don't need to be
            specified. Default is {}.
        numpy_args: dictionary, optional
            The keyword arguments that will be passed to the numpy function. Array doesn't need to be specified.
            Default is {}.
        distributed_result: bool, optional
            Specify whether the result of the heat function is distributed across all nodes or all nodes have the full
            result. Default is True.
        data_types: list of numpy dtypes, optional
            Tensors with all of these dtypes will be created and tested. Each type must to be a numpy dtype.
            Default is [numpy.int32, numpy.int64, numpy.float32, numpy.float64]
        low: int, optional
            In case one of the data_types has integer types, this is the lower bound for the random values.
            Default is -10000
        high: int, optional
            In case one of the data_types has integer types, this is the upper bound for the random values.
            Default is 10000

        Raises
        ------
        AssertionError if the functions do not perform equally.

        Examples
        --------
        >>> import numpy as np
        >>> import heat as ht
        >>> self.assert_func_equal((2, 2), ht.exp, np.exp)

        >>> self.assert_func_equal((2, 2), ht.exp, np.log)
        AssertionError: [...]
        >>> self.assert_func_equal((1, 3, 5), ht.any, np.any, distributed_result=False)

        >>> heat_args = {"sorted": True, "axis": 0}
        >>> numpy_args = {"axis": 0}
        >>> self.assert_func_equal(
        ...     [5, 5, 5, 5], ht.unique, np.unique, heat_arg=heat_args, numpy_args=numpy_args
        ... )
        """
        if not isinstance(shape, tuple) and not isinstance(shape, list):
            raise ValueError(f"The shape must be either a list or a tuple but was {type(shape)}")

        if self.is_mps and np.float64 in data_types:
            # MPS does not support float64
            data_types = [dtype for dtype in data_types if dtype != np.float64]

        for dtype in data_types:
            tensor = self.__create_random_np_array(shape, dtype=dtype, low=low, high=high)
            self.assert_func_equal_for_tensor(
                tensor=tensor,
                heat_func=heat_func,
                numpy_func=numpy_func,
                heat_args=heat_args,
                numpy_args=numpy_args,
                distributed_result=distributed_result,
            )

    def assert_func_equal_for_tensor(
        self,
        tensor: Union[np.ndarray,torch.Tensor],
        heat_func: Callable[..., Any],
        numpy_func: Callable[..., Any],
        heat_args:Optional[dict[str,Any]]=None,
        numpy_args:Optional[dict[str,Any]]=None,
        distributed_result:bool=True,
    ) -> None:
        """
        This function tests if the heat function and the numpy function create the equal result on the given tensor.

        Parameters
        ----------
        tensor: torch.Tensor or numpy.ndarray
            The tensor on which the heat function will be executed.
        heat_func: function
            The function that is to be tested
        numpy_func: function
            The numpy implementation of an equivalent function to test against
        heat_args: dictionary, optional
            The keyword arguments that will be passed to the heat function. Array and split function don't need to be
            specified. Default is {}.
        numpy_args: dictionary, optional
            The keyword arguments that will be passed to the numpy function. Array doesn't need to be specified.
            Default is {}.
        distributed_result: bool, optional
            Specify whether the result of the heat function is distributed across all nodes or all nodes have the full
            result. Default is True.

        Raises
        ------
        AssertionError if the functions to not perform equally.

        Examples
        --------
        >>> import numpy as np
        >>> import heat as ht
        >>> a = np.arange(10)
        >>> self.assert_func_equal_for_tensor(a, ht.exp, np.exp)

        >>> self.assert_func_equal_for_tensor(a, ht.exp, np.log)
        AssertionError: [...]
        >>> self.assert_func_equal_for_tensor(a, ht.any, np.any, distributed_result=False)

        >>> a = torch.ones([5, 5, 5, 5])
        >>> heat_args = {"sorted": True, "axis": 0}
        >>> numpy_args = {"axis": 0}
        >>> self.assert_func_equal_for_tensor(
        ...     a, ht.unique, np.unique, heat_arg=heat_args, numpy_args=numpy_args
        ... )
        """
        self.assertTrue(callable(heat_func))
        self.assertTrue(callable(numpy_func))

        if heat_args is None:
            heat_args = {}
        if numpy_args is None:
            numpy_args = {}

        if isinstance(tensor, np.ndarray):
            torch_tensor = torch.from_numpy(tensor.copy())
            torch_tensor = torch_tensor.to(self.device.torch_device)
            np_array = tensor
        elif isinstance(tensor, torch.Tensor):
            torch_tensor = tensor
            np_array = tensor.cpu().numpy().copy()
        else:
            raise TypeError(
                f"The input tensors type must be one of [tuple, list, numpy.ndarray, torch.tensor] but is {type(tensor)}"
            )

        dtype = types.canonical_heat_type(torch_tensor.dtype)
        np_res = numpy_func(np_array, **numpy_args)
        if not isinstance(np_res, np.ndarray):
            np_res = np.array(np_res)

        for i in range(len(tensor.shape)):
            ht_array = factories.array(
                torch_tensor, split=i, dtype=dtype, device=self.device, comm=self.comm
            )
            ht_res = heat_func(ht_array, **heat_args)

            self.assertEqual(ht_array.device, ht_res.device)
            self.assertEqual(ht_array.larray.device, ht_res.larray.device)
            if distributed_result:
                self.assert_array_equal(ht_res, np_res)
            else:
                self.assertTrue(np.array_equal(ht_res.larray.cpu().numpy(), np_res))

    def assertTrue_memory_layout(self, tensor: ht.DNDarray, order: str) -> None:
        """
        Checks that the memory layout of a given heat tensor is as specified by argument order.

        Parameters
        ----------
        order: str, 'C' for C-like (row-major), 'F' for Fortran-like (column-major) memory layout.
        """
        stride = tensor.larray.stride()
        row_major = all(np.diff(list(stride)) <= 0)
        column_major = all(np.diff(list(stride)) >= 0)
        if order == "C":
            return self.assertTrue(row_major)
        elif order == "F":
            return self.assertTrue(column_major)
        else:
            raise ValueError(f"expected order to be 'C' or 'F', but was {order}")

    def __create_random_np_array(self, shape: Union[list[Any],tuple[Any]], dtype:type=np.float32, low:int=-10000, high:int=10000) -> np.ndarray:
        """
        Creates a random array based on the input parameters.
        The used seed will be printed to stdout for debugging purposes.

        Parameters
        ----------
        shape: list or tuple
            The shape of the random array to be created.
        dtype: np.dtype, optional
            The datatype of the resulting array.
            If dtype is subclass of np.floating then numpy.random.randn is used to create random values.
            If dtype is subclass of np.integer then numpy.random.randint is called with the low and high argument to
            create the random values.
            Default is numpy.float64
        low: int, optional
            In case dtype is an integer type, this is the lower bound for the random values.
            Default is -10000
        low: int, optional
            In case dtype is an integer type, this is the upper bound for the random values.
            Default is 10000

        Returns
        -------
        res: numpy.ndarray
            An array of random values with the specified shape and dtype.

        Raises
        ------
        ValueError if the dtype is not a subtype of numpy.integer or numpy.floating
        """
        seed = np.random.randint(1000000, size=(1,))
        # print("using seed {} for random values".format(seed))
        self.comm.Bcast(seed, root=0)
        np.random.seed(seed=seed.item())
        if issubclass(dtype, np.floating):
            array: np.ndarray = np.random.randn(*shape)
        elif issubclass(dtype, np.integer):
            array = np.random.randint(low=low, high=high, size=shape)
        else:
            raise ValueError(
                f"Unsupported dtype. Expected a subclass of `np.floating` or `np.integer` but got {dtype}"
            )
        array = array.astype(dtype)
        return array
