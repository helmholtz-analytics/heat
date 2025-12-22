Module heat.core.tests.test_suites.basic_test
=============================================

Classes
-------

`TestCase(methodName='runTest')`
:   A class whose instances are single test cases.

    By default, the test code itself should be placed in a method named
    'runTest'.

    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.

    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.

    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.

    When subclassing TestCase, you can set these attributes:
    * failureException: determines which exception will be raised when
        the instance's assertion methods fail; test methods raising this
        exception will be deemed to have 'failed' rather than 'errored'.
    * longMessage: determines whether long messages (including repr of
        objects used in assert methods) will be printed on failure in *addition*
        to any explicit message passed.
    * maxDiff: sets the maximum length of a diff in failure messages
        by assert methods using difflib. It is looked up as an instance
        attribute so can be configured by individual tests if required.

    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Descendants

    * heat.classification.tests.test_knn.TestKNN
    * heat.cluster.tests.test_batchparallelclustering.TestAuxiliaryFunctions
    * heat.cluster.tests.test_batchparallelclustering.TestBatchParallelKCluster
    * heat.cluster.tests.test_kmeans.TestKMeans
    * heat.cluster.tests.test_kmedians.TestKMedians
    * heat.cluster.tests.test_kmedoids.TestKMeans
    * heat.cluster.tests.test_spectral.TestSpectral
    * heat.core.linalg.tests.test_basics.TestLinalgBasics
    * heat.core.linalg.tests.test_eigh.TestEigh
    * heat.core.linalg.tests.test_polar.TestZolopolar
    * heat.core.linalg.tests.test_qr.TestQR
    * heat.core.linalg.tests.test_solver.TestSolver
    * heat.core.linalg.tests.test_svd.TestTallSkinnySVD
    * heat.core.linalg.tests.test_svd.TestZoloSVD
    * heat.core.linalg.tests.test_svdtools.TestHSVD
    * heat.core.linalg.tests.test_svdtools.TestISVD
    * heat.core.linalg.tests.test_svdtools.TestRSVD
    * heat.core.tests.test_arithmetics.TestArithmetics
    * heat.core.tests.test_communication.TestCommunication
    * heat.core.tests.test_complex_math.TestComplex
    * heat.core.tests.test_constants.TestConstants
    * heat.core.tests.test_devices.TestDevices
    * heat.core.tests.test_dndarray.TestDNDarray
    * heat.core.tests.test_exponential.TestExponential
    * heat.core.tests.test_factories.TestFactories
    * heat.core.tests.test_indexing.TestIndexing
    * heat.core.tests.test_io.TestIO
    * heat.core.tests.test_logical.TestLogical
    * heat.core.tests.test_manipulations.TestManipulations
    * heat.core.tests.test_memory.TestMemory
    * heat.core.tests.test_operations.TestOperations
    * heat.core.tests.test_printing.TestPrinting
    * heat.core.tests.test_printing.TestPrintingGPU
    * heat.core.tests.test_random.TestRandom_Batchparallel
    * heat.core.tests.test_random.TestRandom_Threefry
    * heat.core.tests.test_relational.TestRelational
    * heat.core.tests.test_rounding.TestRounding
    * heat.core.tests.test_sanitation.TestSanitation
    * heat.core.tests.test_signal.TestSignal
    * heat.core.tests.test_statistics.TestStatistics
    * heat.core.tests.test_stride_tricks.TestStrideTricks
    * heat.core.tests.test_suites.test_basic_test.TestBasicTest
    * heat.core.tests.test_tiling.TestSplitTiles
    * heat.core.tests.test_tiling.TestSquareDiagTiles
    * heat.core.tests.test_trigonometrics.TestTrigonometrics
    * heat.core.tests.test_types.TestTypeConversion
    * heat.core.tests.test_types.TestTypes
    * heat.core.tests.test_vmap.TestVmap
    * heat.decomposition.tests.test_dmd.TestDMD
    * heat.decomposition.tests.test_dmd.TestDMDc
    * heat.decomposition.tests.test_pca.TestIncrementalPCA
    * heat.decomposition.tests.test_pca.TestPCA
    * heat.fft.tests.test_fft.TestFFT
    * heat.graph.tests.test_laplacian.TestLaplacian
    * heat.naive_bayes.tests.test_gaussiannb.TestGaussianNB
    * heat.optim.tests.test_dp_optimizer.TestDASO
    * heat.optim.tests.test_optim.TestLRScheduler
    * heat.optim.tests.test_optim.TestOptim
    * heat.optim.tests.test_utils.TestUtils
    * heat.preprocessing.tests.test_preprocessing.TestMaxAbsScaler
    * heat.preprocessing.tests.test_preprocessing.TestMinMaxScaler
    * heat.preprocessing.tests.test_preprocessing.TestNormalizer
    * heat.preprocessing.tests.test_preprocessing.TestRobustScaler
    * heat.preprocessing.tests.test_preprocessing.TestStandardScaler
    * heat.regression.tests.test_lasso.TestLasso
    * heat.sparse.tests.test_arithmetics_csr.TestArithmeticsCSR
    * heat.sparse.tests.test_dcscmatrix.TestDCSC_matrix
    * heat.sparse.tests.test_dcsrmatrix.TestDCSR_matrix
    * heat.sparse.tests.test_factories.TestFactories
    * heat.sparse.tests.test_manipulations.TestManipulations
    * heat.spatial.tests.test_distances.TestDistances
    * heat.utils.data.tests.test_matrixgallery.TestMatrixgallery
    * heat.utils.data.tests.test_spherical.TestCreateClusters

    ### Class variables

    `device: heat.core.devices.Device`
    :

    `envar: str | None`
    :

    `other_device: heat.core.devices.Device | None`
    :

    ### Static methods

    `get_hostnames() ‑> list[str]`
    :

    `setUpClass() ‑> None`
    :   Read the environment variable 'HEAT_TEST_USE_DEVICE' and return the requested devices.
        Supported values
            - cpu: Use CPU only (default)
            - gpu: Use GPU only

        Raises
        ------
        RuntimeError if value of 'HEAT_TEST_USE_DEVICE' is not recognized

    ### Instance variables

    `comm: heat.core.communication.MPICommunication`
    :

    ### Methods

    `assertTrue_memory_layout(self, tensor: heat.core.dndarray.DNDarray, order: str) ‑> None`
    :   Checks that the memory layout of a given heat tensor is as specified by argument order.

        Parameters
        ----------
        order: str, 'C' for C-like (row-major), 'F' for Fortran-like (column-major) memory layout.

    `assert_array_equal(self, heat_array: heat.core.dndarray.DNDarray, expected_array: numpy.ndarray | torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08) ‑> None`
    :   Check if the heat_array is equivalent to the expected_array. Therefore first the split heat_array is compared to
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

    `assert_func_equal(self, shape: tuple[typing.Any, ...] | list[typing.Any], heat_func: Callable[..., Any], numpy_func: Callable[..., Any], distributed_result: bool = True, heat_args: dict[str, typing.Any] | None = None, numpy_args: dict[str, typing.Any] | None = None, data_types: tuple[type, ...] = (<class 'numpy.int32'>, <class 'numpy.int64'>, <class 'numpy.float32'>, <class 'numpy.float64'>), low: int = -10000, high: int = 10000) ‑> None`
    :   This function will create random tensors of the given shape with different data types.
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

    `assert_func_equal_for_tensor(self, tensor: numpy.ndarray | torch.Tensor, heat_func: Callable[..., Any], numpy_func: Callable[..., Any], heat_args: dict[str, typing.Any] | None = None, numpy_args: dict[str, typing.Any] | None = None, distributed_result: bool = True) ‑> None`
    :   This function tests if the heat function and the numpy function create the equal result on the given tensor.

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

    `get_rank(self) ‑> int | None`
    :

    `get_size(self) ‑> int | None`
    :
