Module heat.core.tests.test_dndarray
====================================

Classes
-------

`TestDNDarray(methodName='runTest')`
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

    * heat.core.tests.test_suites.basic_test.TestCase
    * unittest.case.TestCase

    ### Methods

    `test_and(self)`
    :

    `test_array(self)`
    :

    `test_array_function(self)`
    :

    `test_array_ufunc(self)`
    :

    `test_astype(self)`
    :

    `test_balance_and_lshape_map(self)`
    :

    `test_bool_cast(self)`
    :

    `test_collect(self)`
    :

    `test_complex_cast(self)`
    :

    `test_counts_displs(self)`
    :

    `test_fill_diagonal(self)`
    :

    `test_flatten(self)`
    :

    `test_float_cast(self)`
    :

    `test_gethalo(self)`
    :

    `test_int_cast(self)`
    :

    `test_invert(self)`
    :

    `test_is_balanced(self)`
    :

    `test_is_distributed(self)`
    :

    `test_item(self)`
    :

    `test_larray(self)`
    :

    `test_len(self)`
    :

    `test_lloc(self)`
    :

    `test_lnbytes(self)`
    :

    `test_nbytes(self)`
    :

    `test_ndim(self)`
    :

    `test_numpy(self)`
    :

    `test_or(self)`
    :

    `test_partitioned(self)`
    :

    `test_redistribute(self)`
    :

    `test_resplit(self)`
    :

    `test_setitem_getitem(self)`
    :

    `test_size_gnumel(self)`
    :

    `test_stride_and_strides(self)`
    :

    `test_tolist(self)`
    :

    `test_torch_function(self)`
    :

    `test_torch_proxy(self)`
    :

    `test_xor(self)`
    :
