Module heat.core.tests.test_communication
=========================================

Classes
-------

`TestCommunication(methodName='runTest')`
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

    `test_allgather(self)`
    :

    `test_allgatherv(self)`
    :

    `test_allgathervSorting(self)`
    :

    `test_allreduce(self)`
    :

    `test_alltoall(self)`
    :

    `test_alltoallSorting(self)`
    :

    `test_alltoallv(self)`
    :

    `test_bcast(self)`
    :

    `test_contiguous_memory_buffer(self)`
    :

    `test_cuda_aware_mpi(self)`
    :

    `test_default_comm(self)`
    :

    `test_exscan(self)`
    :

    `test_gather(self)`
    :

    `test_gatherv(self)`
    :

    `test_iallgather(self)`
    :

    `test_iallgatherv(self)`
    :

    `test_iallreduce(self)`
    :

    `test_ialltoall(self)`
    :

    `test_ialltoallv(self)`
    :

    `test_ibcast(self)`
    :

    `test_iexscan(self)`
    :

    `test_igather(self)`
    :

    `test_igatherv(self)`
    :

    `test_ireduce(self)`
    :

    `test_iscan(self)`
    :

    `test_iscatter(self)`
    :

    `test_iscatterv(self)`
    :

    `test_largecount_workaround_Allreduce(self)`
    :

    `test_largecount_workaround_IsendRecv(self)`
    :

    `test_mpi_communicator(self)`
    :

    `test_mpi_in_place(self)`
    :

    `test_non_contiguous_memory_buffer(self)`
    :

    `test_reduce(self)`
    :

    `test_scan(self)`
    :

    `test_scatter(self)`
    :

    `test_scatter_like_axes(self)`
    :

    `test_scatterv(self)`
    :

    `test_self_communicator(self)`
    :

    `test_split(self)`
    :
