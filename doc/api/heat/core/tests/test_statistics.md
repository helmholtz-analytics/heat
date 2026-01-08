Module heat.core.tests.test_statistics
======================================

Classes
-------

`TestStatistics(methodName='runTest')`
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

    `test_argmax(self)`
    :

    `test_argmin(self)`
    :

    `test_average(self)`
    :

    `test_bincount(self)`
    :

    `test_bucketize(self)`
    :

    `test_cov(self)`
    :

    `test_digitize(self)`
    :

    `test_histc(self)`
    :

    `test_kurtosis(self)`
    :

    `test_max(self)`
    :

    `test_maximum(self)`
    :

    `test_mean(self)`
    :

    `test_min(self)`
    :

    `test_minimum(self)`
    :

    `test_percentile(self)`
    :

    `test_percentile_sketched(self)`
    :

    `test_skew(self)`
    :

    `test_std(self)`
    :

    `test_var(self)`
    :
