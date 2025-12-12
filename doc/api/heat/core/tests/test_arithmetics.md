Module heat.core.tests.test_arithmetics
=======================================

Classes
-------

`TestArithmetics(methodName='runTest')`
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

    `test_add(self)`
    :

    `test_add_(self)`
    :

    `test_bitwise_and(self)`
    :

    `test_bitwise_and_(self)`
    :

    `test_bitwise_or(self)`
    :

    `test_bitwise_or_(self)`
    :

    `test_bitwise_xor(self)`
    :

    `test_bitwise_xor_(self)`
    :

    `test_copysign(self)`
    :

    `test_copysign_(self)`
    :

    `test_cumprod(self)`
    :

    `test_cumprod_(self)`
    :

    `test_cumsum(self)`
    :

    `test_cumsum_(self)`
    :

    `test_diff(self)`
    :

    `test_div(self)`
    :

    `test_div_(self)`
    :

    `test_divmod(self)`
    :

    `test_floordiv(self)`
    :

    `test_floordiv_(self)`
    :

    `test_fmod(self)`
    :

    `test_fmod_(self)`
    :

    `test_gcd(self)`
    :

    `test_gcd_(self)`
    :

    `test_hypot(self)`
    :

    `test_hypot_(self)`
    :

    `test_invert(self)`
    :

    `test_invert_(self)`
    :

    `test_lcm(self)`
    :

    `test_lcm_(self)`
    :

    `test_left_shift(self)`
    :

    `test_left_shift_(self)`
    :

    `test_mul(self)`
    :

    `test_mul_(self)`
    :

    `test_nan_to_num(self)`
    :

    `test_nan_to_num_(self)`
    :

    `test_nanprod(self)`
    :

    `test_nansum(self)`
    :

    `test_neg(self)`
    :

    `test_neg_(self)`
    :

    `test_pos(self)`
    :

    `test_pow(self)`
    :

    `test_pow_(self)`
    :

    `test_prod(self)`
    :

    `test_remainder(self)`
    :

    `test_remainder_(self)`
    :

    `test_right_hand_side_operations(self)`
    :   This test ensures that for each arithmetic operation (e.g. +, -, *, ...) that is implemented
        in the tensor class, it works both ways.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.float32([[1., 2.], [3., 4.]])
        >>> assert T * 3 == 3 * T

    `test_right_shift(self)`
    :

    `test_right_shift_(self)`
    :

    `test_sub(self)`
    :

    `test_sub_(self)`
    :

    `test_sum(self)`
    :
