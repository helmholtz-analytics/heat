import operator

import heat as ht
import numpy as np
import torch

from .test_suites.basic_test import TestCase


class TestArithmetics(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestArithmetics, cls).setUpClass()
        cls.a_scalar = 2.0

        cls.a_vector = ht.float32([2, 2])
        cls.another_vector = ht.float32([2, 2, 2])

        cls.a_tensor = ht.array([[1.0, 2.0], [3.0, 4.0]])
        cls.another_tensor = ht.array([[2.0, 2.0], [2.0, 2.0]])
        cls.a_split_tensor = cls.another_tensor.copy().resplit_(0)

        cls.an_int_scalar = 2

        cls.an_int_vector = ht.array([2, 2])
        cls.another_int_vector = ht.array([2, 2, 2, 2])

        cls.an_int_tensor = ht.array([[1, 2], [3, 4]])
        cls.another_int_tensor = ht.array([[2, 2], [2, 2]])
        cls.a_split_int_tensor = cls.an_int_tensor.copy().resplit_(0)

        cls.erroneous_type = (2, 2)
        cls.a_string = "T"

        cls.a_boolean_vector = ht.array([False, True, False, True])
        cls.another_boolean_vector = ht.array([False, False, True, True])

    def test_add(self):
        # test basics
        result = ht.array([[3.0, 4.0], [5.0, 6.0]])

        self.assertTrue(ht.equal(ht.add(self.a_scalar, self.a_scalar), ht.float32(4.0)))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.add(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.add(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.add(self.a_split_tensor, self.a_tensor), result))

        self.assertTrue(ht.equal(self.a_tensor + self.a_scalar, result))
        self.assertTrue(ht.equal(self.a_scalar + self.a_tensor, result))

        # Single element split
        a = ht.array([1], split=0)
        b = ht.array([1, 2], split=0)
        c = ht.add(a, b)
        self.assertTrue(ht.equal(c, ht.array([2, 3])))
        if c.comm.size > 1:
            if c.comm.rank < 2:
                self.assertEqual(c.larray.size()[0], 1)
            else:
                self.assertEqual(c.larray.size()[0], 0)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0)
        b = ht.zeros(10, split=0)
        c = a[:-1] + b[1:]
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == a[:-1].lshape)

        c = a[1:-1] + b[1:-1]  # test unbalanced
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == a[1:-1].lshape)

        # test one unsplit
        a = ht.ones(10, split=None)
        b = ht.zeros(10, split=0)
        c = a[:-1] + b[1:]
        self.assertTrue((c == 1).all())
        self.assertEqual(c.lshape, b[1:].lshape)
        c = b[:-1] + a[1:]
        self.assertTrue((c == 1).all())
        self.assertEqual(c.lshape, b[:-1].lshape)

        # broadcast in split dimension
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        c = a + b
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == b.lshape)
        c = b + a
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == b.lshape)

        # out parameter, where parameter
        a = ht.ones((2, 2), split=0)
        b = out = ht.ones((2, 2), split=0)
        where = ht.array([[True, False], [False, True]], split=0)
        ht.add(a, b, out=out, where=where)
        self.assertTrue(ht.equal(out, ht.array([[2, 1], [1, 2]])))
        self.assertEqual(out.split, 0)
        self.assertIs(out, b)

        with self.assertRaises(ValueError):
            ht.add(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.add(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.add("T", "s")
        with self.assertRaises(TypeError):
            self.a_tensor + "s"

    def test_add_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[3.0, 4.0], [5.0, 6.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.add_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.add_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.add_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.add_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.__iadd__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor += a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([1], split=0)
        a += b
        self.assertTrue(ht.equal(a, ht.array([2, 3])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0)
        b = ht.zeros(10, split=0)
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a += b[1:]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0)  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a += b[1:-1]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.zeros((2, 10), split=0)
        b = ht.ones((1, 10), split=0)
        a_lshape = a.lshape
        a += b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0)
        b = ht.ones(10, split=None)
        a_lshape = a.lshape
        a += b
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1)
        b = ht.ones(3, split=0)
        a_lshape = a.lshape
        a += b
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.add_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.add_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.add_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.add_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a += b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a += b

        a = ht.ones(9).reshape((3, 3), new_split=0)
        b = ht.ones(9).reshape((3, 3), new_split=1)
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a += b
            self.assertTrue((a == 2).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a += b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a += b

    def test_bitwise_and(self):
        int_result = ht.array([[0, 2], [2, 0]])
        boolean_result = ht.array([False, False, False, True])

        self.assertTrue(
            ht.equal(ht.bitwise_and(self.an_int_tensor, self.an_int_scalar), int_result)
        )
        self.assertTrue(
            ht.equal(ht.bitwise_and(self.an_int_tensor, self.an_int_vector), int_result)
        )
        self.assertTrue(
            ht.equal(
                ht.bitwise_and(self.a_boolean_vector, self.another_boolean_vector), boolean_result
            )
        )
        self.assertTrue(
            ht.equal(ht.bitwise_and(self.a_split_int_tensor, self.an_int_vector), int_result)
        )

        self.assertTrue(ht.equal(self.an_int_tensor & self.an_int_scalar, int_result))
        self.assertTrue(ht.equal(self.an_int_scalar & self.an_int_tensor, int_result))

        # out parameter, where parameter
        out = ht.zeros_like(self.an_int_tensor)
        where = ht.array([[True, False], [False, True]])
        ht.bitwise_and(self.an_int_tensor, self.an_int_scalar, out=out, where=where)
        self.assertTrue(ht.equal(out, ht.array([[0, 0], [0, 0]])))

        with self.assertRaises(TypeError):
            ht.bitwise_and(self.a_tensor, self.another_tensor)
        with self.assertRaises(ValueError):
            ht.bitwise_and(self.an_int_vector, self.another_int_vector)
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.bitwise_and("T", "s")
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.an_int_tensor, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.an_int_scalar, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_and("s", self.an_int_scalar)
        with self.assertRaises(TypeError):
            ht.bitwise_and(self.an_int_scalar, self.a_scalar)
        with self.assertRaises(TypeError):
            self.a_tensor & "s"

    def test_bitwise_and_(self):
        # Copies of class variables for the in-place operations
        a_tensor = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        an_int_vector = self.an_int_vector.copy()
        another_int_vector = self.another_int_vector.copy()
        an_int_tensor = an_int_tensor_double = self.an_int_tensor.copy()
        another_int_tensor = self.another_int_tensor.copy()
        a_boolean_vector = a_boolean_vector_double = self.a_boolean_vector.copy()
        another_boolean_vector = self.another_boolean_vector.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        int_result = ht.array([[0, 2], [2, 0]])
        boolean_result = ht.array([False, False, False, True])

        # We identify the underlying PyTorch objects to check whether operations are really in-place
        underlying_int_torch_tensor = an_int_tensor.larray
        underlying_boolean_torch_tensor = a_boolean_vector.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(
            ht.equal(an_int_tensor.bitwise_and_(an_int_scalar), int_result)
        )  # test result
        self.assertTrue(ht.equal(an_int_tensor, int_result))  # test result in-place
        self.assertIs(an_int_tensor, an_int_tensor_double)  # test DNDarray in-place
        self.assertIs(
            an_int_tensor.larray, underlying_int_torch_tensor
        )  # test torch tensor in-place
        self.assertTrue(
            ht.equal(an_int_scalar, self.an_int_scalar)
        )  # test if other input is unchanged
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int_tensor.bitwise_and_(another_int_tensor), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(another_int_tensor, self.another_int_tensor))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(ht.equal(an_int_tensor.bitwise_and_(an_int_vector), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_vector, self.an_int_vector))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(
            ht.equal(a_boolean_vector.bitwise_and_(another_boolean_vector), boolean_result)
        )
        self.assertTrue(ht.equal(a_boolean_vector, boolean_result))
        self.assertIs(a_boolean_vector, a_boolean_vector_double)
        self.assertIs(a_boolean_vector.larray, underlying_boolean_torch_tensor)
        self.assertTrue(ht.equal(another_boolean_vector, self.another_boolean_vector))
        underlying_boolean_torch_tensor.copy_(self.a_boolean_vector.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(an_int_tensor.__iand__(an_int_scalar), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, self.an_int_scalar))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        an_int_tensor &= an_int_scalar
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, self.an_int_scalar))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([1], split=0)
        a &= b
        self.assertTrue(ht.equal(a, ht.array([1, 0])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0, dtype=int) * 11
        b = ht.ones(10, split=0, dtype=int) * 13
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a &= b[1:]
        self.assertTrue((a == 9).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0, dtype=int) * 11  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a &= b[1:-1]
        self.assertTrue((a == 9).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0, dtype=int) * 11
        b = ht.ones((1, 10), split=0, dtype=int) * 13
        a_lshape = a.lshape
        a &= b
        self.assertTrue((a == 9).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0, dtype=int) * 11
        b = ht.ones(10, split=None, dtype=int) * 13
        a_lshape = a.lshape
        a &= b
        self.assertTrue((a == 9).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12, dtype=int).reshape((4, 3), new_split=1) * 11
        b = ht.ones(3, split=0, dtype=int) * 13
        a_lshape = a.lshape
        a &= b
        self.assertTrue((a == 9).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            a_tensor.bitwise_and_(another_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(ValueError):
            an_int_tensor.bitwise_and_(another_int_vector)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.bitwise_and_(erroneous_type)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            an_int_scalar.bitwise_and_(an_int_tensor)
        an_int_scalar = self.an_int_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.bitwise_and_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0, dtype=int)
        b = ht.zeros((2, 10), split=0, dtype=int)
        with self.assertRaises(ValueError):
            a &= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None, dtype=int)
        b = ht.ones(10, split=0, dtype=int)
        with self.assertRaises(ValueError):
            a &= b

        a = ht.ones(9, dtype=int).reshape((3, 3), new_split=0) * 11
        b = ht.ones(9, dtype=int).reshape((3, 3), new_split=1) * 13
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a &= b
            self.assertTrue((a == 9).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a &= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.int32)
        b = ht.ones(3, dtype=ht.int64)
        with self.assertRaises(TypeError):
            a &= b

    def test_bitwise_or(self):
        int_result = ht.array([[3, 2], [3, 6]])
        boolean_result = ht.array([False, True, True, True])

        self.assertTrue(ht.equal(ht.bitwise_or(self.an_int_tensor, self.an_int_scalar), int_result))
        self.assertTrue(ht.equal(ht.bitwise_or(self.an_int_tensor, self.an_int_vector), int_result))
        self.assertTrue(
            ht.equal(
                ht.bitwise_or(self.a_boolean_vector, self.another_boolean_vector), boolean_result
            )
        )
        self.assertTrue(
            ht.equal(
                ht.bitwise_or(self.an_int_tensor.copy().resplit_(0), self.an_int_vector), int_result
            )
        )

        self.assertTrue(ht.equal(self.an_int_tensor | self.an_int_scalar, int_result))
        self.assertTrue(ht.equal(self.an_int_scalar | self.an_int_tensor, int_result))

        # out parameter, where parameter
        out = ht.zeros_like(self.an_int_tensor)
        where = ht.array([[True, False], [False, True]])
        ht.bitwise_or(self.an_int_tensor, self.an_int_scalar, out=out, where=where)
        self.assertTrue(ht.equal(out, ht.array([[3, 0], [0, 6]])))

        with self.assertRaises(TypeError):
            ht.bitwise_or(self.a_tensor, self.another_tensor)
        with self.assertRaises(ValueError):
            ht.bitwise_or(self.an_int_vector, self.another_int_vector)
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.bitwise_or("T", "s")
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.an_int_tensor, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.an_int_scalar, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_or("s", self.an_int_scalar)
        with self.assertRaises(TypeError):
            ht.bitwise_or(self.an_int_scalar, self.a_scalar)
        with self.assertRaises(TypeError):
            self.a_tensor | "s"

    def test_bitwise_or_(self):
        # Copies of class variables for the in-place operations
        a_tensor = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        an_int_vector = self.an_int_vector.copy()
        another_int_vector = self.another_int_vector.copy()
        an_int_tensor = an_int_tensor_double = self.an_int_tensor.copy()
        another_int_tensor = self.another_int_tensor.copy()
        a_boolean_vector = a_boolean_vector_double = self.a_boolean_vector.copy()
        another_boolean_vector = self.another_boolean_vector.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        int_result = ht.array([[3, 2], [3, 6]])
        boolean_result = ht.array([False, True, True, True])

        # We identify the underlying PyTorch objects to check whether operations are really in-place
        underlying_int_torch_tensor = an_int_tensor.larray
        underlying_boolean_torch_tensor = a_boolean_vector.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(
            ht.equal(an_int_tensor.bitwise_or_(an_int_scalar), int_result)
        )  # test result
        self.assertTrue(ht.equal(an_int_tensor, int_result))  # test result in-place
        self.assertIs(an_int_tensor, an_int_tensor_double)  # test DNDarray in-place
        self.assertIs(
            an_int_tensor.larray, underlying_int_torch_tensor
        )  # test torch tensor in-place
        self.assertTrue(
            ht.equal(an_int_scalar, self.an_int_scalar)
        )  # test if other input is unchanged
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int_tensor.bitwise_or_(another_int_tensor), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(another_int_tensor, self.another_int_tensor))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(ht.equal(an_int_tensor.bitwise_or_(an_int_vector), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_vector, self.an_int_vector))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(
            ht.equal(a_boolean_vector.bitwise_or_(another_boolean_vector), boolean_result)
        )
        self.assertTrue(ht.equal(a_boolean_vector, boolean_result))
        self.assertIs(a_boolean_vector, a_boolean_vector_double)
        self.assertIs(a_boolean_vector.larray, underlying_boolean_torch_tensor)
        self.assertTrue(ht.equal(another_boolean_vector, self.another_boolean_vector))
        underlying_boolean_torch_tensor.copy_(self.a_boolean_vector.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(an_int_tensor.__ior__(an_int_scalar), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, self.an_int_scalar))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        an_int_tensor |= an_int_scalar
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, self.an_int_scalar))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([1], split=0)
        a |= b
        self.assertTrue(ht.equal(a, ht.array([1, 3])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0, dtype=int) * 11
        b = ht.ones(10, split=0, dtype=int) * 13
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a |= b[1:]
        self.assertTrue((a == 15).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0, dtype=int) * 11  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a |= b[1:-1]
        self.assertTrue((a == 15).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0, dtype=int) * 11
        b = ht.ones((1, 10), split=0, dtype=int) * 13
        a_lshape = a.lshape
        a |= b
        self.assertTrue((a == 15).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0, dtype=int) * 11
        b = ht.ones(10, split=None, dtype=int) * 13
        a_lshape = a.lshape
        a |= b
        self.assertTrue((a == 15).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12, dtype=int).reshape((4, 3), new_split=1) * 11
        b = ht.ones(3, split=0, dtype=int) * 13
        a_lshape = a.lshape
        a |= b
        self.assertTrue((a == 15).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            a_tensor.bitwise_or_(another_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(ValueError):
            an_int_tensor.bitwise_or_(another_int_vector)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.bitwise_or_(erroneous_type)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            an_int_scalar.bitwise_or_(an_int_tensor)
        an_int_scalar = self.an_int_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.bitwise_or_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0, dtype=int)
        b = ht.zeros((2, 10), split=0, dtype=int)
        with self.assertRaises(ValueError):
            a |= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None, dtype=int)
        b = ht.ones(10, split=0, dtype=int)
        with self.assertRaises(ValueError):
            a |= b

        a = ht.ones(9, dtype=int).reshape((3, 3), new_split=0) * 11
        b = ht.ones(9, dtype=int).reshape((3, 3), new_split=1) * 13
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a |= b
            self.assertTrue((a == 15).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a |= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.int32)
        b = ht.ones(3, dtype=ht.int64)
        with self.assertRaises(TypeError):
            a |= b

    def test_bitwise_xor(self):
        int_result = ht.array([[3, 0], [1, 6]])
        boolean_result = ht.array([False, True, True, False])

        self.assertTrue(
            ht.equal(ht.bitwise_xor(self.an_int_tensor, self.an_int_scalar), int_result)
        )
        self.assertTrue(
            ht.equal(ht.bitwise_xor(self.an_int_tensor, self.an_int_vector), int_result)
        )
        self.assertTrue(
            ht.equal(
                ht.bitwise_xor(self.a_boolean_vector, self.another_boolean_vector), boolean_result
            )
        )
        self.assertTrue(
            ht.equal(
                ht.bitwise_xor(self.an_int_tensor.copy().resplit_(0), self.an_int_vector),
                int_result,
            )
        )

        self.assertTrue(ht.equal(self.an_int_tensor ^ self.an_int_scalar, int_result))
        self.assertTrue(ht.equal(self.an_int_scalar ^ self.an_int_tensor, int_result))

        # out parameter, where parameter
        out = ht.zeros_like(self.an_int_tensor)
        where = ht.array([[True, False], [False, True]])
        ht.bitwise_xor(self.an_int_tensor, self.an_int_scalar, out=out, where=where)
        self.assertTrue(ht.equal(out, ht.array([[3, 0], [0, 6]])))

        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.a_tensor, self.another_tensor)
        with self.assertRaises(ValueError):
            ht.bitwise_xor(self.an_int_vector, self.another_int_vector)
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.bitwise_xor("T", "s")
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.an_int_tensor, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.an_int_scalar, "s")
        with self.assertRaises(TypeError):
            ht.bitwise_xor("s", self.an_int_scalar)
        with self.assertRaises(TypeError):
            ht.bitwise_xor(self.an_int_scalar, self.a_scalar)
        with self.assertRaises(TypeError):
            self.a_tensor ^ "s"

    def test_bitwise_xor_(self):
        # Copies of class variables for the in-place operations
        a_tensor = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        an_int_vector = self.an_int_vector.copy()
        another_int_vector = self.another_int_vector.copy()
        an_int_tensor = an_int_tensor_double = self.an_int_tensor.copy()
        another_int_tensor = self.another_int_tensor.copy()
        a_boolean_vector = a_boolean_vector_double = self.a_boolean_vector.copy()
        another_boolean_vector = self.another_boolean_vector.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        int_result = ht.array([[3, 0], [1, 6]])
        boolean_result = ht.array([False, True, True, False])

        # We identify the underlying PyTorch objects to check whether operations are really in-place
        underlying_int_torch_tensor = an_int_tensor.larray
        underlying_boolean_torch_tensor = a_boolean_vector.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(
            ht.equal(an_int_tensor.bitwise_xor_(an_int_scalar), int_result)
        )  # test result
        self.assertTrue(ht.equal(an_int_tensor, int_result))  # test result in-place
        self.assertIs(an_int_tensor, an_int_tensor_double)  # test DNDarray in-place
        self.assertIs(
            an_int_tensor.larray, underlying_int_torch_tensor
        )  # test torch tensor in-place
        self.assertTrue(
            ht.equal(an_int_scalar, self.an_int_scalar)
        )  # test if other input is unchanged
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int_tensor.bitwise_xor_(another_int_tensor), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(another_int_tensor, self.another_int_tensor))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(ht.equal(an_int_tensor.bitwise_xor_(an_int_vector), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_vector, self.an_int_vector))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(
            ht.equal(a_boolean_vector.bitwise_xor_(another_boolean_vector), boolean_result)
        )
        self.assertTrue(ht.equal(a_boolean_vector, boolean_result))
        self.assertIs(a_boolean_vector, a_boolean_vector_double)
        self.assertIs(a_boolean_vector.larray, underlying_boolean_torch_tensor)
        self.assertTrue(ht.equal(another_boolean_vector, self.another_boolean_vector))
        underlying_boolean_torch_tensor.copy_(self.a_boolean_vector.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(an_int_tensor.__ixor__(an_int_scalar), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, self.an_int_scalar))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        an_int_tensor ^= an_int_scalar
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, self.an_int_scalar))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([1], split=0)
        a ^= b
        self.assertTrue(ht.equal(a, ht.array([0, 3])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0, dtype=int) * 11
        b = ht.ones(10, split=0, dtype=int) * 13
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a ^= b[1:]
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0, dtype=int) * 11  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a ^= b[1:-1]
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0, dtype=int) * 11
        b = ht.ones((1, 10), split=0, dtype=int) * 13
        a_lshape = a.lshape
        a ^= b
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0, dtype=int) * 11
        b = ht.ones(10, split=None, dtype=int) * 13
        a_lshape = a.lshape
        a ^= b
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12, dtype=int).reshape((4, 3), new_split=1) * 11
        b = ht.ones(3, split=0, dtype=int) * 13
        a_lshape = a.lshape
        a ^= b
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            a_tensor.bitwise_xor_(another_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(ValueError):
            an_int_tensor.bitwise_xor_(another_int_vector)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.bitwise_xor_(erroneous_type)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            an_int_scalar.bitwise_xor_(an_int_tensor)
        an_int_scalar = self.an_int_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.bitwise_xor_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0, dtype=int)
        b = ht.zeros((2, 10), split=0, dtype=int)
        with self.assertRaises(ValueError):
            a ^= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None, dtype=int)
        b = ht.ones(10, split=0, dtype=int)
        with self.assertRaises(ValueError):
            a ^= b

        a = ht.ones(9, dtype=int).reshape((3, 3), new_split=0) * 11
        b = ht.ones(9, dtype=int).reshape((3, 3), new_split=1) * 13
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a ^= b
            self.assertTrue((a == 6).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a ^= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.int32)
        b = ht.ones(3, dtype=ht.int64)
        with self.assertRaises(TypeError):
            a ^= b

    def test_copysign(self):
        a = ht.array([3, 2, -8, -2, 4])
        b = ht.array([3.0, 2.0, -8.0, -2.0, 4.0])
        result = ht.array([3.0, 2.0, 8.0, 2.0, 4.0])

        self.assertAlmostEqual(ht.mean(ht.copysign(a, 1.0) - result).item(), 0.0)
        self.assertAlmostEqual(ht.mean(ht.copysign(a, -1.0) + result).item(), 0.0)
        self.assertAlmostEqual(ht.mean(ht.copysign(a, a) - a).item(), 0.0)
        self.assertAlmostEqual(ht.mean(ht.copysign(b, b) - b).item(), 0.0)
        self.assertEqual(ht.copysign(a, 1.0).dtype, ht.float32)
        self.assertEqual(ht.copysign(b, 1.0).dtype, ht.float32)
        self.assertNotEqual(ht.copysign(a, 1.0).dtype, ht.int64)

        with self.assertRaises(TypeError):
            ht.copysign(a, "T")
        with self.assertRaises(TypeError):
            ht.copysign(a, 1j)

    def test_copysign_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        another_vector = self.another_vector.copy()
        a_tensor = self.a_tensor.copy()
        an_int_tensor = self.an_int_tensor.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        a_float_vector = a_float_vector_double = ht.array([3.0, 2.0, -8.0, -2.0, 4.0])
        result = ht.array([3.0, 2.0, 8.0, 2.0, 4.0])
        another_float_vector = ht.array([-1.0, 2.0, -3.0, 4.0, -5.0])
        another_result = ht.array([-3.0, 2.0, -8.0, 2.0, -4.0])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_float_vector.larray

        # Check for some possible combinations of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_float_vector.copysign_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_float_vector, result))  # test result in-place
        self.assertIs(a_float_vector, a_float_vector_double)  # test DNDarray in-place
        self.assertIs(a_float_vector.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(ht.array([3.0, 2.0, -8.0, -2.0, 4.0]).larray)  # reset

        self.assertTrue(ht.equal(a_float_vector.copysign_(-a_scalar), -result))
        self.assertTrue(ht.equal(a_float_vector, -result))
        self.assertIs(a_float_vector, a_float_vector_double)
        self.assertIs(a_float_vector.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(-a_scalar, -self.a_scalar))
        underlying_torch_tensor.copy_(ht.array([3.0, 2.0, -8.0, -2.0, 4.0]).larray)

        self.assertTrue(ht.equal(a_float_vector.copysign_(another_float_vector), another_result))
        self.assertTrue(ht.equal(a_float_vector, another_result))
        self.assertIs(a_float_vector, a_float_vector_double)
        self.assertIs(a_float_vector.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_float_vector, ht.array([-1.0, 2.0, -3.0, 4.0, -5.0])))
        underlying_torch_tensor.copy_(ht.array([3.0, 2.0, -8.0, -2.0, 4.0]).larray)

        # Single element split
        a = a_double = ht.array([1.0, 2.0], split=0)
        b = ht.array([1.0], split=0)
        a.copysign_(b)
        self.assertTrue(ht.equal(a, ht.array([1.0, 2.0])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0, dtype=float) * 2
        b = ht.ones(10, split=0, dtype=float) * (-1)
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a.copysign_(b[1:])
        self.assertTrue((a == -2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0, dtype=float) * 2  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a.copysign_(b[1:-1])
        self.assertTrue((a == -2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0, dtype=float) * 2
        b = ht.ones((1, 10), split=0, dtype=float) * (-1)
        a_lshape = a.lshape
        a.copysign_(b)
        self.assertTrue((a == -2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0, dtype=float) * 2
        b = ht.ones(10, split=None, dtype=float) * (-1)
        a_lshape = a.lshape
        a.copysign_(b)
        self.assertTrue((a == -2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12, dtype=float).reshape((4, 3), new_split=1) * 2
        b = ht.ones(3, split=0, dtype=float) * (-1)
        a_lshape = a.lshape
        a.copysign_(b)
        self.assertTrue((a == -2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.copysign_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.copysign_(a_tensor)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.copysign_(an_int_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.copysign_(1j)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.copysign_(an_int_tensor)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.copysign_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.copysign_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.copysign_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0, dtype=float) * 2
        b = ht.ones((2, 10), split=0, dtype=float) * (-1)
        with self.assertRaises(ValueError):
            a.copysign_(b)

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None, dtype=float) * 2
        b = ht.ones(10, split=0, dtype=float) * (-1)
        with self.assertRaises(ValueError):
            a.copysign_(b)

        a = ht.ones(9, dtype=float).reshape((3, 3), new_split=0) * 2
        b = ht.ones(9, dtype=float).reshape((3, 3), new_split=1) * (-1)
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a.copysign_(b)
            self.assertTrue((a == -2).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a.copysign_(b)

    def test_cumprod(self):
        a = ht.full((2, 4), 2, dtype=ht.int32)
        result = ht.array([[2, 4, 8, 16], [2, 4, 8, 16]], dtype=ht.int32)

        # split = None
        cumprod = ht.cumprod(a, 1)
        self.assertTrue(ht.equal(cumprod, result))

        # Alias
        cumprod = ht.cumproduct(a, 1)
        self.assertTrue(ht.equal(cumprod, result))

        a = ht.full((4, 2), 2, dtype=ht.int64, split=0)
        result = ht.array([[2, 2], [4, 4], [8, 8], [16, 16]], dtype=ht.int64, split=0)

        cumprod = ht.cumprod(a, 0)
        self.assertTrue(ht.equal(cumprod, result))

        # 3D
        out = ht.empty((2, 2, 2), dtype=ht.float32, split=0)

        a = ht.full((2, 2, 2), 2, split=0)
        result = ht.array([[[2, 2], [2, 2]], [[4, 4], [4, 4]]], dtype=ht.float32, split=0)

        cumprod = ht.cumprod(a, 0, out=out)
        self.assertTrue(ht.equal(cumprod, out))
        self.assertTrue(ht.equal(cumprod, result))

        a = ht.full((2, 2, 2), 2, dtype=ht.int32, split=1)
        result = ht.array([[[2, 2], [4, 4]], [[2, 2], [4, 4]]], dtype=ht.float32, split=1)

        cumprod = ht.cumprod(a, 1, dtype=ht.float64)
        self.assertTrue(ht.equal(cumprod, result))

        a = ht.full((2, 2, 2), 2, dtype=ht.float32, split=2)
        result = ht.array([[[2, 4], [2, 4]], [[2, 4], [2, 4]]], dtype=ht.float32, split=2)

        cumprod = ht.cumprod(a, 2)
        self.assertTrue(ht.equal(cumprod, result))

        with self.assertRaises(NotImplementedError):
            ht.cumprod(ht.ones((2, 2)), axis=None)
        with self.assertRaises(TypeError):
            ht.cumprod(ht.ones((2, 2)), axis="1")
        with self.assertRaises(ValueError):
            ht.cumprod(a, 2, out=out)
        with self.assertRaises(ValueError):
            ht.cumprod(ht.ones((2, 2)), 2)

    def test_cumprod_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        # tests for split=None
        a_tensor = a_tensor_double = ht.full((2, 4), 2, dtype=ht.int32)
        result_0 = ht.array([[2, 2, 2, 2], [4, 4, 4, 4]], dtype=ht.int32)
        result_1 = ht.array([[2, 4, 8, 16], [2, 4, 8, 16]], dtype=ht.int32)

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for some possible combinations of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.cumprod_(0), result_0))  # test result
        self.assertTrue(ht.equal(a_tensor, result_0))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        underlying_torch_tensor.copy_(ht.full((2, 4), 2, dtype=ht.int32).larray)  # reset

        self.assertTrue(ht.equal(a_tensor.cumprod_(1), result_1))
        self.assertTrue(ht.equal(a_tensor, result_1))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        underlying_torch_tensor.copy_(ht.full((2, 4), 2, dtype=ht.int32).larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.cumproduct_(0), result_0))
        self.assertTrue(ht.equal(a_tensor, result_0))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        underlying_torch_tensor.copy_(ht.full((2, 4), 2, dtype=ht.int32).larray)

        # tests for splits in 2D
        a_split_tensor = a_split_tensor_double = a_tensor.resplit(0)
        underlying_split_torch_tensor = a_split_tensor.larray
        self.assertTrue(ht.equal(a_split_tensor.cumprod_(0), result_0))
        self.assertTrue(ht.equal(a_split_tensor, result_0))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        a_split_tensor = a_split_tensor_double = a_tensor.resplit(1)
        underlying_split_torch_tensor = a_split_tensor.larray
        self.assertTrue(ht.equal(a_split_tensor.cumprod_(1), result_1))
        self.assertTrue(ht.equal(a_split_tensor, result_1))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        # tests for splits in 3D
        a_split_tensor = a_split_tensor_double = ht.full((2, 2, 2), 2, dtype=ht.int32, split=0)
        underlying_split_torch_tensor = a_split_tensor.larray
        result_0 = ht.array([[[2, 2], [2, 2]], [[4, 4], [4, 4]]], dtype=ht.int32, split=0)
        self.assertTrue(ht.equal(a_split_tensor.cumprod_(0), result_0))
        self.assertTrue(ht.equal(a_split_tensor, result_0))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        a_split_tensor = a_split_tensor_double = ht.full((2, 2, 2), 2, dtype=ht.float32, split=1)
        underlying_split_torch_tensor = a_split_tensor.larray
        result_1 = ht.array([[[2, 2], [4, 4]], [[2, 2], [4, 4]]], dtype=ht.float32, split=1)
        self.assertTrue(ht.equal(a_split_tensor.cumprod_(1), result_1))
        self.assertTrue(ht.equal(a_split_tensor, result_1))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        a_split_tensor = a_split_tensor_double = ht.full((2, 2, 2), 2, split=2)
        underlying_split_torch_tensor = a_split_tensor.larray
        result_2 = ht.array([[[2, 4], [2, 4]], [[2, 4], [2, 4]]], split=2)
        self.assertTrue(ht.equal(a_split_tensor.cumprod_(2), result_2))
        self.assertTrue(ht.equal(a_split_tensor, result_2))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        # Single element split
        a = a_double = ht.array([2, 2], split=0)
        a.cumprod_(0)
        self.assertTrue(ht.equal(a, ht.array([2, 4])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(NotImplementedError):
            a_tensor.cumprod_(axis=None)
        with self.assertRaises(TypeError):
            a_tensor.cumprod_(axis="1")
        with self.assertRaises(TypeError):
            a_tensor.cumprod_(0.5)
        with self.assertRaises(ValueError):
            a_tensor.cumprod_(2)
        with self.assertRaises(TypeError):
            a_tensor.cumprod_(a_tensor)
        with self.assertRaises(ValueError):
            a_tensor.cumprod_(erroneous_type)  # (2, 2) is interpreted as invalid axis here
        with self.assertRaises(AttributeError):
            a_scalar.cumprod_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.cumprod_("s")
        a_string = self.a_string  # reset

    def test_cumsum(self):
        a = ht.ones((2, 4), dtype=ht.int32)
        result = ht.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=ht.int32)

        # split = None
        cumsum = ht.cumsum(a, 1)
        self.assertTrue(ht.equal(cumsum, result))

        a = ht.ones((4, 2), dtype=ht.int64, split=0)
        result = ht.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ht.int64, split=0)

        cumsum = ht.cumsum(a, 0)
        self.assertTrue(ht.equal(cumsum, result))

        # 3D
        out = ht.empty((2, 2, 2), dtype=ht.float32, split=0)

        a = ht.ones((2, 2, 2), split=0)
        result = ht.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=ht.float32, split=0)

        cumsum = ht.cumsum(a, 0, out=out)
        self.assertTrue(ht.equal(cumsum, out))
        self.assertTrue(ht.equal(cumsum, result))

        a = ht.ones((2, 2, 2), dtype=ht.int32, split=1)
        result = ht.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]], dtype=ht.float32, split=1)

        cumsum = ht.cumsum(a, 1, dtype=ht.float64)
        self.assertTrue(ht.equal(cumsum, result))

        a = ht.ones((2, 2, 2), dtype=ht.float32, split=2)
        result = ht.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], dtype=ht.float32, split=2)

        cumsum = ht.cumsum(a, 2)
        self.assertTrue(ht.equal(cumsum, result))

        with self.assertRaises(NotImplementedError):
            ht.cumsum(ht.ones((2, 2)), axis=None)
        with self.assertRaises(TypeError):
            ht.cumsum(ht.ones((2, 2)), axis="1")
        with self.assertRaises(ValueError):
            ht.cumsum(a, 2, out=out)
        with self.assertRaises(ValueError):
            ht.cumsum(ht.ones((2, 2)), 2)

    def test_cumsum_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        # tests for split=None
        a_tensor = a_tensor_double = ht.ones((2, 4), dtype=ht.int32)
        result_0 = ht.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=ht.int32)
        result_1 = ht.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=ht.int32)

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for some possible combinations of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.cumsum_(0), result_0))  # test result
        self.assertTrue(ht.equal(a_tensor, result_0))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        underlying_torch_tensor.copy_(ht.ones((2, 4), dtype=ht.int32).larray)  # reset

        self.assertTrue(ht.equal(a_tensor.cumsum_(1), result_1))
        self.assertTrue(ht.equal(a_tensor, result_1))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        underlying_torch_tensor.copy_(ht.ones((2, 4), dtype=ht.int32).larray)

        # tests for splits in 2D
        a_split_tensor = a_split_tensor_double = a_tensor.resplit(0)
        underlying_split_torch_tensor = a_split_tensor.larray
        self.assertTrue(ht.equal(a_split_tensor.cumsum_(0), result_0))
        self.assertTrue(ht.equal(a_split_tensor, result_0))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        a_split_tensor = a_split_tensor_double = a_tensor.resplit(1)
        underlying_split_torch_tensor = a_split_tensor.larray
        self.assertTrue(ht.equal(a_split_tensor.cumsum_(1), result_1))
        self.assertTrue(ht.equal(a_split_tensor, result_1))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        # tests for splits in 3D
        a_split_tensor = a_split_tensor_double = ht.ones((2, 2, 2), dtype=ht.int32, split=0)
        underlying_split_torch_tensor = a_split_tensor.larray
        result_0 = ht.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=ht.int32, split=0)
        self.assertTrue(ht.equal(a_split_tensor.cumsum_(0), result_0))
        self.assertTrue(ht.equal(a_split_tensor, result_0))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        a_split_tensor = a_split_tensor_double = ht.ones((2, 2, 2), dtype=ht.float32, split=1)
        underlying_split_torch_tensor = a_split_tensor.larray
        result_1 = ht.array([[[1, 1], [2, 2]], [[1, 1], [2, 2]]], dtype=ht.float32, split=1)
        self.assertTrue(ht.equal(a_split_tensor.cumsum_(1), result_1))
        self.assertTrue(ht.equal(a_split_tensor, result_1))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        a_split_tensor = a_split_tensor_double = ht.ones((2, 2, 2), split=2)
        underlying_split_torch_tensor = a_split_tensor.larray
        result_2 = ht.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], split=2)
        self.assertTrue(ht.equal(a_split_tensor.cumsum_(2), result_2))
        self.assertTrue(ht.equal(a_split_tensor, result_2))
        self.assertIs(a_split_tensor, a_split_tensor_double)
        self.assertIs(a_split_tensor.larray, underlying_split_torch_tensor)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        a.cumsum_(0)
        self.assertTrue(ht.equal(a, ht.array([1, 3])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(NotImplementedError):
            a_tensor.cumsum_(axis=None)
        with self.assertRaises(TypeError):
            a_tensor.cumsum_(axis="1")
        with self.assertRaises(TypeError):
            a_tensor.cumsum_(0.5)
        with self.assertRaises(ValueError):
            a_tensor.cumsum_(2)
        with self.assertRaises(TypeError):
            a_tensor.cumsum_(a_tensor)
        with self.assertRaises(ValueError):
            a_tensor.cumsum_(erroneous_type)  # (2, 2) is interpreted as invalid axis here
        with self.assertRaises(AttributeError):
            a_scalar.cumsum_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.cumsum_("s")
        a_string = self.a_string  # reset

    def test_diff(self):
        ht_array = ht.random.rand(20, 20, 20, split=None)
        arb_slice = [0] * 3
        for dim in range(0, 3):  # loop over 3 dimensions
            arb_slice[dim] = slice(None)
            tup_arb = tuple(arb_slice)
            np_array = ht_array[tup_arb].numpy()
            for ax in range(dim + 1):  # loop over the possible axis values
                for sp in range(dim + 1):  # loop over the possible split values
                    lp_array = ht.manipulations.resplit(ht_array[tup_arb], sp)
                    # loop to 3 for the number of times to do the diff
                    for nl in range(1, 4):
                        # only generating the number once and then
                        ht_diff = ht.diff(lp_array, n=nl, axis=ax)
                        np_diff = ht.array(np.diff(np_array, n=nl, axis=ax))

                        self.assertTrue(ht.equal(ht_diff, np_diff))
                        self.assertEqual(ht_diff.split, sp)
                        self.assertEqual(ht_diff.dtype, lp_array.dtype)

                        # test prepend/append. Note heat's intuitive casting vs. numpy's safe casting
                        append_shape = lp_array.gshape[:ax] + (1,) + lp_array.gshape[ax + 1 :]
                        ht_append = ht.ones(
                            append_shape, dtype=lp_array.dtype, split=lp_array.split
                        )

                        ht_diff_pend = ht.diff(lp_array, n=nl, axis=ax, prepend=0, append=ht_append)
                        np_append = np.ones(append_shape, dtype=lp_array.larray.cpu().numpy().dtype)
                        np_diff_pend = ht.array(
                            np.diff(np_array, n=nl, axis=ax, prepend=0, append=np_append)
                        )
                        self.assertTrue(ht.equal(ht_diff_pend, np_diff_pend))
                        self.assertEqual(ht_diff_pend.split, sp)
                        self.assertEqual(ht_diff_pend.dtype, ht.float64)

        np_array = ht_array.numpy()
        ht_diff = ht.diff(ht_array, n=2)
        np_diff = ht.array(np.diff(np_array, n=2))
        self.assertTrue(ht.equal(ht_diff, np_diff))
        self.assertEqual(ht_diff.split, None)
        self.assertEqual(ht_diff.dtype, ht_array.dtype)

        ht_array = ht.random.rand(20, 20, 20, split=1, dtype=ht.float64)
        np_array = ht_array.copy().numpy()
        ht_diff = ht.diff(ht_array, n=2)
        np_diff = ht.array(np.diff(np_array, n=2))
        self.assertTrue(ht.equal(ht_diff, np_diff))
        self.assertEqual(ht_diff.split, 1)
        self.assertEqual(ht_diff.dtype, ht_array.dtype)

        # test n=0
        ht_diff = ht.diff(ht_array, n=0)
        self.assertTrue(ht.equal(ht_diff, ht_array))

        # edge cases
        ht_arr = ht.ones((2, 1, 2), split=0)
        ht_diff = ht.diff(ht_arr, axis=1)
        self.assertEqual(ht_diff.gshape, (2, 0, 2))

        ht_arr = ht.ones((2, 1, 2), split=1)
        ht_diff = ht.diff(ht_arr, axis=1)
        self.assertEqual(ht_diff.gshape, (2, 0, 2))

        # raises
        with self.assertRaises(ValueError):
            ht.diff(ht_array, n=-2)
        with self.assertRaises(TypeError):
            ht.diff(ht_array, axis="string")
        with self.assertRaises(TypeError):
            ht.diff("string", axis=2)
        t_prepend = torch.zeros(ht_array.gshape)
        with self.assertRaises(TypeError):
            ht.diff(ht_array, prepend=t_prepend)
        append_wrong_shape = ht.ones(ht_array.gshape)
        with self.assertRaises(ValueError):
            ht.diff(ht_array, axis=0, append=append_wrong_shape)

    def test_div(self):
        result = ht.array([[0.5, 1.0], [1.5, 2.0]])
        reciprocal = ht.array([[2.0, 1.0], [2.0 / 3.0, 0.5]])

        self.assertTrue(ht.equal(ht.div(self.a_scalar, self.a_scalar), ht.float32(1.0)))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.div(self.a_scalar, self.a_tensor), reciprocal))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.div(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.div(self.a_split_tensor, self.a_tensor), reciprocal))

        self.assertTrue(ht.equal(self.a_tensor / self.a_scalar, result))
        self.assertTrue(ht.equal(self.a_scalar / self.a_tensor, reciprocal))

        a = out = ht.empty((2, 2))
        ht.div(self.a_tensor, self.a_scalar, out=out)
        self.assertTrue(ht.equal(out, result))
        self.assertIs(a, out)
        b = ht.array([[1.0, 2.0], [3.0, 4.0]])
        ht.div(b, self.another_tensor, out=b)
        self.assertTrue(ht.equal(b, result))
        out = ht.empty((2, 2), split=self.a_split_tensor.split)
        ht.div(self.a_split_tensor, self.a_tensor, out=out)
        self.assertTrue(ht.equal(out, reciprocal))
        self.assertEqual(self.a_split_tensor.split, out.split)

        result_where = ht.array([[1.0, 2.0], [1.5, 2.0]])
        self.assertTrue(
            ht.equal(
                ht.div(self.a_tensor, self.a_scalar, where=self.a_tensor > 2)[1, :],
                result_where[1, :],
            )
        )

        a = self.a_tensor.copy()
        ht.div(a, self.a_scalar, out=a, where=a > 2)
        self.assertTrue(ht.equal(a, result_where))
        out = ht.array([[1.0, 2.0], [3.0, 4.0]], split=1)
        where = ht.array([[True, True], [False, True]], split=None)
        ht.div(out, self.another_tensor, out=out, where=where)
        self.assertTrue(ht.equal(out, ht.array([[0.5, 1.0], [3.0, 2.0]])))
        self.assertEqual(1, out.split)
        out = ht.array([[1.0, 2.0], [3.0, 4.0]], split=0)
        where.resplit_(0)
        ht.div(out, self.another_tensor, out=out, where=where)
        self.assertTrue(ht.equal(out, ht.array([[0.5, 1.0], [3.0, 2.0]])))
        self.assertEqual(0, out.split)

        result_where_broadcasted = ht.array([[1.0, 1.0], [3.0, 2.0]])
        a = self.a_tensor.copy()
        ht.div(a, self.a_scalar, out=a, where=ht.array([False, True]))
        self.assertTrue(ht.equal(a, result_where_broadcasted))
        a = self.a_tensor.copy().resplit_(0)
        ht.div(a, self.a_scalar, out=a, where=ht.array([False, True], split=0))
        self.assertTrue(ht.equal(a, result_where_broadcasted))
        self.assertEqual(0, a.split)

        with self.assertRaises(ValueError):
            ht.div(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.div(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.div("T", "s")
        with self.assertRaises(ValueError):
            ht.div(self.a_split_tensor, self.a_tensor, out=ht.empty((2, 2), split=None))
        if a.comm.size > 1:
            with self.assertRaises(NotImplementedError):
                ht.div(
                    self.a_split_tensor,
                    self.a_tensor,
                    where=ht.array([[True, False], [False, True]], split=1),
                )
        with self.assertRaises(TypeError):
            self.a_tensor / "T"

    def test_div_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[0.5, 1.0], [1.5, 2.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.div_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.div_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.div_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.div_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.divide_(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.__itruediv__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor /= a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Single element split
        a = a_double = ht.array([2.0, 4.0], split=0)
        b = ht.array([2.0], split=0)
        a /= b
        self.assertTrue(ht.equal(a, ht.array([1.0, 2.0])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 2
        b = ht.ones(10, split=0) * 2
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a /= b[1:]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 2  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a /= b[1:-1]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 2
        b = ht.ones((1, 10), split=0) * 2
        a_lshape = a.lshape
        a /= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 2
        b = ht.ones(10, split=None) * 2
        a_lshape = a.lshape
        a /= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 2
        b = ht.ones(3, split=0) * 2
        a_lshape = a.lshape
        a /= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.div_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.div_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.div_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.div_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a /= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a /= b

        a = ht.ones(9).reshape((3, 3), new_split=0) * 2
        b = ht.ones(9).reshape((3, 3), new_split=1) * 2
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a /= b
            self.assertTrue((a == 1).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a /= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a /= b

    def test_divmod(self):
        # basic tests as floor_device and mod are tested separately
        result = (
            ht.array(
                [
                    [
                        0.0,
                        1.0,
                    ],
                    [1.0, 2.0],
                ]
            ),
            ht.array([[1.0, 0.0], [1.0, 0.0]]),
        )
        dm = ht.divmod(self.a_tensor, self.a_scalar)

        self.assertIsInstance(dm, tuple)
        self.assertTrue(ht.equal(ht.divmod(self.a_tensor, self.a_scalar)[0], result[0]))
        self.assertTrue(ht.equal(ht.divmod(self.a_tensor, self.a_scalar)[1], result[1]))

        result = (ht.array([1.0, 1.0]), ht.array([0.0, 0.0]))
        dm = divmod(self.a_scalar, self.a_vector)
        self.assertTrue(ht.equal(dm[0], result[0]))
        self.assertTrue(ht.equal(dm[1], result[1]))

        # out parameter
        out = (ht.empty((2, 2), split=0), ht.empty((2, 2), split=0))
        ht.divmod(self.a_split_tensor, self.a_scalar, out=out)
        self.assertTrue(ht.equal(out[0], ht.array([[1.0, 1.0], [1.0, 1.0]])))
        self.assertTrue(ht.equal(out[1], ht.array([[0.0, 0.0], [0.0, 0.0]])))

        with self.assertRaises(TypeError):
            divmod(self.another_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.divmod(ht.zeros((2, 2)), ht.zeros((2, 2)), out=1)
        with self.assertRaises(ValueError):
            ht.divmod(ht.zeros((2, 2)), ht.zeros((2, 2)), out=(1, 2, 3))
        with self.assertRaises(TypeError):
            ht.divmod(
                ht.zeros((2, 2)),
                ht.zeros((2, 2)),
                ht.empty((2, 2)),
                ht.empty((2, 2)),
                out=(ht.empty((2, 2)), None),
            )
        with self.assertRaises(TypeError):
            ht.divmod(
                ht.zeros((2, 2)),
                ht.zeros((2, 2)),
                ht.empty((2, 2)),
                ht.empty((2, 2)),
                out=(None, ht.empty((2, 2))),
            )
        with self.assertRaises(TypeError):
            divmod(ht.zeros((2, 2)), "T")

    def test_floordiv(self):
        result = ht.array([[0.0, 1.0], [1.0, 2.0]])
        reversal_result = ht.array([[2.0, 1.0], [0.0, 0.0]])

        self.assertTrue(ht.equal(ht.floordiv(self.a_scalar, self.a_scalar), ht.float32(1.0)))
        self.assertTrue(ht.equal(ht.floordiv(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.floordiv(self.a_scalar, self.a_tensor), reversal_result))
        self.assertTrue(ht.equal(ht.floordiv(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.floordiv(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.floordiv(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.floordiv(self.a_split_tensor, self.a_tensor), reversal_result))

        self.assertTrue(ht.equal(self.a_tensor // self.a_scalar, result))
        self.assertTrue(ht.equal(self.a_scalar // self.a_tensor, reversal_result))

        # Single element split
        a = ht.array([1], split=0)
        b = ht.array([1, 2], split=0)
        c = ht.floordiv(a, b)
        self.assertTrue(ht.equal(c, ht.array([1, 0])))
        if c.comm.size > 1:
            if c.comm.rank < 2:
                self.assertEqual(c.larray.size()[0], 1)
            else:
                self.assertEqual(c.larray.size()[0], 0)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=0) * 2
        c = a[:-1] // b[1:]
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == a[:-1].lshape)

        c = a[1:-1] // b[1:-1]  # test unbalanced
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == a[1:-1].lshape)

        # test one unsplit
        a = ht.ones(10, split=None) * 3
        b = ht.ones(10, split=0) * 2
        c = a[:-1] // b[1:]
        self.assertTrue((c == 1).all())
        self.assertEqual(c.lshape, b[1:].lshape)
        c = b[:-1] // a[1:]
        self.assertTrue((c == 0).all())
        self.assertEqual(c.lshape, b[:-1].lshape)

        # broadcast in split dimension
        a = ht.ones((1, 10), split=0) * 3
        b = ht.ones((2, 10), split=0) * 2
        c = a // b
        self.assertTrue((c == 1).all())
        self.assertTrue(c.lshape == b.lshape)
        c = b // a
        self.assertTrue((c == 0).all())
        self.assertTrue(c.lshape == b.lshape)

        with self.assertRaises(ValueError):
            ht.floordiv(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.floordiv(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.floordiv("T", "s")
        with self.assertRaises(TypeError):
            "T" // self.a_tensor

    def test_floordiv_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[0.0, 1.0], [1.0, 2.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.floordiv_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.floordiv_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.floordiv_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.floordiv_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.floor_divide_(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.__ifloordiv__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor //= a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Single element split
        a = a_double = ht.array([2, 5], split=0)
        b = ht.array([2], split=0)
        a //= b
        self.assertTrue(ht.equal(a, ht.array([1, 2])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=0) * 2
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a //= b[1:]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 3  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a //= b[1:-1]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 3
        b = ht.ones((1, 10), split=0) * 2
        a_lshape = a.lshape
        a //= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=None) * 2
        a_lshape = a.lshape
        a //= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 3
        b = ht.ones(3, split=0) * 2
        a_lshape = a.lshape
        a //= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.floordiv_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.floordiv_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.floordiv_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.floordiv_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a //= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a //= b

        a = ht.ones(9).reshape((3, 3), new_split=0) * 3
        b = ht.ones(9).reshape((3, 3), new_split=1) * 2
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a //= b
            self.assertTrue((a == 1).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a //= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a //= b

    def test_fmod(self):
        result = ht.array([[1.0, 0.0], [1.0, 0.0]])
        another_int_tensor = ht.array([[5, 3], [4, 1]])
        integer_result = ht.array([[1, 1], [0, 1]])
        commutated_result = ht.array([[0.0, 0.0], [2.0, 2.0]])
        zero_tensor = ht.zeros((2, 2))

        a_float = ht.array([5.3])
        another_float = ht.array([1.9])
        result_float = ht.array([1.5])

        self.assertTrue(ht.equal(ht.fmod(self.a_scalar, self.a_scalar), ht.float32(0.0)))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.a_tensor), zero_tensor))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.fmod(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.fmod(another_int_tensor, self.an_int_scalar), integer_result))
        self.assertTrue(ht.equal(ht.fmod(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.fmod(self.a_split_tensor, self.a_tensor), commutated_result))
        self.assertTrue(ht.allclose(ht.fmod(a_float, another_float), result_float))

        with self.assertRaises(ValueError):
            ht.fmod(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.fmod(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.fmod("T", "s")

    def test_fmod_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[1.0, 0.0], [1.0, 0.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.fmod_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.fmod_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.fmod_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.fmod_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Difference compared to 'mod_'/'remainder_'
        vector_3 = ht.array([0.4, -3.0])
        result_2 = ht.array([[0.2, 2.0], [0.2, 1.0]])  # mod_ result == [[0.2, -1.0], [0.2, -2.0]]
        result_3 = ht.array([0.4, -1.0])  # mod_ result == [0.4, 1.0]

        self.assertTrue(ht.allclose(a_tensor.fmod_(vector_3), result_2))
        self.assertTrue(ht.allclose(a_tensor, result_2))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(vector_3, ht.array([0.4, -3.0])))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.allclose(vector_3.fmod_(a_vector), result_3))
        self.assertTrue(ht.allclose(vector_3, result_3))

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([2], split=0)
        a.fmod_(b)
        self.assertTrue(ht.equal(a, ht.array([1, 0])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=0) * 2
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a.fmod_(b[1:])
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 3  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a.fmod_(b[1:-1])
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 3
        b = ht.ones((1, 10), split=0) * 2
        a_lshape = a.lshape
        a.fmod_(b)
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=None) * 2
        a_lshape = a.lshape
        a.fmod_(b)
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 3
        b = ht.ones(3, split=0) * 2
        a_lshape = a.lshape
        a.fmod_(b)
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.fmod_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.fmod_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.fmod_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.fmod_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a.fmod_(b)

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a.fmod_(b)

        a = ht.ones(9).reshape((3, 3), new_split=0) * 3
        b = ht.ones(9).reshape((3, 3), new_split=1) * 2
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a.fmod_(b)
            self.assertTrue((a == 1).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a.fmod_(b)

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a.fmod_(b)

    def test_gcd(self):
        a = ht.array([5, 10, 15])
        b = ht.array([3, 4, 5])
        c = ht.array([3.0, 4.0, 5.0])
        result = ht.array([1, 2, 5])

        self.assertTrue(ht.equal(ht.gcd(a, b), result))
        self.assertTrue(ht.equal(ht.gcd(a, a), a))
        self.assertEqual(ht.gcd(a, b).dtype, ht.int64)

        with self.assertRaises(TypeError):
            ht.gcd(a, c)
        with self.assertRaises(ValueError):
            ht.gcd(a, ht.array([15, 20]))

    def test_gcd_(self):
        # Copies of class variables for the in-place operations
        a_tensor = self.a_tensor.copy()
        an_int_scalar = self.an_int_scalar
        an_int_vector = self.an_int_vector.copy()
        another_int_vector = self.another_int_vector.copy()
        an_int_tensor = an_int_tensor_double = self.an_int_tensor.copy()
        another_int_tensor = self.another_int_tensor.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[1, 2], [1, 2]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_int_torch_tensor = an_int_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(an_int_tensor.gcd_(an_int_scalar), result))  # test result
        self.assertTrue(ht.equal(an_int_tensor, result))  # test result in-place
        self.assertIs(an_int_tensor, an_int_tensor_double)  # test DNDarray in-place
        self.assertIs(
            an_int_tensor.larray, underlying_int_torch_tensor
        )  # test torch tensor in-place
        self.assertTrue(
            ht.equal(an_int_scalar, self.an_int_scalar)
        )  # test if other input is unchanged
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int_tensor.gcd_(another_int_tensor), result))
        self.assertTrue(ht.equal(an_int_tensor, result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(another_int_tensor, self.another_int_tensor))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(ht.equal(an_int_tensor.gcd_(an_int_vector), result))
        self.assertTrue(ht.equal(an_int_tensor, result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_vector, self.an_int_vector))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([1], split=0)
        a.gcd_(b)
        self.assertTrue(ht.equal(a, ht.array([1, 1])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0, dtype=int) * 6
        b = ht.ones(10, split=0, dtype=int) * 4
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a.gcd_(b[1:])
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0, dtype=int) * 6  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a.gcd_(b[1:-1])
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), dtype=int, split=0) * 6
        b = ht.ones((1, 10), dtype=int, split=0) * 4
        a_lshape = a.lshape
        a.gcd_(b)
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, dtype=int, split=0) * 6
        b = ht.ones(10, dtype=int, split=None) * 4
        a_lshape = a.lshape
        a.gcd_(b)
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12, dtype=int).reshape((4, 3), new_split=1) * 6
        b = ht.ones(3, dtype=int, split=0) * 4
        a_lshape = a.lshape
        a.gcd_(b)
        self.assertTrue((a == 2).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            an_int_tensor.gcd_(a_tensor)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.gcd_(an_int_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(ValueError):
            an_int_tensor.gcd_(another_int_vector)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.gcd_(erroneous_type)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            an_int_scalar.gcd_(an_int_tensor)
        an_int_scalar = self.an_int_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.gcd_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), dtype=int, split=0) * 6
        b = ht.ones((2, 10), dtype=int, split=0) * 4
        with self.assertRaises(ValueError):
            a.gcd_(b)

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, dtype=int, split=None) * 6
        b = ht.ones(10, dtype=int, split=0) * 4
        with self.assertRaises(ValueError):
            a.gcd_(b)

        a = ht.ones(9, dtype=int).reshape((3, 3), new_split=0) * 6
        b = ht.ones(9, dtype=int).reshape((3, 3), new_split=1) * 4
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a.gcd_(b)
            self.assertTrue((a == 2).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a.gcd_(b)

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32) * 6
        b = ht.ones(3, dtype=ht.float64) * 4
        with self.assertRaises(TypeError):
            a.gcd_(b)

    def test_hypot(self):
        a = ht.array([2.0])
        b = ht.array([1.0, 3.0, 5.0])
        gt = ht.array([5, 13, 29])
        result = (ht.hypot(a, b) ** 2).astype(ht.int64)

        self.assertTrue(ht.equal(gt, result))
        self.assertEqual(result.dtype, ht.int64)

        with self.assertRaises(TypeError):
            ht.hypot(a)
        with self.assertRaises(TypeError):
            ht.hypot("a", "b")
        with self.assertRaises(TypeError):
            ht.hypot(a.astype(ht.int32), b.astype(ht.int32))

    def test_hypot_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        another_vector = self.another_vector.copy()
        a_tensor = self.a_tensor.copy()
        an_int_vector = self.an_int_vector.copy()
        an_int_tensor = self.an_int_tensor.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        a_vector = a_vector_double = ht.array([1.0, 3.0, 5.0])
        gt = ht.array([5.0, 13.0, 29.0])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_vector.larray

        # Check for some possible combinations of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_vector.hypot_(a_scalar).pow_(2), gt))  # test result
        self.assertTrue(ht.equal(a_vector, gt))  # test result in-place
        self.assertIs(a_vector, a_vector_double)  # test DNDarray in-place
        self.assertIs(a_vector.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(ht.array([1.0, 3.0, 5.0]).larray)  # reset

        self.assertTrue(ht.equal(a_vector.hypot_(another_vector).pow_(2), gt))
        self.assertTrue(ht.equal(a_vector, gt))
        self.assertIs(a_vector, a_vector_double)
        self.assertIs(a_vector.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_vector, self.another_vector))
        underlying_torch_tensor.copy_(ht.array([1.0, 3.0, 5.0]).larray)

        # Single element split
        a = a_double = ht.array([1.0, 3.0], split=0)
        b = ht.array([2.0], split=0)
        a.hypot_(b).pow_(2)
        self.assertTrue(ht.equal(a, ht.array([5.0, 13.0])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0)
        b = ht.ones(10, split=0) * 2
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a.hypot_(b[1:]).pow_(2)
        self.assertTrue((a == 5).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0)  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a.hypot_(b[1:-1]).pow_(2)
        self.assertTrue((a == 5).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0)
        b = ht.ones((1, 10), split=0) * 2
        a_lshape = a.lshape
        a.hypot_(b).pow_(2)
        self.assertTrue((a == 5).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0)
        b = ht.ones(10, split=None) * 2
        a_lshape = a.lshape
        a.hypot_(b).pow_(2)
        self.assertTrue((a == 5).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1)
        b = ht.ones(3, split=0) * 2
        a_lshape = a.lshape
        a.hypot_(b).pow_(2)
        self.assertTrue((a == 5).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            an_int_tensor.hypot_(a_tensor)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.hypot_(an_int_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.hypot_(an_int_vector)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(ValueError):
            a_tensor.hypot_(a_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.hypot_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.hypot_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.hypot_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a.hypot_(b)

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a.hypot_(b)

        a = ht.ones(9).reshape((3, 3), new_split=0)
        b = ht.ones(9).reshape((3, 3), new_split=1) * 2
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a.hypot_(b).pow_(2)
            self.assertTrue((a == 5).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a.hypot_(b)

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a.hypot_(b)

    def test_invert(self):
        int8_tensor = ht.array([[0, 1], [2, -2]], dtype=ht.int8)
        uint8_tensor = ht.array([[23, 2], [45, 234]], dtype=ht.uint8)
        bool_tensor = ht.array([[False, True], [True, False]])
        float_tensor = ht.array([[0.4, 1.3], [1.3, -2.1]])
        int8_result = ht.array([[-1, -2], [-3, 1]])
        uint8_result = ht.array([[232, 253], [210, 21]])
        bool_result = ht.array([[True, False], [False, True]])

        self.assertTrue(ht.equal(ht.invert(int8_tensor), int8_result))
        self.assertTrue(ht.equal(ht.invert(int8_tensor.copy().resplit_(0)), int8_result))
        self.assertTrue(ht.equal(ht.invert(uint8_tensor), uint8_result))
        self.assertTrue(ht.equal(~bool_tensor, bool_result))

        with self.assertRaises(TypeError):
            ht.invert(float_tensor)

    def test_invert_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_boolean_vector = a_boolean_vector_double = self.a_boolean_vector.copy()
        a_tensor = self.a_tensor.copy()
        an_int_tensor = an_int_tensor_double = self.an_int_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        int_result = ht.array([[-2, -3], [-4, -5]], dtype=ht.int64)
        an_int8_tensor = an_int8_tensor_double = ht.array([[0, 1], [2, -2]], dtype=ht.int8)
        an_uint8_tensor = an_uint8_tensor_double = ht.array([[23, 2], [45, 234]], dtype=ht.uint8)
        int8_result = ht.array([[-1, -2], [-3, 1]])
        uint8_result = ht.array([[232, 253], [210, 21]])
        bool_result = ht.array([True, False, True, False])

        # We identify the underlying PyTorch objects to check whether operations are really in-place
        underlying_int_torch_tensor = an_int_tensor.larray
        underlying_int8_torch_tensor = an_int8_tensor.larray
        underlying_uint8_torch_tensor = an_uint8_tensor.larray
        underlying_boolean_torch_tensor = a_boolean_vector.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(an_int_tensor.invert_(), int_result))  # test result
        self.assertTrue(ht.equal(an_int_tensor, int_result))  # test result in-place
        self.assertIs(an_int_tensor, an_int_tensor_double)  # test DNDarray in-place
        self.assertIs(
            an_int_tensor.larray, underlying_int_torch_tensor
        )  # test torch tensor in-place
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int8_tensor.invert_(), int8_result))
        self.assertTrue(ht.equal(an_int8_tensor, int8_result))
        self.assertIs(an_int8_tensor, an_int8_tensor_double)
        self.assertIs(an_int8_tensor.larray, underlying_int8_torch_tensor)

        self.assertTrue(ht.equal(an_uint8_tensor.invert_(), uint8_result))
        self.assertTrue(ht.equal(an_uint8_tensor, uint8_result))
        self.assertIs(an_uint8_tensor, an_uint8_tensor_double)
        self.assertIs(an_uint8_tensor.larray, underlying_uint8_torch_tensor)

        self.assertTrue(ht.equal(a_boolean_vector.invert_(), bool_result))
        self.assertTrue(ht.equal(a_boolean_vector, bool_result))
        self.assertIs(a_boolean_vector, a_boolean_vector_double)
        self.assertIs(a_boolean_vector.larray, underlying_boolean_torch_tensor)
        underlying_boolean_torch_tensor.copy_(self.a_boolean_vector.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(an_int_tensor.bitwise_not_(), int_result))
        self.assertTrue(ht.equal(an_int_tensor, int_result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            a_tensor.invert_()
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.invert_(an_int_scalar)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            erroneous_type.invert_()
        erroneous_type = self.erroneous_type  # reset
        with self.assertRaises(AttributeError):
            a_scalar.invert_()
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.invert_()
        a_string = self.a_string  # reset

    def test_lcm(self):
        a = ht.array([5, 10, 15])
        b = ht.array([3, 4, 5])
        c = ht.array([3.0, 4.0, 5.0])
        result = ht.array([15, 20, 15])

        self.assertTrue(ht.equal(ht.lcm(a, b), result))
        self.assertTrue(ht.equal(ht.lcm(a, a), a))
        self.assertEqual(ht.lcm(a, b).dtype, ht.int64)

        with self.assertRaises(TypeError):
            ht.lcm(a, c)
        with self.assertRaises(ValueError):
            ht.lcm(a, ht.array([15, 20]))

    def test_lcm_(self):
        # Copies of class variables for the in-place operations
        a_tensor = self.a_tensor.copy()
        an_int_scalar = self.an_int_scalar
        an_int_vector = self.an_int_vector.copy()
        another_int_vector = self.another_int_vector.copy()
        an_int_tensor = an_int_tensor_double = self.an_int_tensor.copy()
        another_int_tensor = self.another_int_tensor.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[2, 2], [6, 4]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_int_torch_tensor = an_int_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(an_int_tensor.lcm_(an_int_scalar), result))  # test result
        self.assertTrue(ht.equal(an_int_tensor, result))  # test result in-place
        self.assertIs(an_int_tensor, an_int_tensor_double)  # test DNDarray in-place
        self.assertIs(
            an_int_tensor.larray, underlying_int_torch_tensor
        )  # test torch tensor in-place
        self.assertTrue(
            ht.equal(an_int_scalar, self.an_int_scalar)
        )  # test if other input is unchanged
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int_tensor.lcm_(another_int_tensor), result))
        self.assertTrue(ht.equal(an_int_tensor, result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(another_int_tensor, self.another_int_tensor))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        self.assertTrue(ht.equal(an_int_tensor.lcm_(an_int_vector), result))
        self.assertTrue(ht.equal(an_int_tensor, result))
        self.assertIs(an_int_tensor, an_int_tensor_double)
        self.assertIs(an_int_tensor.larray, underlying_int_torch_tensor)
        self.assertTrue(ht.equal(an_int_vector, self.an_int_vector))
        underlying_int_torch_tensor.copy_(self.an_int_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([2], split=0)
        a.lcm_(b)
        self.assertTrue(ht.equal(a, ht.array([2, 2])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0, dtype=int) * 2
        b = ht.ones(10, split=0, dtype=int) * 3
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a.lcm_(b[1:])
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0, dtype=int) * 2  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a.lcm_(b[1:-1])
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), dtype=int, split=0) * 2
        b = ht.ones((1, 10), dtype=int, split=0) * 3
        a_lshape = a.lshape
        a.lcm_(b)
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, dtype=int, split=0) * 2
        b = ht.ones(10, dtype=int, split=None) * 3
        a_lshape = a.lshape
        a.lcm_(b)
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12, dtype=int).reshape((4, 3), new_split=1) * 2
        b = ht.ones(3, dtype=int, split=0) * 3
        a_lshape = a.lshape
        a.lcm_(b)
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            an_int_tensor.lcm_(a_tensor)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.lcm_(an_int_tensor)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(ValueError):
            an_int_tensor.lcm_(another_int_vector)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            an_int_tensor.lcm_(erroneous_type)
        an_int_tensor = self.an_int_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            an_int_scalar.lcm_(an_int_tensor)
        an_int_scalar = self.an_int_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.lcm_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), dtype=int, split=0) * 2
        b = ht.ones((2, 10), dtype=int, split=0) * 3
        with self.assertRaises(ValueError):
            a.lcm_(b)

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, dtype=int, split=None) * 2
        b = ht.ones(10, dtype=int, split=0) * 3
        with self.assertRaises(ValueError):
            a.lcm_(b)

        a = ht.ones(9, dtype=int).reshape((3, 3), new_split=0) * 2
        b = ht.ones(9, dtype=int).reshape((3, 3), new_split=1) * 3
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a.lcm_(b)
            self.assertTrue((a == 6).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a.lcm_(b)

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32) * 2
        b = ht.ones(3, dtype=ht.float64) * 3
        with self.assertRaises(TypeError):
            a.lcm_(b)

    def test_left_shift(self):
        int_tensor = ht.array([[0, 1], [2, 3]])

        self.assertTrue(ht.equal(ht.left_shift(int_tensor, 1), int_tensor * 2))
        self.assertTrue(ht.equal(ht.left_shift(int_tensor.copy().resplit_(0), 1), int_tensor * 2))
        self.assertTrue(ht.equal(int_tensor << 2, int_tensor * 4))
        self.assertTrue(
            ht.equal(1 << ht.ones(3, dtype=ht.int32), ht.array([2, 2, 2], dtype=ht.int32))
        )

        ht.left_shift(int_tensor, 1, out=int_tensor, where=int_tensor > 1)
        self.assertTrue(ht.equal(int_tensor, ht.array([[0, 1], [4, 6]])))

        with self.assertRaises(TypeError):
            ht.left_shift(int_tensor, 2.4)
        res = ht.left_shift(ht.array([True]), 2)
        self.assertTrue(res == 4)
        with self.assertRaises(TypeError):
            int_tensor << "s"

    def test_left_shift_(self):
        """
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string
        """

        int_result = ht.array([[4, 8], [12, 16]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = self.an_int_tensor.larray

        # Check for some possible combinations of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(
            ht.equal(ht.left_shift_(self.an_int_tensor, self.an_int_scalar), int_result)
        )  # test result
        self.assertTrue(ht.equal(self.an_int_tensor, int_result))  # test in-place
        self.assertIs(self.an_int_tensor.larray, underlying_torch_tensor)  # test in-place
        self.assertTrue(ht.equal(self.an_int_scalar, ht.int(2)))  # test if other input is unchanged
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        underlying_torch_tensor = self.an_int_tensor.larray  # reset

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            ht.left_shift_(self.an_int_tensor, self.a_scalar)
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        with self.assertRaises(TypeError):
            ht.left_shift_(self.a_scalar, self.an_int_tensor)
        self.a_scalar = self.a_scalar  # reset
        with self.assertRaises(TypeError):
            ht.left_shift_(self.a_boolean_vector, self.a_scalar)
        self.a_boolean_vector = ht.array([False, True, False, True])
        with self.assertRaises(ValueError):
            ht.left_shift_(self.an_int_tensor, self.another_int_vector)
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        with self.assertRaises(TypeError):
            ht.left_shift_(self.an_int_tensor, self.erroneous_type)
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        with self.assertRaises(TypeError):
            ht.left_shift_("T", "s")

    def test_mul(self):
        result = ht.array([[2.0, 4.0], [6.0, 8.0]])

        self.assertTrue(ht.equal(ht.mul(self.a_scalar, self.a_scalar), ht.array(4.0)))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.mul(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.mul(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.mul(self.a_split_tensor, self.a_tensor), result))

        self.assertTrue(ht.equal(self.a_tensor * self.a_scalar, result))
        self.assertTrue(ht.equal(self.a_scalar * self.a_tensor, result))

        with self.assertRaises(ValueError):
            ht.mul(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.mul(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.mul("T", "s")
        with self.assertRaises(TypeError):
            self.a_tensor * "T"

    def test_mul_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[2.0, 4.0], [6.0, 8.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.mul_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.mul_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.mul_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.mul_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.multiply_(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.__imul__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor *= a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([2], split=0)
        a *= b
        self.assertTrue(ht.equal(a, ht.array([2, 4])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 2
        b = ht.ones(10, split=0) * 3
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a *= b[1:]
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 2  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a *= b[1:-1]
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 2
        b = ht.ones((1, 10), split=0) * 3
        a_lshape = a.lshape
        a *= b
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 2
        b = ht.ones(10, split=None) * 3
        a_lshape = a.lshape
        a *= b
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 2
        b = ht.ones(3, split=0) * 3
        a_lshape = a.lshape
        a *= b
        self.assertTrue((a == 6).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.mul_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.mul_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.mul_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.mul_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a *= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a *= b

        a = ht.ones(9).reshape((3, 3), new_split=0) * 2
        b = ht.ones(9).reshape((3, 3), new_split=1) * 3
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a *= b
            self.assertTrue((a == 6).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a *= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a *= b

    def test_nan_to_num(self):
        arr = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf])
        a = ht.nan_to_num(arr)
        self.assertTrue(torch.equal(a.larray, torch.nan_to_num(arr.larray)))

        arr = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf], split=0)
        a = ht.nan_to_num(arr, nan=0, posinf=1, neginf=-1)
        self.assertTrue(
            torch.equal(a.larray, torch.nan_to_num(arr.larray, nan=0, posinf=1, neginf=-1))
        )

    def test_nan_to_num_(self):
        """
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string
        """

        # test one: unsplit
        arr = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf])
        input_torch_tensor = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf]).larray
        output_torch_tensor = torch.nan_to_num(input_torch_tensor)

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = arr.larray

        ht.nan_to_num_(arr)

        self.assertTrue(torch.equal(arr.larray, output_torch_tensor))  # test result
        self.assertIs(arr.larray, underlying_torch_tensor)  # test in-place

        # test two: split
        arr = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf], split=0)
        input_torch_tensor = ht.array([1, 2, 3, ht.nan, ht.inf, -ht.inf], split=0).larray
        output_torch_tensor = torch.nan_to_num(input_torch_tensor, nan=0, posinf=1, neginf=-1)

        underlying_torch_tensor = arr.larray

        ht.nan_to_num_(arr, nan=0, posinf=1, neginf=-1)

        self.assertTrue(torch.equal(arr.larray, output_torch_tensor))
        self.assertIs(arr.larray, underlying_torch_tensor)

    def test_nanprod(self):
        array_len = 11

        # check prod over all float elements of 1d tensor locally
        shape_noaxis = ht.zeros(array_len)
        shape_noaxis[0] = ht.nan
        no_axis_nanprod = ht.nanprod(shape_noaxis)

        self.assertIsInstance(no_axis_nanprod, ht.DNDarray)
        self.assertEqual(no_axis_nanprod.shape, tuple())
        self.assertEqual(no_axis_nanprod.lshape, tuple())
        self.assertEqual(no_axis_nanprod.dtype, ht.float32)
        self.assertEqual(no_axis_nanprod.larray.dtype, torch.float32)
        self.assertEqual(no_axis_nanprod.split, None)
        self.assertEqual(no_axis_nanprod.larray, 0)

        out_noaxis = ht.array(1, dtype=shape_noaxis.dtype)
        ht.nanprod(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis.larray == shape_noaxis.larray.nan_to_num().prod())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0).astype(ht.float32)
        shape_noaxis_split[0] = ht.nan
        shape_noaxis_split_nanprod = ht.nanprod(shape_noaxis_split)

        self.assertIsInstance(shape_noaxis_split_nanprod, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_nanprod.shape, tuple())
        self.assertEqual(shape_noaxis_split_nanprod.lshape, tuple())
        self.assertEqual(shape_noaxis_split_nanprod.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_nanprod.larray.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_nanprod.split, None)
        self.assertEqual(shape_noaxis_split_nanprod, np.math.factorial(10))

        out_noaxis = ht.array(1, dtype=shape_noaxis_split.dtype)
        ht.nanprod(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, np.math.factorial(10))

    def test_nansum(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        shape_noaxis[0] = ht.nan
        no_axis_nansum = ht.nansum(shape_noaxis)

        self.assertIsInstance(no_axis_nansum, ht.DNDarray)
        self.assertEqual(no_axis_nansum.shape, tuple())
        self.assertEqual(no_axis_nansum.lshape, tuple())
        self.assertEqual(no_axis_nansum.dtype, ht.float32)
        self.assertEqual(no_axis_nansum.larray.dtype, torch.float32)
        self.assertEqual(no_axis_nansum.split, None)
        self.assertEqual(no_axis_nansum.larray, array_len - 1)

        out_noaxis = ht.array(0, dtype=shape_noaxis.dtype)
        ht.nansum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis.larray == shape_noaxis.larray.nansum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0).astype(ht.float32)
        shape_noaxis_split[0] = ht.nan
        shape_noaxis_split_nansum = ht.nansum(shape_noaxis_split)

        self.assertIsInstance(shape_noaxis_split_nansum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_nansum.shape, tuple())
        self.assertEqual(shape_noaxis_split_nansum.lshape, tuple())
        self.assertEqual(shape_noaxis_split_nansum.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_nansum.larray.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_nansum.split, None)
        self.assertEqual(shape_noaxis_split_nansum, 55)

        out_noaxis = ht.array(0, dtype=shape_noaxis_split.dtype)
        ht.nansum(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 55)

    def test_neg(self):
        self.assertTrue(ht.equal(ht.neg(ht.array([-1, 1])), ht.array([1, -1])))
        self.assertTrue(ht.equal(-ht.array([-1.0, 1.0]), ht.array([1.0, -1.0])))

        a = ht.array([1 + 1j, 2 - 2j, 3, 4j, 5], split=0)
        b = out = ht.empty(5, dtype=ht.complex64, split=0)
        ht.negative(a, out=out)
        self.assertTrue(ht.equal(out, ht.array([-1 - 1j, -2 + 2j, -3, -4j, -5], split=0)))
        self.assertIs(out, b)

        with self.assertRaises(TypeError):
            ht.neg(1)

    def test_neg_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        an_int_vector = an_int_vector_double = self.an_int_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[-1.0, -2.0], [-3.0, -4.0]])
        int_result = ht.array([-2, -2])
        a_complex_vector = a_complex_vector_double = ht.array([1 + 1j, 2 - 2j, 3, 4j, 5], split=0)
        complex_result = ht.array([-1 - 1j, -2 + 2j, -3, -4j, -5], split=0)

        # We identify the underlying PyTorch objects to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray
        underlying_int_torch_tensor = an_int_vector.larray
        underlying_complex_torch_tensor = a_complex_vector.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.neg_(), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(an_int_vector.neg_(), int_result))
        self.assertTrue(ht.equal(an_int_vector, int_result))
        self.assertIs(an_int_vector, an_int_vector_double)
        self.assertIs(an_int_vector.larray, underlying_int_torch_tensor)
        underlying_int_torch_tensor.copy_(self.an_int_vector.larray)

        self.assertTrue(ht.equal(a_complex_vector.neg_(), complex_result))
        self.assertTrue(ht.equal(a_complex_vector, complex_result))
        self.assertIs(a_complex_vector, a_complex_vector_double)
        self.assertIs(a_complex_vector.larray, underlying_complex_torch_tensor)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.negative_(), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            a_tensor.neg_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            erroneous_type.neg_()
        erroneous_type = self.erroneous_type  # reset
        with self.assertRaises(AttributeError):
            a_scalar.neg_()
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.neg_()
        a_string = self.a_string  # reset

    def test_pos(self):
        self.assertTrue(ht.equal(ht.pos(ht.array([-1, 1])), ht.array([-1, 1])))
        self.assertTrue(ht.equal(+ht.array([-1.0, 1.0]), ht.array([-1.0, 1.0])))

        a = ht.array([1 + 1j, 2 - 2j, 3, 4j, 5], split=0)
        b = out = ht.empty(5, dtype=ht.complex64, split=0)
        ht.positive(a, out=out)
        self.assertTrue(ht.equal(out, a))
        self.assertIs(out, b)

        with self.assertRaises(TypeError):
            ht.pos(1)

    def test_pow(self):
        result = ht.array([[1.0, 4.0], [9.0, 16.0]])
        commutated_result = ht.array([[2.0, 4.0], [8.0, 16.0]])
        self.assertTrue(ht.equal(ht.pow(self.a_scalar, self.a_scalar), ht.array(4.0)))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.pow(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.pow(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.pow(self.a_split_tensor, self.a_tensor), commutated_result))

        self.assertTrue(ht.equal(self.a_tensor**self.a_scalar, result))
        self.assertTrue(ht.equal(self.a_scalar**self.a_tensor, commutated_result))

        # test scalar base and exponent
        self.assertTrue(ht.equal(ht.pow(2, 3), ht.array(8)))
        self.assertTrue(ht.equal(ht.pow(2, 3.5), ht.array(11.313708498984761)))

        # test exceptions
        with self.assertRaises(ValueError):
            ht.pow(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.pow(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.pow("T", "s")
        with self.assertRaises(TypeError):
            pow(self.a_tensor, 2, 3)
        with self.assertRaises(TypeError):
            self.a_tensor ** "T"

    def test_pow_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[1.0, 4.0], [9.0, 16.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.pow_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.pow_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.pow_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.pow_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.power_(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.__ipow__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor **= a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([2], split=0)
        a **= b
        self.assertTrue(ht.equal(a, ht.array([1, 4])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 2
        b = ht.zeros(10, split=0)
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a **= b[1:]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 2  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a **= b[1:-1]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 2
        b = ht.zeros((1, 10), split=0)
        a_lshape = a.lshape
        a **= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 2
        b = ht.zeros(10, split=None)
        a_lshape = a.lshape
        a **= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 2
        b = ht.zeros(3, split=0)
        a_lshape = a.lshape
        a **= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.pow_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.pow_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.pow_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.pow_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a **= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a **= b

        a = ht.ones(9).reshape((3, 3), new_split=0) * 2
        b = ht.zeros(9).reshape((3, 3), new_split=1)
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a **= b
            self.assertTrue((a == 1).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a **= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a **= b

    def test_prod(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        no_axis_prod = shape_noaxis.prod()

        self.assertIsInstance(no_axis_prod, ht.DNDarray)
        self.assertEqual(no_axis_prod.shape, ())
        self.assertEqual(no_axis_prod.lshape, ())
        self.assertEqual(no_axis_prod.dtype, ht.float32)
        self.assertEqual(no_axis_prod.larray.dtype, torch.float32)
        self.assertEqual(no_axis_prod.split, None)
        self.assertEqual(no_axis_prod.larray, 1)

        out_noaxis = ht.zeros(())
        ht.prod(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 1)

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(1, array_len, split=0)
        shape_noaxis_split_prod = shape_noaxis_split.prod()

        self.assertIsInstance(shape_noaxis_split_prod, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_prod.shape, ())
        self.assertEqual(shape_noaxis_split_prod.lshape, ())
        self.assertEqual(shape_noaxis_split_prod.dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_prod.larray.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_prod.split, None)
        self.assertEqual(shape_noaxis_split_prod, 3628800)

        out_noaxis = ht.zeros(())
        ht.prod(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 3628800)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.full((3, 3, 3), 2)
        no_axis_prod = shape_noaxis.prod()

        self.assertIsInstance(no_axis_prod, ht.DNDarray)
        self.assertEqual(no_axis_prod.shape, ())
        self.assertEqual(no_axis_prod.lshape, ())
        self.assertEqual(no_axis_prod.dtype, ht.float32)
        self.assertEqual(no_axis_prod.larray.dtype, torch.float32)
        self.assertEqual(no_axis_prod.split, None)
        self.assertEqual(no_axis_prod.larray, 134217728)

        out_noaxis = ht.zeros(())
        ht.prod(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 134217728)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.full((3, 3, 3), 2, split=0)
        split_axis_prod = shape_noaxis_split_axis.prod(axis=0)

        self.assertIsInstance(split_axis_prod, ht.DNDarray)
        self.assertEqual(split_axis_prod.shape, (3, 3))
        self.assertEqual(split_axis_prod.dtype, ht.float32)
        self.assertEqual(split_axis_prod.larray.dtype, torch.float32)
        self.assertEqual(split_axis_prod.split, None)

        out_axis = ht.ones((3, 3))
        ht.prod(shape_noaxis, axis=0, out=out_axis)
        self.assertTrue(
            (
                out_axis.larray
                == torch.full((3,), 8, dtype=torch.float, device=self.device.torch_device)
            ).all()
        )

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.full((1, 2, 3, 4, 5), 2, split=1)
        shape_noaxis_split_axis_neg_prod = shape_noaxis_split_axis_neg.prod(axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_prod, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_axis_neg_prod.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_prod.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_prod.larray.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_prod.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5), split=1)
        ht.prod(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # check sum over all float elements of splitted 3d tensor with tuple axis
        shape_split_axis_tuple = ht.ones((3, 4, 5), split=1)
        shape_split_axis_tuple_prod = shape_split_axis_tuple.prod(axis=(-2, -3))
        expected_result = ht.ones((5,))

        self.assertIsInstance(shape_split_axis_tuple_prod, ht.DNDarray)
        self.assertEqual(shape_split_axis_tuple_prod.shape, (5,))
        self.assertEqual(shape_split_axis_tuple_prod.dtype, ht.float32)
        self.assertEqual(shape_split_axis_tuple_prod.larray.dtype, torch.float32)
        self.assertEqual(shape_split_axis_tuple_prod.split, None)
        self.assertTrue((shape_split_axis_tuple_prod == expected_result).all())

        # empty array
        empty = ht.array([])
        self.assertEqual(ht.prod(empty), ht.array([1.0]))

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).prod(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).prod(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).prod(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).prod(axis="bad_axis_type")

    def test_remainder(self):
        a_tensor = ht.array([[1, 4], [2, 2]])
        another_tensor = ht.array([[1, 2], [3, 4]])
        a_result = ht.array([[0, 0], [2, 2]])
        another_result = ht.array([[1, 0], [0, 0]])

        self.assertTrue(ht.equal(ht.mod(a_tensor, another_tensor), a_result))
        self.assertTrue(ht.equal(a_tensor % self.an_int_scalar, another_result))
        self.assertTrue(ht.equal(self.an_int_scalar % another_tensor, a_result))

        with self.assertRaises(TypeError):
            ht.mod(a_tensor, "T")
        with self.assertRaises(TypeError):
            ht.mod("T", another_tensor)
        with self.assertRaises(TypeError):
            a_tensor % "s"

    def test_remainder_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[1.0, 0.0], [1.0, 0.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.mod_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.mod_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.mod_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.mod_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.remainder_(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.__imod__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor %= a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Difference compared to 'fmod_'
        vector_3 = ht.array([0.4, -3.0])
        result_2 = ht.array([[0.2, -1.0], [0.2, -2.0]])  # fmod_ result == [[0.2, 2.0], [0.2, 1.0]]
        result_3 = ht.array([0.4, 1.0])  # fmod_ result == [0.4, -1.0]

        self.assertTrue(ht.allclose(a_tensor.mod_(vector_3), result_2))
        self.assertTrue(ht.allclose(a_tensor, result_2))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(vector_3, ht.array([0.4, -3.0])))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.allclose(vector_3.mod_(a_vector), result_3))
        self.assertTrue(ht.allclose(vector_3, result_3))

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([2], split=0)
        a %= b
        self.assertTrue(ht.equal(a, ht.array([1, 0])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=0) * 2
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a %= b[1:]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 3  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a %= b[1:-1]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 3
        b = ht.ones((1, 10), split=0) * 2
        a_lshape = a.lshape
        a %= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 3
        b = ht.ones(10, split=None) * 2
        a_lshape = a.lshape
        a %= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 3
        b = ht.ones(3, split=0) * 2
        a_lshape = a.lshape
        a %= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.mod_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.mod_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.mod_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.mod_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a %= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a %= b

        a = ht.ones(9).reshape((3, 3), new_split=0) * 3
        b = ht.ones(9).reshape((3, 3), new_split=1) * 2
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a %= b
            self.assertTrue((a == 1).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a %= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a %= b

    def test_right_shift(self):
        int_tensor = ht.array([[0, 1], [2, 3]])

        self.assertTrue(ht.equal(ht.right_shift(int_tensor, 1), int_tensor // 2))
        self.assertTrue(ht.equal(ht.right_shift(int_tensor.copy().resplit_(0), 1), int_tensor // 2))
        self.assertTrue(ht.equal(int_tensor >> 2, int_tensor // 4))
        self.assertTrue(
            ht.equal(1 >> ht.ones(3, dtype=ht.int32), ht.array([0, 0, 0], dtype=ht.int32))
        )

        with self.assertRaises(TypeError):
            ht.right_shift(int_tensor, 2.4)

        res = ht.right_shift(ht.array([True]), 2)
        self.assertTrue(res == 0)
        with self.assertRaises(TypeError):
            int_tensor >> "s"

    def test_right_shift_(self):
        """
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string
        """

        int_result = ht.array([[0, 0], [0, 1]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = self.an_int_tensor.larray

        # Check for some possible combinations of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(
            ht.equal(ht.right_shift_(self.an_int_tensor, self.an_int_scalar), int_result)
        )  # test result
        self.assertTrue(ht.equal(self.an_int_tensor, int_result))  # test in-place
        self.assertIs(self.an_int_tensor.larray, underlying_torch_tensor)  # test in-place
        self.assertTrue(ht.equal(self.an_int_scalar, 2))  # test if other input is unchanged
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        underlying_torch_tensor = self.an_int_tensor.larray  # reset

        # test function with wrong inputs
        with self.assertRaises(TypeError):
            ht.right_shift_(self.an_int_tensor, self.a_scalar)
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        with self.assertRaises(TypeError):
            ht.right_shift_(self.a_scalar, self.an_int_tensor)
        self.a_scalar = self.a_scalar  # reset
        with self.assertRaises(TypeError):
            ht.right_shift_(self.a_boolean_vector, self.a_scalar)
        self.a_boolean_vector = ht.array([False, True, False, True])
        with self.assertRaises(ValueError):
            ht.right_shift_(self.an_int_tensor, self.another_int_vector)
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        with self.assertRaises(TypeError):
            ht.right_shift_(self.an_int_tensor, self.erroneous_type)
        self.an_int_tensor.larray = ht.array([[1, 2], [3, 4]]).larray  # reset
        with self.assertRaises(TypeError):
            ht.right_shift_("T", "s")

    def test_sub(self):
        result = ht.array([[-1.0, 0.0], [1.0, 2.0]])
        minus_result = ht.array([[1.0, 0.0], [-1.0, -2.0]])

        self.assertTrue(ht.equal(ht.sub(self.a_scalar, self.a_scalar), ht.array(0.0)))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.sub(self.a_scalar, self.a_tensor), minus_result))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.sub(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.sub(self.a_split_tensor, self.a_tensor), minus_result))

        self.assertTrue(ht.equal(self.a_tensor - self.a_scalar, result))
        self.assertTrue(ht.equal(self.a_scalar - self.a_tensor, minus_result))

        with self.assertRaises(ValueError):
            ht.sub(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.sub(self.a_tensor, self.erroneous_type)
        with self.assertRaises(TypeError):
            ht.sub("T", "s")
        with self.assertRaises(TypeError):
            self.a_tensor - "T"

    def test_sub_(self):
        # Copies of class variables for the in-place operations
        a_scalar = self.a_scalar
        a_vector = self.a_vector.copy()
        another_vector = self.another_vector.copy()
        a_tensor = a_tensor_double = self.a_tensor.copy()
        another_tensor = self.another_tensor.copy()
        an_int_scalar = self.an_int_scalar
        erroneous_type = self.erroneous_type
        a_string = self.a_string

        result = ht.array([[-1.0, 0.0], [1.0, 2.0]])

        # We identify the underlying PyTorch object to check whether operations are really in-place
        underlying_torch_tensor = a_tensor.larray

        # Check for every possible combination of inputs whether the right solution is computed and
        # saved in the right place and whether the second input stays unchanged. After every tested
        # computation, we reset changed variables.
        self.assertTrue(ht.equal(a_tensor.sub_(a_scalar), result))  # test result
        self.assertTrue(ht.equal(a_tensor, result))  # test result in-place
        self.assertIs(a_tensor, a_tensor_double)  # test DNDarray in-place
        self.assertIs(a_tensor.larray, underlying_torch_tensor)  # test torch tensor in-place
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))  # test if other input is unchanged
        underlying_torch_tensor.copy_(self.a_tensor.larray)  # reset

        self.assertTrue(ht.equal(a_tensor.sub_(another_tensor), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(another_tensor, self.another_tensor))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.sub_(a_vector), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_vector, self.a_vector))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.sub_(an_int_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(an_int_scalar, 2))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Test other possible ways to call this function
        self.assertTrue(ht.equal(a_tensor.subtract_(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        self.assertTrue(ht.equal(a_tensor.__isub__(a_scalar), result))
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        a_tensor -= a_scalar
        self.assertTrue(ht.equal(a_tensor, result))
        self.assertIs(a_tensor, a_tensor_double)
        self.assertIs(a_tensor.larray, underlying_torch_tensor)
        self.assertTrue(ht.equal(a_scalar, self.a_scalar))
        underlying_torch_tensor.copy_(self.a_tensor.larray)

        # Single element split
        a = a_double = ht.array([1, 2], split=0)
        b = ht.array([1], split=0)
        a -= b
        self.assertTrue(ht.equal(a, ht.array([0, 1])))
        if a.comm.size > 1:
            if a.comm.rank < 2:
                self.assertEqual(a.larray.size()[0], 1)
            else:
                self.assertEqual(a.larray.size()[0], 0)
        self.assertIs(a, a_double)

        # test with differently distributed DNDarrays
        a = ht.ones(10, split=0) * 2
        b = ht.ones(10, split=0)
        a = a_double = a[:-1]
        a_lshape = a.lshape
        a -= b[1:]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test unbalanced
        a = ht.ones(10, split=0) * 2  # reset
        a = a_double = a[1:-1]
        a_lshape = a.lshape
        a -= b[1:-1]
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # broadcast in split dimension
        a = a_double = ht.ones((2, 10), split=0) * 2
        b = ht.ones((1, 10), split=0)
        a_lshape = a.lshape
        a -= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # first input is split, second is unsplit
        a = a_double = ht.ones(10, split=0) * 2
        b = ht.ones(10, split=None)
        a_lshape = a.lshape
        a -= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # differently split inputs but broadcastable underlying torch.Tensors, s.t. we can process
        # the inputs in-place without resplitting
        a = a_double = ht.ones(12).reshape((4, 3), new_split=1) * 2
        b = ht.ones(3, split=0)
        a_lshape = a.lshape
        a -= b
        self.assertTrue((a == 1).all())
        self.assertTrue(a.lshape == a_lshape)
        self.assertIs(a, a_double)

        # test function with wrong inputs
        with self.assertRaises(ValueError):
            a_tensor.sub_(another_vector)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(TypeError):
            a_tensor.sub_(erroneous_type)
        a_tensor = self.a_tensor.copy()  # reset
        with self.assertRaises(AttributeError):
            a_scalar.sub_(a_tensor)
        a_scalar = self.a_scalar  # reset
        with self.assertRaises(AttributeError):
            a_string.sub_("s")
        a_string = self.a_string  # reset

        # test function for broadcasting in wrong direction
        a = ht.ones((1, 10), split=0)
        b = ht.zeros((2, 10), split=0)
        with self.assertRaises(ValueError):
            a -= b

        # test function with differently split inputs that are not processible in-place without
        # resplitting
        a = ht.ones(10, split=None)
        b = ht.ones(10, split=0)
        with self.assertRaises(ValueError):
            a -= b

        a = ht.ones(9).reshape((3, 3), new_split=0) * 2
        b = ht.ones(9).reshape((3, 3), new_split=1)
        if a.comm.Get_size() == 1:
            a_double = a
            a_lshape = a.lshape
            a -= b
            self.assertTrue((a == 1).all())
            self.assertTrue(a.lshape == a_lshape)
            self.assertIs(a, a_double)
        else:
            with self.assertRaises(ValueError):
                a -= b

        # test function for invalid casting between data types
        a = ht.ones(3, dtype=ht.float32)
        b = ht.ones(3, dtype=ht.float64)
        with self.assertRaises(TypeError):
            a -= b

    def test_sum(self):
        array_len = 11

        # check sum over all float elements of 1d tensor locally
        shape_noaxis = ht.ones(array_len)
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.DNDarray)
        self.assertEqual(no_axis_sum.shape, ())
        self.assertEqual(no_axis_sum.lshape, ())
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum.larray.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum.larray, array_len)

        out_noaxis = ht.zeros(())
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertTrue(out_noaxis.larray == shape_noaxis.larray.sum())

        # check sum over all float elements of split 1d tensor
        shape_noaxis_split = ht.arange(array_len, split=0)
        shape_noaxis_split_sum = shape_noaxis_split.sum()

        self.assertIsInstance(shape_noaxis_split_sum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_sum.shape, ())
        self.assertEqual(shape_noaxis_split_sum.lshape, ())
        self.assertEqual(shape_noaxis_split_sum.dtype, ht.int64)
        self.assertEqual(shape_noaxis_split_sum.larray.dtype, torch.int64)
        self.assertEqual(shape_noaxis_split_sum.split, None)
        self.assertEqual(shape_noaxis_split_sum, 55)

        out_noaxis = ht.zeros(())
        ht.sum(shape_noaxis_split, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 55)

        # check sum over all float elements of 3d tensor locally
        shape_noaxis = ht.ones((3, 3, 3))
        no_axis_sum = shape_noaxis.sum()

        self.assertIsInstance(no_axis_sum, ht.DNDarray)
        self.assertEqual(no_axis_sum.shape, ())
        self.assertEqual(no_axis_sum.lshape, ())
        self.assertEqual(no_axis_sum.dtype, ht.float32)
        self.assertEqual(no_axis_sum.larray.dtype, torch.float32)
        self.assertEqual(no_axis_sum.split, None)
        self.assertEqual(no_axis_sum.larray, 27)

        out_noaxis = ht.zeros(())
        ht.sum(shape_noaxis, out=out_noaxis)
        self.assertEqual(out_noaxis.larray, 27)

        # check sum over all float elements of split 3d tensor
        shape_noaxis_split_axis = ht.ones((3, 3, 3), split=0)
        split_axis_sum = shape_noaxis_split_axis.sum(axis=0)

        self.assertIsInstance(split_axis_sum, ht.DNDarray)
        self.assertEqual(split_axis_sum.shape, (3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum.larray.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, None)

        # check split semantics
        shape_noaxis_split_axis = ht.ones((3, 3, 3), split=2)
        split_axis_sum = shape_noaxis_split_axis.sum(axis=1)
        self.assertIsInstance(split_axis_sum, ht.DNDarray)
        self.assertEqual(split_axis_sum.shape, (3, 3))
        self.assertEqual(split_axis_sum.dtype, ht.float32)
        self.assertEqual(split_axis_sum.larray.dtype, torch.float32)
        self.assertEqual(split_axis_sum.split, 1)

        out_noaxis = ht.zeros((3, 3))
        ht.sum(shape_noaxis, axis=0, out=out_noaxis)
        self.assertTrue(
            (
                out_noaxis.larray
                == torch.full((3, 3), 3, dtype=torch.float, device=self.device.torch_device)
            ).all()
        )

        # check sum over all float elements of splitted 5d tensor with negative axis
        shape_noaxis_split_axis_neg = ht.ones((1, 2, 3, 4, 5), split=1)
        shape_noaxis_split_axis_neg_sum = shape_noaxis_split_axis_neg.sum(axis=-2)

        self.assertIsInstance(shape_noaxis_split_axis_neg_sum, ht.DNDarray)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.shape, (1, 2, 3, 5))
        self.assertEqual(shape_noaxis_split_axis_neg_sum.dtype, ht.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.larray.dtype, torch.float32)
        self.assertEqual(shape_noaxis_split_axis_neg_sum.split, 1)

        out_noaxis = ht.zeros((1, 2, 3, 5), split=1)
        ht.sum(shape_noaxis_split_axis_neg, axis=-2, out=out_noaxis)

        # check sum over all float elements of splitted 3d tensor with tuple axis
        shape_split_axis_tuple = ht.ones((3, 4, 5), split=1)
        shape_split_axis_tuple_sum = shape_split_axis_tuple.sum(axis=(-2, -3))
        expected_result = ht.ones((5,)) * 12.0

        self.assertIsInstance(shape_split_axis_tuple_sum, ht.DNDarray)
        self.assertEqual(shape_split_axis_tuple_sum.shape, (5,))
        self.assertEqual(shape_split_axis_tuple_sum.dtype, ht.float32)
        self.assertEqual(shape_split_axis_tuple_sum.larray.dtype, torch.float32)
        self.assertEqual(shape_split_axis_tuple_sum.split, None)
        self.assertTrue((shape_split_axis_tuple_sum == expected_result).all())

        # empty array
        empty = ht.array([])
        self.assertEqual(ht.sum(empty), ht.array([0.0]))

        # exceptions
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=1)
        with self.assertRaises(ValueError):
            ht.ones(array_len).sum(axis=-2)
        with self.assertRaises(ValueError):
            ht.ones((4, 4)).sum(axis=0, out=out_noaxis)
        with self.assertRaises(TypeError):
            ht.ones(array_len).sum(axis="bad_axis_type")

    def test_right_hand_side_operations(self):
        """
        This test ensures that for each arithmetic operation (e.g. +, -, *, ...) that is implemented
        in the tensor class, it works both ways.

        Examples
        --------
        >>> import heat as ht
        >>> T = ht.float32([[1., 2.], [3., 4.]])
        >>> assert T * 3 == 3 * T
        """
        operators = (
            ("__add__", operator.add, True),
            ("__sub__", operator.sub, False),
            ("__mul__", operator.mul, True),
            ("__truediv__", operator.truediv, False),
            ("__floordiv__", operator.floordiv, False),
            ("__mod__", operator.mod, False),
            ("__pow__", operator.pow, False),
        )
        tensor = ht.float32([[1, 4], [2, 3]])
        num = 3
        for attr, op, commutative in operators:
            try:
                func = tensor.__getattribute__(attr)
            except AttributeError:
                continue
            self.assertTrue(callable(func))
            res_1 = op(tensor, num)
            res_2 = op(num, tensor)
            if commutative:
                self.assertTrue(ht.equal(res_1, res_2))
        # TODO: Test with split tensors when binary operations are working properly for split tensors
