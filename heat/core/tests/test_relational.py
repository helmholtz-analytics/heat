import unittest
import os
import heat as ht

ht.use_device(os.environ.get("DEVICE"))


class TestRelational(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a_scalar = 2.0
        cls.an_int_scalar = 2

        cls.a_vector = ht.float32([2, 2])
        cls.another_vector = ht.float32([2, 2, 2])

        cls.a_tensor = ht.array([[1.0, 2.0], [3.0, 4.0]])
        cls.another_tensor = ht.array([[2.0, 2.0], [2.0, 2.0]])
        cls.a_split_tensor = cls.another_tensor.copy().resplit_(0)
        cls.split_ones_tensor = ht.ones((2, 2), split=1)

        cls.errorneous_type = (2, 2)

    def test_eq(self):
        result = ht.uint8([[0, 1], [0, 0]])

        self.assertTrue(ht.equal(ht.eq(self.a_scalar, self.a_scalar), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.eq(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.eq(self.a_split_tensor, self.a_tensor), result))

        with self.assertRaises(ValueError):
            ht.eq(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.eq(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.eq("self.a_tensor", "s")

    def test_equal(self):
        self.assertTrue(ht.equal(self.a_tensor, self.a_tensor))
        self.assertFalse(ht.equal(self.a_tensor, self.another_tensor))
        self.assertFalse(ht.equal(self.a_tensor, self.a_scalar))
        self.assertFalse(ht.equal(self.another_tensor, self.a_scalar))

    def test_ge(self):
        result = ht.uint8([[0, 1], [1, 1]])
        commutated_result = ht.uint8([[1, 1], [0, 0]])

        self.assertTrue(ht.equal(ht.ge(self.a_scalar, self.a_scalar), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.ge(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.ge(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.ge(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.ge(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.ge("self.a_tensor", "s")

    def test_gt(self):
        result = ht.uint8([[0, 0], [1, 1]])
        commutated_result = ht.uint8([[1, 0], [0, 0]])

        self.assertTrue(ht.equal(ht.gt(self.a_scalar, self.a_scalar), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.gt(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.gt(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.gt(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.gt(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.gt("self.a_tensor", "s")

    def test_le(self):
        result = ht.uint8([[1, 1], [0, 0]])
        commutated_result = ht.uint8([[0, 1], [1, 1]])

        self.assertTrue(ht.equal(ht.le(self.a_scalar, self.a_scalar), ht.uint8([1])))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.le(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.le(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.le(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.le(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.le("self.a_tensor", "s")

    def test_lt(self):
        result = ht.uint8([[1, 0], [0, 0]])
        commutated_result = ht.uint8([[0, 0], [1, 1]])

        self.assertTrue(ht.equal(ht.lt(self.a_scalar, self.a_scalar), ht.uint8([0])))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.lt(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.lt(self.a_split_tensor, self.a_tensor), commutated_result))

        with self.assertRaises(ValueError):
            ht.lt(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.lt(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.lt("self.a_tensor", "s")

    def test_ne(self):
        result = ht.uint8([[1, 0], [1, 1]])

        # self.assertTrue(ht.equal(ht.ne(self.a_scalar, self.a_scalar), ht.uint8([0])))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.a_scalar), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_scalar, self.a_tensor), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.another_tensor), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.a_vector), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.ne(self.a_split_tensor, self.a_tensor), result))
        self.assertTrue(ht.equal(self.a_split_tensor != self.a_tensor, result))

        with self.assertRaises(ValueError):
            ht.ne(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.ne(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.ne("self.a_tensor", "s")
