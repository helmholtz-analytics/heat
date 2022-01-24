import heat as ht
from .test_suites.basic_test import TestCase


class TestRelational(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestRelational, cls).setUpClass()
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
        result = ht.array([[False, True], [False, False]])

        self.assertTrue(ht.equal(ht.eq(self.a_scalar, self.a_scalar), ht.array(True)))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.eq(self.a_scalar, self.a_tensor), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.eq(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.eq(self.a_split_tensor, self.a_tensor), result))

        self.assertEqual(ht.eq(self.a_split_tensor, self.a_tensor).dtype, ht.bool)

        with self.assertRaises(ValueError):
            ht.eq(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.eq(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.eq("self.a_tensor", "s")

    def test_equal(self):
        self.assertTrue(ht.equal(self.a_tensor, self.a_tensor))
        self.assertFalse(ht.equal(self.a_tensor[1:], self.a_tensor))
        self.assertFalse(ht.equal(self.a_split_tensor[1:], self.a_tensor[1:]))
        self.assertFalse(ht.equal(self.a_tensor[1:], self.a_split_tensor[1:]))
        self.assertFalse(ht.equal(self.a_tensor, self.another_tensor))
        self.assertFalse(ht.equal(self.a_tensor, self.a_scalar))
        self.assertFalse(ht.equal(self.a_scalar, self.a_tensor))
        self.assertFalse(ht.equal(self.a_scalar, self.a_tensor[0, 0]))
        self.assertFalse(ht.equal(self.a_tensor[0, 0], self.a_scalar))
        self.assertFalse(ht.equal(self.another_tensor, self.a_scalar))
        self.assertTrue(ht.equal(self.split_ones_tensor[:, 0], self.split_ones_tensor[:, 1]))
        self.assertTrue(ht.equal(self.split_ones_tensor[:, 1], self.split_ones_tensor[:, 0]))
        self.assertFalse(ht.equal(self.a_tensor, self.a_split_tensor))
        self.assertFalse(ht.equal(self.a_split_tensor, self.a_tensor))

        arr = ht.array([[1, 2], [3, 4]], comm=ht.MPI_SELF)
        with self.assertRaises(NotImplementedError):
            ht.equal(self.a_tensor, arr)
        with self.assertRaises(ValueError):
            ht.equal(self.a_split_tensor, self.a_split_tensor.resplit(1))

    def test_ge(self):
        result = ht.uint8([[False, True], [True, True]])
        commutated_result = ht.array([[True, True], [False, False]])

        self.assertTrue(ht.equal(ht.ge(self.a_scalar, self.a_scalar), ht.array(True)))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.ge(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.ge(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.ge(self.a_split_tensor, self.a_tensor), commutated_result))

        self.assertEqual(ht.ge(self.a_split_tensor, self.a_tensor).dtype, ht.bool)

        with self.assertRaises(ValueError):
            ht.ge(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.ge(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.ge("self.a_tensor", "s")

    def test_gt(self):
        result = ht.array([[False, False], [True, True]])
        commutated_result = ht.array([[True, False], [False, False]])

        self.assertTrue(ht.equal(ht.gt(self.a_scalar, self.a_scalar), ht.array(False)))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.gt(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.gt(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.gt(self.a_split_tensor, self.a_tensor), commutated_result))

        self.assertEqual(ht.gt(self.a_split_tensor, self.a_tensor).dtype, ht.bool)

        with self.assertRaises(ValueError):
            ht.gt(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.gt(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.gt("self.a_tensor", "s")

    def test_le(self):
        result = ht.array([[True, True], [False, False]])
        commutated_result = ht.array([[False, True], [True, True]])

        self.assertTrue(ht.equal(ht.le(self.a_scalar, self.a_scalar), ht.array(True)))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.le(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.le(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.le(self.a_split_tensor, self.a_tensor), commutated_result))

        self.assertEqual(ht.le(self.a_split_tensor, self.a_tensor).dtype, ht.bool)

        with self.assertRaises(ValueError):
            ht.le(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.le(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.le("self.a_tensor", "s")

    def test_lt(self):
        result = ht.array([[True, False], [False, False]])
        commutated_result = ht.array([[False, False], [True, True]])

        self.assertTrue(ht.equal(ht.lt(self.a_scalar, self.a_scalar), ht.array(False)))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.a_scalar), result))
        self.assertTrue(ht.equal(ht.lt(self.a_scalar, self.a_tensor), commutated_result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.another_tensor), result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.a_vector), result))
        self.assertTrue(ht.equal(ht.lt(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.lt(self.a_split_tensor, self.a_tensor), commutated_result))

        self.assertEqual(ht.lt(self.a_split_tensor, self.a_tensor).dtype, ht.bool)

        with self.assertRaises(ValueError):
            ht.lt(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.lt(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.lt("self.a_tensor", "s")

    def test_ne(self):
        result = ht.array([[True, False], [True, True]])

        # self.assertTrue(ht.equal(ht.ne(self.a_scalar, self.a_scalar), ht.array([False])))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.a_scalar), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_scalar, self.a_tensor), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.another_tensor), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.a_vector), result))
        # self.assertTrue(ht.equal(ht.ne(self.a_tensor, self.an_int_scalar), result))
        self.assertTrue(ht.equal(ht.ne(self.a_split_tensor, self.a_tensor), result))
        self.assertTrue(ht.equal(self.a_split_tensor != self.a_tensor, result))

        self.assertEqual(ht.ne(self.a_split_tensor, self.a_tensor).dtype, ht.bool)

        with self.assertRaises(ValueError):
            ht.ne(self.a_tensor, self.another_vector)
        with self.assertRaises(TypeError):
            ht.ne(self.a_tensor, self.errorneous_type)
        with self.assertRaises(TypeError):
            ht.ne("self.a_tensor", "s")
