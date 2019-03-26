import unittest

import heat as ht

FLOAT_EPSILON = 1e-4

T = ht.float32([
    [1, 2],
    [3, 4]
])
s = 2.0
s_int = 2
T1 = ht.float32([
    [2, 2],
    [2, 2]
])
v = ht.float32([2, 2])
v2 = ht.float32([2, 2, 2])
T_s = ht.tensor(T1._tensor__array, T1.shape, T1.dtype, 0, T1.device, T1.comm)
otherType = (2,2)

class TestOperations(unittest.TestCase):

    def test_add(self):
        T_r = ht.float32([
            [3, 4],
            [5, 6]
        ])

        self.assertTrue(ht.equal(ht.add(s, s), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.add(T, s),T_r))
        self.assertTrue(ht.equal(ht.add(s, T), T_r))
        self.assertTrue(ht.equal(ht.add(T, T1), T_r))
        self.assertTrue(ht.equal(ht.add(T, v), T_r))
        self.assertTrue(ht.equal(ht.add(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.add(T_s, T), T_r))

        with self.assertRaises(ValueError):
            ht.add(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.add(T, T_s)
        with self.assertRaises(TypeError):
            ht.add(T, otherType)
        with self.assertRaises(TypeError):
            ht.add('T', 's')

    def test_sub(self):
        T_r = ht.float32([
            [-1, 0],
            [1, 2]
        ])

        T_r_minus = ht.float32([
            [1, 0],
            [-1, -2]
        ])

        self.assertTrue(ht.equal(ht.sub(s, s), ht.float32([0.0])))
        self.assertTrue(ht.equal(ht.sub(T, s),T_r))
        self.assertTrue(ht.equal(ht.sub(s, T), T_r_minus))
        self.assertTrue(ht.equal(ht.sub(T, T1), T_r))
        self.assertTrue(ht.equal(ht.sub(T, v), T_r))
        self.assertTrue(ht.equal(ht.sub(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.sub(T_s, T), T_r_minus))

        with self.assertRaises(ValueError):
            ht.sub(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.sub(T, T_s)
        with self.assertRaises(TypeError):
            ht.sub(T, otherType)
        with self.assertRaises(TypeError):
            ht.sub('T', 's')

    def test_mul(self):
        T_r = ht.float32([
            [2, 4],
            [6, 8]
        ])

        self.assertTrue(ht.equal(ht.mul(s, s), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.mul(T, s),T_r))
        self.assertTrue(ht.equal(ht.mul(s, T), T_r))
        self.assertTrue(ht.equal(ht.mul(T, T1), T_r))
        self.assertTrue(ht.equal(ht.mul(T, v), T_r))
        self.assertTrue(ht.equal(ht.mul(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.mul(T_s, T), T_r))


        with self.assertRaises(ValueError):
            ht.mul(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.mul(T, T_s)
        with self.assertRaises(TypeError):
            ht.mul(T, otherType)
        with self.assertRaises(TypeError):
            ht.mul('T', 's')

    def test_div(self):
        T_r = ht.float32([
            [0.5, 1],
            [1.5, 2]
        ])

        T_inv = ht.float32([
            [2, 1],
            [2/3, 0.5]
        ])

        self.assertTrue(ht.equal(ht.div(s, s), ht.float32([1.0])))
        self.assertTrue(ht.equal(ht.div(T, s),T_r))
        self.assertTrue(ht.equal(ht.div(s, T), T_inv))
        self.assertTrue(ht.equal(ht.div(T, T1), T_r))
        self.assertTrue(ht.equal(ht.div(T, v), T_r))
        self.assertTrue(ht.equal(ht.div(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.div(T_s, T), T_inv))


        with self.assertRaises(ValueError):
            ht.div(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.div(T, T_s)
        with self.assertRaises(TypeError):
            ht.div(T, otherType)
        with self.assertRaises(TypeError):
            ht.div('T', 's')

    def test_pow(self):
        T_r = ht.float32([
            [1, 4],
            [9, 16]
        ])

        T_inv = ht.float32([
            [2, 4],
            [8, 16]
        ])

        self.assertTrue(ht.equal(ht.pow(s, s), ht.float32([4.0])))
        self.assertTrue(ht.equal(ht.pow(T, s), T_r))
        self.assertTrue(ht.equal(ht.pow(s, T), T_inv))
        self.assertTrue(ht.equal(ht.pow(T, T1), T_r))
        self.assertTrue(ht.equal(ht.pow(T, v), T_r))
        self.assertTrue(ht.equal(ht.pow(T, s_int), T_r))
        self.assertTrue(ht.equal(ht.pow(T_s, T), T_inv))

        with self.assertRaises(ValueError):
            ht.pow(T, v2)
        with self.assertRaises(NotImplementedError):
            ht.pow(T, T_s)
        with self.assertRaises(TypeError):
            ht.pow(T, otherType)
        with self.assertRaises(TypeError):
            ht.pow('T', 's')
