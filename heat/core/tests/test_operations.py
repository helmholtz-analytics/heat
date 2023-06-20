import torch
import platform

import heat as ht
import numpy as np
from .test_suites.basic_test import TestCase


class TestOperations(TestCase):
    def test___binary_bit_op_broadcast(self):
        # broadcast without split
        left_tensor = ht.ones((4, 1), dtype=ht.int32)
        right_tensor = ht.ones((1, 2), dtype=ht.int32)
        result = left_tensor & right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor & left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=0 for both operants
        left_tensor = ht.ones((4, 1), split=0, dtype=ht.int32)
        right_tensor = ht.ones((1, 2), split=0, dtype=ht.int32)
        result = left_tensor | right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor | left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=1 for both operants
        left_tensor = ht.ones((4, 1), split=1, dtype=ht.int32)
        right_tensor = ht.ones((1, 2), split=1, dtype=ht.int32)
        result = left_tensor ^ right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor ^ left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=1 for second operant
        left_tensor = ht.ones((4, 1), dtype=ht.int32)
        right_tensor = ht.ones((1, 2), split=1, dtype=ht.int32)
        result = left_tensor & right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor & left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with split=0 for first operant
        left_tensor = ht.ones((4, 1), split=0, dtype=ht.int32)
        right_tensor = ht.ones((1, 2), dtype=ht.int32)
        result = left_tensor | right_tensor
        self.assertEqual(result.shape, (4, 2))
        result = right_tensor | left_tensor
        self.assertEqual(result.shape, (4, 2))

        # broadcast with unequal dimensions and one splitted tensor
        left_tensor = ht.ones((2, 4, 1), split=0, dtype=ht.int32)
        right_tensor = ht.ones((1, 2), dtype=ht.int32)
        result = left_tensor ^ right_tensor
        self.assertEqual(result.shape, (2, 4, 2))
        result = right_tensor ^ left_tensor
        self.assertEqual(result.shape, (2, 4, 2))

        # broadcast with unequal dimensions, a scalar, and one splitted tensor
        left_scalar = ht.np.int32(1)
        right_tensor = ht.ones((1, 2), split=0, dtype=ht.int32)
        result = ht.bitwise_or(left_scalar, right_tensor)
        self.assertEqual(result.shape, (1, 2))
        result = right_tensor | left_scalar
        self.assertEqual(result.shape, (1, 2))

        # broadcast with unequal dimensions and two splitted tensors
        left_tensor = ht.ones((4, 1, 3, 1, 2), split=2, dtype=torch.uint8)
        right_tensor = ht.ones((1, 3, 1), split=0, dtype=torch.uint8)
        result = left_tensor & right_tensor
        self.assertEqual(result.shape, (4, 1, 3, 3, 2))
        result = right_tensor & left_tensor
        self.assertEqual(result.shape, (4, 1, 3, 3, 2))

        with self.assertRaises(TypeError):
            ht.bitwise_and(ht.ones((1, 2)), "wrong type")
        if ht.MPI_WORLD.size > 1:
            with self.assertRaises(NotImplementedError):
                ht.bitwise_or(
                    ht.ones((1, 2), dtype=ht.int32, split=0),
                    ht.ones((1, 2), dtype=ht.int32, split=1),
                )

        a = ht.ones((4, 4), split=None)
        b = ht.zeros((4, 4), split=0)
        self.assertTrue(ht.equal(a * b, b))
        self.assertTrue(ht.equal(b * a, b))
        self.assertTrue(ht.equal(a[0] * b[0], b[0]))
        self.assertTrue(ht.equal(b[0] * a[0], b[0]))
        self.assertTrue(ht.equal(a * b[0:1], b))
        self.assertTrue(ht.equal(b[0:1] * a, b))
        self.assertTrue(ht.equal(a[0:1] * b, b))
        self.assertTrue(ht.equal(b * a[0:1], b))

        if ht.MPI_WORLD.size > 1:
            c = ht.array([1, 2, 3, 4], comm=ht.MPI_SELF)
            with self.assertRaises(NotImplementedError):
                b + c
            with self.assertRaises(NotImplementedError):
                a.resplit(1) * b
        # skip tests on arm64 architecture
        if platform.machine() != "arm64":
            with self.assertRaises(TypeError):
                ht.minimum(a, np.float128(1))
            with self.assertRaises(TypeError):
                ht.minimum(np.float128(1), a)
        with self.assertRaises(ValueError):
            a[2:] * b
