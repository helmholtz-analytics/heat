import numpy as np
import torch
import heat as ht

from .test_suites.basic_test import TestCase


class TestComplex(TestCase):
    def test_abs(self):
        a = ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], split=0)
        absolute = ht.absolute(a)
        res = torch.abs(a.larray)

        self.assertIs(absolute.device, self.device)
        self.assertIs(absolute.dtype, ht.float)
        self.assertEqual(absolute.shape, (5,))
        self.assertTrue(torch.equal(absolute.larray, res))

        a = ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], dtype=ht.complex128)
        absolute = ht.absolute(a)
        res = torch.abs(a.larray)

        self.assertIs(absolute.device, self.device)
        self.assertIs(absolute.dtype, ht.double)
        self.assertEqual(absolute.shape, (5,))
        self.assertTrue(torch.equal(absolute.larray, res))

    def test_angle(self):
        a = ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], split=0)
        angle = ht.angle(a)
        res = torch.angle(a.larray)

        self.assertIs(angle.device, self.device)
        self.assertIs(angle.dtype, ht.float)
        self.assertEqual(angle.shape, (5,))
        self.assertTrue(torch.equal(angle.larray, res))

        ht.angle(ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], dtype=ht.complex128), deg=True)
        angle = ht.angle(a, deg=True)
        res = ht.array(
            [0.0, 90.0, 45.0, 135.0, -45.0], dtype=ht.float32, device=self.device, split=0
        )

        self.assertIs(angle.device, self.device)
        self.assertIs(angle.dtype, ht.float32)
        self.assertEqual(angle.shape, (5,))
        self.assertTrue(ht.equal(angle, res))

        # Not complex
        a = ht.ones((4, 4), split=1)
        angle = ht.angle(a)
        res = ht.zeros((4, 4), split=1)

        self.assertIs(angle.device, self.device)
        self.assertIs(angle.dtype, ht.float32)
        self.assertEqual(angle.shape, (4, 4))
        self.assertTrue(ht.equal(angle, res))

    def test_conjugate(self):
        a = ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], split=0)
        conj = ht.conjugate(a)
        res = ht.array(
            [1 - 0j, -1j, 1 - 1j, -2 - 2j, 3 + 3j], dtype=ht.complex64, device=self.device, split=0
        )

        self.assertIs(conj.device, self.device)
        self.assertIs(conj.dtype, ht.complex64)
        self.assertEqual(conj.shape, (5,))
        # equal on complex numbers does not work on PyTorch
        self.assertTrue(ht.equal(ht.real(conj), ht.real(res)))
        self.assertTrue(ht.equal(ht.imag(conj), ht.imag(res)))

        # Not complex
        a = ht.ones((4, 4))
        conj = ht.conj(a)
        res = ht.ones((4, 4))

        self.assertIs(conj.device, self.device)
        self.assertIs(conj.dtype, ht.float32)
        self.assertEqual(conj.shape, (4, 4))
        self.assertTrue(ht.equal(conj, res))

    def test_imag(self):
        a = ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], split=0)
        imag = ht.imag(a)
        res = ht.array([0.0, 1.0, 1.0, 2.0, -3.0], dtype=ht.float32, device=self.device, split=0)

        self.assertIs(imag.device, self.device)
        self.assertIs(imag.dtype, ht.float)
        self.assertEqual(imag.shape, (5,))
        self.assertTrue(ht.equal(imag, res))

        # Not complex
        a = ht.ones((4, 4))
        imag = a.imag
        res = ht.zeros((4, 4))

        self.assertIs(imag.device, self.device)
        self.assertIs(imag.dtype, ht.float32)
        self.assertEqual(imag.shape, (4, 4))
        self.assertTrue(ht.equal(imag, res))

    def test_real(self):
        a = ht.array([1.0, 1.0j, 1 + 1j, -2 + 2j, 3 - 3j], split=0)
        real = ht.real(a)
        res = ht.array([1.0, 0.0, 1.0, -2.0, 3.0], dtype=ht.float32, device=self.device, split=0)

        self.assertIs(real.device, self.device)
        self.assertIs(real.dtype, ht.float)
        self.assertEqual(real.shape, (5,))
        self.assertTrue(ht.equal(real, res))

        # Not complex
        a = ht.ones((4, 4), split=1)
        real = a.real
        res = ht.ones((4, 4), split=1)

        self.assertIs(real.device, self.device)
        self.assertIs(real.dtype, ht.float32)
        self.assertEqual(real.shape, (4, 4))
        self.assertIs(real, a)

    # This test will be redundant with PyTorch 1.7
    def test_full(self):
        a = ht.full((4, 4), 1 + 1j)

        self.assertIs(a.dtype, ht.complex64)
