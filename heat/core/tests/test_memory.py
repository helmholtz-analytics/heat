import unittest
import torch
import numpy as np
import heat as ht

from heat.core.tests.test_suites.basic_test import BasicTest


class TestMemory(unittest.TestCase):
    def test_copy(self):
        tensor = ht.ones(5)
        copied = tensor.copy()

        # test identity inequality and value equality
        self.assertIsNot(tensor, copied)
        self.assertIsNot(tensor._DNDarray__array, copied._DNDarray__array)
        self.assertTrue((tensor == copied)._DNDarray__array.all())

        # test exceptions
        with self.assertRaises(TypeError):
            ht.copy("hello world")

    def test_sanitize_memory_layout(self):
        # non distributed, 2D
        a_torch = torch.arange(12).reshape(4, 3)
        a_heat_C = ht.array(a_torch)
        a_heat_F = ht.array(a_torch, order="F")
        BasicTest.assertTrue_memory_layout(self, a_heat_C, "C")
        BasicTest.assertTrue_memory_layout(self, a_heat_F, "F")
        # non distributed, 5D
        a_torch_5d = torch.arange(4 * 3 * 5 * 2 * 1).reshape(4, 3, 1, 2, 5)
        a_heat_5d_C = ht.array(a_torch_5d)
        a_heat_5d_F = ht.array(a_torch_5d, order="F")
        BasicTest.assertTrue_memory_layout(self, a_heat_5d_C, "C")
        BasicTest.assertTrue_memory_layout(self, a_heat_5d_F, "F")
        # distributed, split, 2D
        # distributed, split, 4D
        # distributed, is_split, 2D
        # distributed, is_split, 4D
        # distributed, after reduction operation

