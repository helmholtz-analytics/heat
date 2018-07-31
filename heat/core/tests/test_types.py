import torch
import unittest

import heat as ht


class TestTypes(unittest.TestCase):
    def assert_is_heat_type(self, heat_type):
        self.assertIsInstance(heat_type, type)
        self.assertTrue(issubclass(heat_type, ht.generic))

    def assert_non_instantiable_heat_type(self, heat_type):
        self.assert_is_heat_type(heat_type)
        with self.assertRaises(TypeError):
            heat_type()

    def assert_is_instantiable_heat_type(self, heat_type, torch_type):
        # ensure the correct type hierarchy
        self.assert_is_heat_type(heat_type)

        # check a type constructor without any value
        no_value = heat_type()
        self.assertIsInstance(no_value, ht.tensor)
        self.assertEqual(no_value.shape, (1,))
        self.assertEqual((no_value._tensor__array == 0).all().item(), 1)
        self.assertEqual(no_value._tensor__array.dtype, torch_type)

        # check a type constructor with a complex value
        ground_truth = [
            [3, 2, 1],
            [4, 5, 6]
        ]
        complex_value = heat_type(ground_truth)
        self.assertIsInstance(complex_value, ht.tensor)
        self.assertEqual(complex_value.shape, (2, 3,))
        self.assertEqual((complex_value._tensor__array == torch.tensor(ground_truth, dtype=torch_type)).all().item(), 1)
        self.assertEqual(complex_value._tensor__array.dtype, torch_type)

        # check exception when there is more than one parameter
        with self.assertRaises(TypeError):
            heat_type(ground_truth, ground_truth)

    def test_generic(self):
        self.assert_non_instantiable_heat_type(ht.generic)

    def test_bool(self):
        self.assert_is_instantiable_heat_type(ht.bool, torch.uint8)
        self.assert_is_instantiable_heat_type(ht.bool_, torch.uint8)

    def test_number(self):
        self.assert_non_instantiable_heat_type(ht.number)

    def test_integer(self):
        self.assert_non_instantiable_heat_type(ht.integer)

    def test_signedinteger(self):
        self.assert_non_instantiable_heat_type(ht.signedinteger)

    def test_int8(self):
        self.assert_is_instantiable_heat_type(ht.int8, torch.int8)
        self.assert_is_instantiable_heat_type(ht.byte, torch.int8)

    def test_int16(self):
        self.assert_is_instantiable_heat_type(ht.int16, torch.int16)
        self.assert_is_instantiable_heat_type(ht.short, torch.int16)

    def test_int32(self):
        self.assert_is_instantiable_heat_type(ht.int32, torch.int32)
        self.assert_is_instantiable_heat_type(ht.int, torch.int32)

    def test_int64(self):
        self.assert_is_instantiable_heat_type(ht.int64, torch.int64)
        self.assert_is_instantiable_heat_type(ht.long, torch.int64)

    def test_unsignedinteger(self):
        self.assert_non_instantiable_heat_type(ht.unsignedinteger)

    def test_uint8(self):
        self.assert_is_instantiable_heat_type(ht.uint8, torch.uint8)
        self.assert_is_instantiable_heat_type(ht.ubyte, torch.uint8)

    def test_floating(self):
        self.assert_non_instantiable_heat_type(ht.floating)

    def test_float32(self):
        self.assert_is_instantiable_heat_type(ht.float32, torch.float32)
        self.assert_is_instantiable_heat_type(ht.float, torch.float32)
        self.assert_is_instantiable_heat_type(ht.float_, torch.float32)

    def test_float64(self):
        self.assert_is_instantiable_heat_type(ht.float64, torch.float64)
        self.assert_is_instantiable_heat_type(ht.double, torch.float64)

    def test_flexible(self):
        self.assert_non_instantiable_heat_type(ht.flexible)
