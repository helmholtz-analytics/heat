import unittest

import heat as ht

# +-> bool(kind=b)
# | | | int8, byte
# | | | int16, short
# | | | int32, int
# | | | int64, long
# | | uint8, ubyte
# | float16, half
# | float32, float
# | float64, double(double)


class TestTypes(unittest.TestCase):
    def assert_is_heat_type(self, heat_type):
        self.assertTrue(isinstance(heat_type, type))
        self.assertTrue(issubclass(heat_type, ht.generic))

    def assert_non_instantiable_heat_type(self, heat_type):
        with self.assertRaises(TypeError):
            heat_type()
        self.assert_is_heat_type(heat_type)

    def test_generic(self):
        self.assert_non_instantiable_heat_type(ht.generic)

    def test_number(self):
        self.assert_non_instantiable_heat_type(ht.number)

    def test_integer(self):
        self.assert_non_instantiable_heat_type(ht.integer)

    def test_signedinteger(self):
        self.assert_non_instantiable_heat_type(ht.signedinteger)

    def test_unsignedinteger(self):
        self.assert_non_instantiable_heat_type(ht.unsignedinteger)

    def test_floating(self):
        self.assert_non_instantiable_heat_type(ht.floating)

    def test_flexible(self):
        self.assert_non_instantiable_heat_type(ht.flexible)
