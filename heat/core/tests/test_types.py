import numpy as np
import torch

import heat as ht
from .test_suites.basic_test import TestCase


class TestTypes(TestCase):
    def assert_is_heat_type(self, heat_type):
        self.assertIsInstance(heat_type, type)
        self.assertTrue(issubclass(heat_type, ht.datatype))

    def assert_non_instantiable_heat_type(self, heat_type):
        self.assert_is_heat_type(heat_type)
        with self.assertRaises(TypeError):
            heat_type()

    def assert_is_instantiable_heat_type(self, heat_type, torch_type):
        # ensure the correct type hierarchy
        self.assert_is_heat_type(heat_type)

        # check a type constructor without any value
        no_value = heat_type()
        self.assertIsInstance(no_value, ht.DNDarray)
        self.assertEqual(no_value.shape, (1,))
        self.assertEqual((no_value.larray == 0).all().item(), 1)
        self.assertEqual(no_value.larray.dtype, torch_type)

        # check a type constructor with a complex value
        ground_truth = [[3, 2, 1], [4, 5, 6]]
        elaborate_value = heat_type(ground_truth)
        self.assertIsInstance(elaborate_value, ht.DNDarray)
        self.assertEqual(elaborate_value.shape, (2, 3))
        self.assertEqual(
            (
                elaborate_value.larray
                == torch.tensor(ground_truth, dtype=torch_type, device=self.device.torch_device)
            )
            .all()
            .item(),
            1,
        )
        self.assertEqual(elaborate_value.larray.dtype, torch_type)

        # check exception when there is more than one parameter
        with self.assertRaises(TypeError):
            heat_type(ground_truth, ground_truth)

    def test_generic(self):
        self.assert_non_instantiable_heat_type(ht.datatype)

    def test_bool(self):
        self.assert_is_instantiable_heat_type(ht.bool, torch.bool)
        self.assert_is_instantiable_heat_type(ht.bool_, torch.bool)

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

    def test_complex64(self):
        self.assert_is_instantiable_heat_type(ht.complex64, torch.complex64)
        self.assert_is_instantiable_heat_type(ht.cfloat, torch.complex64)
        self.assert_is_instantiable_heat_type(ht.csingle, torch.complex64)

        self.assertEqual(ht.complex64.char(), "c8")

    def test_complex128(self):
        self.assert_is_instantiable_heat_type(ht.complex128, torch.complex128)
        self.assert_is_instantiable_heat_type(ht.cdouble, torch.complex128)

        self.assertEqual(ht.complex128.char(), "c16")

    def test_iscomplex(self):
        a = ht.array([1, 1.2, 1 + 1j, 1 + 0j])
        s = ht.array([False, False, True, False])
        r = ht.iscomplex(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, 1.2, True], split=0)
        s = ht.array([False, False, False], split=0)
        r = ht.iscomplex(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.zeros((6, 6), dtype=ht.bool, split=0)
        r = ht.iscomplex(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.full((5, 5), 1 + 1j, dtype=ht.int, split=1)
        s = ht.ones((5, 5), dtype=ht.bool, split=1)
        r = ht.iscomplex(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

    def test_isreal(self):
        a = ht.array([1, 1.2, 1 + 1j, 1 + 0j])
        s = ht.array([True, True, False, True])
        r = ht.isreal(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.array([1, 1.2, True], split=0)
        s = ht.array([True, True, True], split=0)
        r = ht.isreal(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.ones((6, 6), dtype=ht.bool, split=0)
        s = ht.ones((6, 6), dtype=ht.bool, split=0)
        r = ht.isreal(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))

        a = ht.full((5, 5), 1 + 1j, dtype=ht.int, split=1)
        s = ht.zeros((5, 5), dtype=ht.bool, split=1)
        r = ht.isreal(a)
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(r.dtype, s.dtype)
        self.assertEqual(r.device, s.device)
        self.assertTrue(ht.equal(r, s))


class TestTypeConversion(TestCase):
    def test_can_cast(self):
        zeros_array = np.zeros((3,), dtype=np.int16)

        # casting - 'no'
        self.assertTrue(ht.can_cast(ht.uint8, ht.uint8, casting="no"))
        self.assertFalse(ht.can_cast(ht.uint8, ht.int16, casting="no"))
        self.assertFalse(ht.can_cast(ht.uint8, ht.int8, casting="no"))
        self.assertFalse(ht.can_cast(ht.float64, ht.bool, casting="no"))
        self.assertTrue(ht.can_cast(1.0, ht.float32, casting="no"))
        self.assertFalse(ht.can_cast(zeros_array, ht.float32, casting="no"))

        # casting - 'safe'
        self.assertTrue(ht.can_cast(ht.uint8, ht.uint8, casting="safe"))
        self.assertTrue(ht.can_cast(ht.uint8, ht.int16, casting="safe"))
        self.assertFalse(ht.can_cast(ht.uint8, ht.int8, casting="safe"))
        self.assertFalse(ht.can_cast(ht.float64, ht.bool, casting="safe"))
        self.assertTrue(ht.can_cast(1.0, ht.float32, casting="safe"))
        self.assertTrue(ht.can_cast(zeros_array, ht.float32, casting="safe"))

        # casting - 'same_kind'
        self.assertTrue(ht.can_cast(ht.uint8, ht.uint8, casting="same_kind"))
        self.assertTrue(ht.can_cast(ht.uint8, ht.int16, casting="same_kind"))
        self.assertTrue(ht.can_cast(ht.uint8, ht.int8, casting="same_kind"))
        self.assertFalse(ht.can_cast(ht.float64, ht.bool, casting="same_kind"))
        self.assertTrue(ht.can_cast(1.0, ht.float32, casting="same_kind"))
        self.assertTrue(ht.can_cast(zeros_array, ht.float32, casting="same_kind"))

        # casting - 'unsafe'
        self.assertTrue(ht.can_cast(ht.uint8, ht.uint8, casting="unsafe"))
        self.assertTrue(ht.can_cast(ht.uint8, ht.int16, casting="unsafe"))
        self.assertTrue(ht.can_cast(ht.uint8, ht.int8, casting="unsafe"))
        self.assertTrue(ht.can_cast(ht.float64, ht.bool, casting="unsafe"))
        self.assertTrue(ht.can_cast(1.0, ht.float32, casting="unsafe"))
        self.assertTrue(ht.can_cast(zeros_array, ht.float32, casting="unsafe"))

        # exceptions
        with self.assertRaises(TypeError):
            ht.can_cast(ht.uint8, ht.uint8, casting=1)
        with self.assertRaises(ValueError):
            ht.can_cast(ht.uint8, ht.uint8, casting="hello world")
        with self.assertRaises(TypeError):
            ht.can_cast({}, ht.uint8, casting="unsafe")
        with self.assertRaises(TypeError):
            ht.can_cast(ht.uint8, {}, casting="unsafe")

    def test_canonical_heat_type(self):
        self.assertEqual(ht.core.types.canonical_heat_type(ht.float32), ht.float32)
        self.assertEqual(ht.core.types.canonical_heat_type("?"), ht.bool)
        self.assertEqual(ht.core.types.canonical_heat_type(int), ht.int32)
        self.assertEqual(ht.core.types.canonical_heat_type("u1"), ht.uint8)
        self.assertEqual(ht.core.types.canonical_heat_type(np.int8), ht.int8)
        self.assertEqual(ht.core.types.canonical_heat_type(torch.short), ht.int16)
        self.assertEqual(ht.core.types.canonical_heat_type(torch.cfloat), ht.complex64)

        with self.assertRaises(TypeError):
            ht.core.types.canonical_heat_type({})
        with self.assertRaises(TypeError):
            ht.core.types.canonical_heat_type(object)
        with self.assertRaises(TypeError):
            ht.core.types.canonical_heat_type(1)
        with self.assertRaises(TypeError):
            ht.core.types.canonical_heat_type("i7")

    def test_heat_type_of(self):
        ht_tensor = ht.zeros((1,), dtype=ht.bool)
        self.assertEqual(ht.core.types.heat_type_of(ht_tensor), ht.bool)

        np_array = np.ones((3,), dtype=np.int32)
        self.assertEqual(ht.core.types.heat_type_of(np_array), ht.int32)

        scalar = 2.0
        self.assertEqual(ht.core.types.heat_type_of(scalar), ht.float32)

        iterable = [3, "hello world"]
        self.assertEqual(ht.core.types.heat_type_of(iterable), ht.int32)

        torch_tensor = torch.full((2,), 1 + 1j, dtype=torch.complex128)
        self.assertEqual(ht.core.types.heat_type_of(torch_tensor), ht.complex128)

        with self.assertRaises(TypeError):
            ht.core.types.heat_type_of({})
        with self.assertRaises(TypeError):
            ht.core.types.heat_type_of(object)

    def test_issubdtype(self):
        # First level
        self.assertTrue(ht.issubdtype(ht.bool, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.bool_, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.number, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.integer, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.signedinteger, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.unsignedinteger, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.floating, ht.datatype))
        self.assertTrue(ht.issubdtype(ht.flexible, ht.datatype))

        # Second level
        self.assertTrue(ht.issubdtype(ht.integer, ht.number))
        self.assertTrue(ht.issubdtype(ht.floating, ht.number))
        self.assertTrue(ht.issubdtype(ht.signedinteger, ht.integer))
        self.assertTrue(ht.issubdtype(ht.unsignedinteger, ht.integer))

        # Third level
        self.assertTrue(ht.issubdtype(ht.int8, ht.signedinteger))
        self.assertTrue(ht.issubdtype(ht.int16, ht.signedinteger))
        self.assertTrue(ht.issubdtype(ht.int32, ht.signedinteger))
        self.assertTrue(ht.issubdtype(ht.int64, ht.signedinteger))
        self.assertTrue(ht.issubdtype(ht.uint8, ht.unsignedinteger))
        self.assertTrue(ht.issubdtype(ht.float32, ht.floating))
        self.assertTrue(ht.issubdtype(ht.float64, ht.floating))

        # Fourth level
        self.assertTrue(ht.issubdtype(ht.byte, ht.int8))
        self.assertTrue(ht.issubdtype(ht.short, ht.int16))
        self.assertTrue(ht.issubdtype(ht.int, ht.int32))
        self.assertTrue(ht.issubdtype(ht.long, ht.int64))
        self.assertTrue(ht.issubdtype(ht.uint8, ht.ubyte))
        self.assertTrue(ht.issubdtype(ht.float32, ht.float))
        self.assertTrue(ht.issubdtype(ht.float32, ht.float_))
        self.assertTrue(ht.issubdtype(ht.float64, ht.double))

        # Small tests char representations (-> canonical_heat_type)
        self.assertTrue("i", ht.int8)
        self.assertTrue(ht.issubdtype("B", ht.uint8))
        self.assertTrue(ht.issubdtype(ht.float64, "f8"))

        # Small tests Exceptions (-> canonical_heat_type)
        with self.assertRaises(TypeError):
            ht.issubdtype(ht.bool, True)
        with self.assertRaises(TypeError):
            ht.issubdtype(4.2, "f")
        with self.assertRaises(TypeError):
            ht.issubdtype({}, ht.int)

    def test_type_promotions(self):
        self.assertEqual(ht.promote_types(ht.uint8, ht.uint8), ht.uint8)
        self.assertEqual(ht.promote_types(ht.int8, ht.uint8), ht.int16)
        self.assertEqual(ht.promote_types(ht.int32, ht.float32), ht.float32)
        self.assertEqual(ht.promote_types("f4", ht.float), ht.float32)
        self.assertEqual(ht.promote_types(ht.bool_, "?"), ht.bool)
        self.assertEqual(ht.promote_types(ht.float32, ht.complex64), ht.complex64)

        # exceptions
        with self.assertRaises(TypeError):
            ht.promote_types(1, "?")
        with self.assertRaises(TypeError):
            ht.promote_types(ht.float32, "hello world")

    def test_result_type(self):
        self.assertEqual(ht.result_type(1), ht.int32)
        self.assertEqual(ht.result_type(1, 1.0), ht.float32)
        self.assertEqual(ht.result_type(1.0, True, 1 + 1j), ht.complex64)
        self.assertEqual(ht.result_type(ht.array(1, dtype=ht.int32), 1), ht.int32)
        self.assertEqual(ht.result_type(1.0, ht.array(1, dtype=ht.int32)), ht.float32)
        self.assertEqual(ht.result_type(ht.uint8, ht.int8), ht.int16)
        self.assertEqual(ht.result_type("b", "f4"), ht.float32)
        self.assertEqual(ht.result_type(ht.array([1], dtype=ht.float64), "f4"), ht.float64)
        self.assertEqual(
            ht.result_type(
                ht.array([1, 2, 3, 4], dtype=ht.float64, split=0),
                1,
                ht.bool,
                "u",
                torch.uint8,
                np.complex128,
                ht.array(1, dtype=ht.int64),
            ),
            ht.complex128,
        )
        self.assertEqual(
            ht.result_type(np.array([1, 2, 3]), np.dtype("int32"), torch.tensor([1, 2, 3])),
            ht.int64,
        )

    def test_finfo(self):
        info32 = ht.finfo(ht.float32)
        self.assertEqual(info32.bits, 32)
        self.assertEqual(info32.max, (2 - 2**-23) * 2**127)
        self.assertEqual(info32.min, -info32.max)
        self.assertEqual(info32.eps, 2**-23)

        with self.assertRaises(TypeError):
            ht.finfo(1)

        with self.assertRaises(TypeError):
            ht.finfo(ht.int32)

        with self.assertRaises(TypeError):
            ht.finfo("float16")

    def test_iinfo(self):
        info32 = ht.iinfo(ht.int32)
        self.assertEqual(info32.bits, 32)
        self.assertEqual(info32.max, 2147483647)
        self.assertEqual(info32.min, -2147483648)

        with self.assertRaises(TypeError):
            ht.iinfo(1.0)

        with self.assertRaises(TypeError):
            ht.iinfo(ht.float64)

        with self.assertRaises(TypeError):
            ht.iinfo("int16")
