import torch
import unittest
import math
import heat as ht
import os

if os.environ.get("DEVICE") == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ht.use_device("gpu" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
    ht.use_device("cpu")


class TestTrigonometrics(unittest.TestCase):
    def test_rad2deg(self):
        # base elements
        elements = [0.0, 0.2, 0.6, 0.9, 1.2, 2.7, 3.14]
        comparison = (
            180.0 * torch.tensor(elements, dtype=torch.float64, device=device) / 3.141592653589793
        )

        # rad2deg with float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_rad2deg = ht.rad2deg(float32_tensor)
        self.assertIsInstance(float32_rad2deg, ht.DNDarray)
        self.assertEqual(float32_rad2deg.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_rad2deg._DNDarray__array.double(), comparison))

        # rad2deg with float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_rad2deg = ht.rad2deg(float64_tensor)
        self.assertIsInstance(float64_rad2deg, ht.DNDarray)
        self.assertEqual(float64_rad2deg.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_rad2deg._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.rad2deg([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.rad2deg("hello world")

    def test_degrees(self):
        # base elements
        elements = [0.0, 0.2, 0.6, 0.9, 1.2, 2.7, 3.14]
        comparison = (
            180.0 * torch.tensor(elements, dtype=torch.float64, device=device) / 3.141592653589793
        )

        # degrees with float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_degrees = ht.degrees(float32_tensor)
        self.assertIsInstance(float32_degrees, ht.DNDarray)
        self.assertEqual(float32_degrees.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_degrees._DNDarray__array.double(), comparison))

        # degrees with float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_degrees = ht.degrees(float64_tensor)
        self.assertIsInstance(float64_degrees, ht.DNDarray)
        self.assertEqual(float64_degrees.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_degrees._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.degrees([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.degrees("hello world")

    def test_deg2rad(self):
        # base elements
        elements = [0.0, 20.0, 45.0, 78.0, 94.0, 120.0, 180.0, 270.0, 311.0]
        comparison = (
            3.141592653589793 * torch.tensor(elements, dtype=torch.float64, device=device) / 180.0
        )

        # deg2rad with float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_deg2rad = ht.deg2rad(float32_tensor)
        self.assertIsInstance(float32_deg2rad, ht.DNDarray)
        self.assertEqual(float32_deg2rad.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_deg2rad._DNDarray__array.double(), comparison))

        # deg2rad with float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_deg2rad = ht.deg2rad(float64_tensor)
        self.assertIsInstance(float64_deg2rad, ht.DNDarray)
        self.assertEqual(float64_deg2rad.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_deg2rad._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.deg2rad([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.deg2rad("hello world")

    def test_radians(self):
        # base elements
        elements = [0.0, 20.0, 45.0, 78.0, 94.0, 120.0, 180.0, 270.0, 311.0]
        comparison = (
            3.141592653589793 * torch.tensor(elements, dtype=torch.float64, device=device) / 180.0
        )

        # radians with float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_radians = ht.radians(float32_tensor)
        self.assertIsInstance(float32_radians, ht.DNDarray)
        self.assertEqual(float32_radians.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_radians._DNDarray__array.double(), comparison))

        # radians with float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_radians = ht.radians(float64_tensor)
        self.assertIsInstance(float64_radians, ht.DNDarray)
        self.assertEqual(float64_radians.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_radians._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.radians([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.radians("hello world")

    def test_arctan(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).atan()

        # arctan of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_arctan = ht.arctan(float32_tensor)
        self.assertIsInstance(float32_arctan, ht.DNDarray)
        self.assertEqual(float32_arctan.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_arctan._DNDarray__array.double(), comparison))

        # arctan of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_arctan = ht.arctan(float64_tensor)
        self.assertIsInstance(float64_arctan, ht.DNDarray)
        self.assertEqual(float64_arctan.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_arctan._DNDarray__array.double(), comparison))

        # arctan of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_arctan = ht.arctan(int32_tensor)
        self.assertIsInstance(int32_arctan, ht.DNDarray)
        self.assertEqual(int32_arctan.dtype, ht.float64)
        self.assertTrue(torch.allclose(float32_arctan._DNDarray__array.double(), comparison))

        # arctan of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_arctan = ht.arctan(int64_tensor)
        self.assertIsInstance(int64_arctan, ht.DNDarray)
        self.assertEqual(int64_arctan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_arctan._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.arctan([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.arctan("hello world")

    def test_arcsin(self):
        # base elements
        elements = [-1.0, -0.83, -0.12, 0.0, 0.24, 0.67, 1.0]
        comparison = torch.tensor(elements, dtype=torch.float64, device=device).asin()

        # arcsin of float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_arcsin = ht.arcsin(float32_tensor)
        self.assertIsInstance(float32_arcsin, ht.DNDarray)
        self.assertEqual(float32_arcsin.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_arcsin._DNDarray__array.double(), comparison))

        # arcsin of float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_arcsin = ht.arcsin(float64_tensor)
        self.assertIsInstance(float64_arcsin, ht.DNDarray)
        self.assertEqual(float64_arcsin.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_arcsin._DNDarray__array.double(), comparison))

        # arcsin of value out of domain
        nan_tensor = ht.array([1.2])
        nan_arcsin = ht.arcsin(nan_tensor)
        self.assertIsInstance(float64_arcsin, ht.DNDarray)
        self.assertEqual(nan_arcsin.dtype, ht.float32)
        self.assertTrue(math.isnan(nan_arcsin._DNDarray__array.item()))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.arcsin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.arcsin("hello world")

    def test_arccos(self):
        # base elements
        elements = [-1.0, -0.83, -0.12, 0.0, 0.24, 0.67, 1.0]
        comparison = torch.tensor(elements, dtype=torch.float64, device=device).acos()

        # arccos of float32
        float32_tensor = ht.array(elements, dtype=ht.float32)
        float32_arccos = ht.arccos(float32_tensor)
        self.assertIsInstance(float32_arccos, ht.DNDarray)
        self.assertEqual(float32_arccos.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_arccos._DNDarray__array.double(), comparison))

        # arccos of float64
        float64_tensor = ht.array(elements, dtype=ht.float64)
        float64_arccos = ht.arccos(float64_tensor)
        self.assertIsInstance(float64_arccos, ht.DNDarray)
        self.assertEqual(float64_arccos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_arccos._DNDarray__array.double(), comparison))

        # arccos of value out of domain
        nan_tensor = ht.array([1.2])
        nan_arccos = ht.arccos(nan_tensor)
        self.assertIsInstance(float64_arccos, ht.DNDarray)
        self.assertEqual(nan_arccos.dtype, ht.float32)
        self.assertTrue(math.isnan(nan_arccos._DNDarray__array.item()))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.arccos([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.arccos("hello world")

    def test_cos(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).cos()

        # cosine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_cos = ht.cos(float32_tensor)
        self.assertIsInstance(float32_cos, ht.DNDarray)
        self.assertEqual(float32_cos.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_cos._DNDarray__array.double(), comparison))

        # cosine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_cos = ht.cos(float64_tensor)
        self.assertIsInstance(float64_cos, ht.DNDarray)
        self.assertEqual(float64_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_cos._DNDarray__array.double(), comparison))

        # cosine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_cos = ht.cos(int32_tensor)
        self.assertIsInstance(int32_cos, ht.DNDarray)
        self.assertEqual(int32_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(float32_cos._DNDarray__array.double(), comparison))

        # cosine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_cos = int64_tensor.cos()
        self.assertIsInstance(int64_cos, ht.DNDarray)
        self.assertEqual(int64_cos.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_cos._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.cos([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.cos("hello world")

    def test_cosh(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).cosh()

        # hyperbolic cosine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_cosh = float32_tensor.cosh()
        self.assertIsInstance(float32_cosh, ht.DNDarray)
        self.assertEqual(float32_cosh.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_cosh._DNDarray__array.double(), comparison))

        # hyperbolic cosine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_cosh = ht.cosh(float64_tensor)
        self.assertIsInstance(float64_cosh, ht.DNDarray)
        self.assertEqual(float64_cosh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_cosh._DNDarray__array.double(), comparison))

        # hyperbolic cosine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_cosh = ht.cosh(int32_tensor)
        self.assertIsInstance(int32_cosh, ht.DNDarray)
        self.assertEqual(int32_cosh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float32_cosh._DNDarray__array.double(), comparison))

        # hyperbolic cosine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_cosh = ht.cosh(int64_tensor)
        self.assertIsInstance(int64_cosh, ht.DNDarray)
        self.assertEqual(int64_cosh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_cosh._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.cosh([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.cosh("hello world")

    def test_sin(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).sin()

        # sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sin = float32_tensor.sin()
        self.assertIsInstance(float32_sin, ht.DNDarray)
        self.assertEqual(float32_sin.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_sin._DNDarray__array.double(), comparison))

        # sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sin = ht.sin(float64_tensor)
        self.assertIsInstance(float64_sin, ht.DNDarray)
        self.assertEqual(float64_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_sin._DNDarray__array.double(), comparison))

        # sine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sin = ht.sin(int32_tensor)
        self.assertIsInstance(int32_sin, ht.DNDarray)
        self.assertEqual(int32_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_sin._DNDarray__array.double(), comparison))

        # sine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sin = ht.sin(int64_tensor)
        self.assertIsInstance(int64_sin, ht.DNDarray)
        self.assertEqual(int64_sin.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_sin._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sin([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sin("hello world")

    def test_sinh(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).sinh()

        # hyperbolic sine of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_sinh = float32_tensor.sinh()
        self.assertIsInstance(float32_sinh, ht.DNDarray)
        self.assertEqual(float32_sinh.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_sinh._DNDarray__array.double(), comparison))

        # hyperbolic sine of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_sinh = ht.sinh(float64_tensor)
        self.assertIsInstance(float64_sinh, ht.DNDarray)
        self.assertEqual(float64_sinh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_sinh._DNDarray__array.double(), comparison))

        # hyperbolic sine of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_sinh = ht.sinh(int32_tensor)
        self.assertIsInstance(int32_sinh, ht.DNDarray)
        self.assertEqual(int32_sinh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_sinh._DNDarray__array.double(), comparison))

        # hyperbolic sine of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_sinh = ht.sinh(int64_tensor)
        self.assertIsInstance(int64_sinh, ht.DNDarray)
        self.assertEqual(int64_sinh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_sinh._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.sinh([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.sinh("hello world")

    def test_tan(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).tan()

        # tangent of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_tan = float32_tensor.tan()
        self.assertIsInstance(float32_tan, ht.DNDarray)
        self.assertEqual(float32_tan.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_tan._DNDarray__array.double(), comparison))

        # tangent of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_tan = ht.tan(float64_tensor)
        self.assertIsInstance(float64_tan, ht.DNDarray)
        self.assertEqual(float64_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_tan._DNDarray__array.double(), comparison))

        # tangent of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_tan = ht.tan(int32_tensor)
        self.assertIsInstance(int32_tan, ht.DNDarray)
        self.assertEqual(int32_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_tan._DNDarray__array.double(), comparison))

        # tangent of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_tan = ht.tan(int64_tensor)
        self.assertIsInstance(int64_tan, ht.DNDarray)
        self.assertEqual(int64_tan.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_tan._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.tan([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.tan("hello world")

    def test_tanh(self):
        # base elements
        elements = 30
        comparison = torch.arange(elements, dtype=torch.float64, device=device).tanh()

        # hyperbolic tangent of float32
        float32_tensor = ht.arange(elements, dtype=ht.float32)
        float32_tanh = float32_tensor.tanh()
        self.assertIsInstance(float32_tanh, ht.DNDarray)
        self.assertEqual(float32_tanh.dtype, ht.float32)
        self.assertTrue(torch.allclose(float32_tanh._DNDarray__array.double(), comparison))

        # hyperbolic tangent of float64
        float64_tensor = ht.arange(elements, dtype=ht.float64)
        float64_tanh = ht.tanh(float64_tensor)
        self.assertIsInstance(float64_tanh, ht.DNDarray)
        self.assertEqual(float64_tanh.dtype, ht.float64)
        self.assertTrue(torch.allclose(float64_tanh._DNDarray__array.double(), comparison))

        # hyperbolic tangent of ints, automatic conversion to intermediate floats
        int32_tensor = ht.arange(elements, dtype=ht.int32)
        int32_tanh = ht.tanh(int32_tensor)
        self.assertIsInstance(int32_tanh, ht.DNDarray)
        self.assertEqual(int32_tanh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int32_tanh._DNDarray__array.double(), comparison))

        # hyperbolic tangent of longs, automatic conversion to intermediate floats
        int64_tensor = ht.arange(elements, dtype=ht.int64)
        int64_tanh = ht.tanh(int64_tensor)
        self.assertIsInstance(int64_tanh, ht.DNDarray)
        self.assertEqual(int64_tanh.dtype, ht.float64)
        self.assertTrue(torch.allclose(int64_tanh._DNDarray__array.double(), comparison))

        # check exceptions
        with self.assertRaises(TypeError):
            ht.tanh([1, 2, 3])
        with self.assertRaises(TypeError):
            ht.tanh("hello world")
