import torch
import os
import unittest
import heat as ht
import numpy as np
import xarray as xr
from mpi4py import MPI

from heat.core.tests.test_suites.basic_test import TestCase

nprocs = MPI.COMM_WORLD.Get_size()


class TestHelpers(TestCase):
    def test_dim_name_idx_conversion(self):
        dims = ["x", "y", "z-axis", "time", None]
        for names in ["z-axis", ("time", "x"), ["x", "y"]]:
            idxs = ht.dxarray.dim_name_to_idx(dims, names)
            # check for correct types (str, tuple or list)
            self.assertTrue(
                type(idxs) == type(names) or (isinstance(names, str) and isinstance(idxs, int))
            )
            # check if dim_name_to_idx and dim_idx_to_name are inverse to each other
            names_back = ht.dxarray.dim_idx_to_name(dims, idxs)
            self.assertEqual(names_back, names)
        # check if TypeError is raised for wrong input types
        names = 3.14
        with self.assertRaises(TypeError):
            ht.dxarray.dim_name_to_idx(dims, names)
        with self.assertRaises(TypeError):
            ht.dxarray.dim_idx_to_name(dims, names)


class TestDXarray(TestCase):
    def test_constructor_and_attributes(self):
        m = 2
        n = 3 * nprocs
        k = 10
        ell = 2

        # test constructor in a case that should work and also test if all attributes of the DXarray are set correctly
        # here we include a dimension ("no_measurements") without coordinates and two dimensions ("x", "y") with physical instead of logical coordinates
        xy = ht.random.rand(m, n, split=1)
        t = ht.linspace(-1, 1, k, split=None)
        attrs_xy = {"units_xy": "meters"}
        xy_coords = ht.dxarray.DXarray(
            xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        data = ht.random.randn(m, n, k, ell, split=1)
        name = "mytestarray"
        attrs = {
            "units time": "seconds",
            "measured data": "something really random and meaningless",
        }
        dims = ["x", "y", "time", "no_measurements"]
        coords = {("x", "y"): xy_coords, "time": t}

        dxarray = ht.dxarray.DXarray(data, dims=dims, coords=coords, name=name, attrs=attrs)

        self.assertEqual(dxarray.name, name)
        self.assertEqual(dxarray.attrs, attrs)
        self.assertEqual(dxarray.dims, dims)
        self.assertEqual(dxarray.coords, coords)
        self.assertTrue(ht.allclose(dxarray.values, data))
        self.assertEqual(dxarray.device, data.device)
        self.assertEqual(dxarray.comm, data.comm)
        self.assertEqual(dxarray.dims_with_coords, ["x", "y", "time"])
        self.assertEqual(dxarray.dims_without_coords, ["no_measurements"])
        self.assertEqual(dxarray.split, "y")
        self.assertEqual(dxarray.balanced, True)

        # test print
        print(dxarray)

        # special case that dim names have to bet set automatically and that there are no coords at all
        dxarray = ht.dxarray.DXarray(data)
        dims = ["dim_0", "dim_1", "dim_2", "dim_3"]
        self.assertEqual(dxarray.dims, dims)
        self.assertEqual(dxarray.dims_with_coords, [])
        self.assertEqual(dxarray.dims_without_coords, dims)
        self.assertEqual(dxarray.split, "dim_1")
        self.assertEqual(dxarray.balanced, True)

        # test print
        print(dxarray)

    def test_sanity_checks(self):
        m = 2
        n = 3 * nprocs
        k = 5 * nprocs
        ell = 2

        # here comes the "correct" data
        xy = ht.random.rand(m, n, split=1)
        t = ht.linspace(-1, 1, k, split=None)
        attrs_xy = {"units_xy": "meters"}
        xy_coords = ht.dxarray.DXarray(
            xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        data = ht.random.randn(m, n, k, ell, split=1)
        name = "mytestarray"
        attrs = {
            "units time": "seconds",
            "measured data": "something really random and meaningless",
        }
        dims = ["x", "y", "time", "no_measurements"]
        coords = {("x", "y"): xy_coords, "time": t}

        # wrong data type for name
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(data, dims=dims, coords=coords, name=3.14, attrs=attrs)

        # wrong data type for attrs
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(data, dims=dims, coords=coords, name=name, attrs=3.14)

        # wrong data type for value
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(3.14, dims=dims, coords=coords, name=name, attrs=attrs)

        # wrong data type for dims
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(data, dims=3.14, coords=coords, name=name, attrs=attrs)

        # wrong data type for coords
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(data, dims=dims, coords=3.14, name=name, attrs=attrs)

        # length of dims and number of dimensions of value array do not match
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=["x", "y", "time"], coords=coords, name=name, attrs=attrs
            )

        # entries of dims are not unique
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=["x", "y", "x", "no_measurements"], coords=coords, name=name, attrs=attrs
            )

        # coordinate array for single dimension is not a DNDarray
        wrong_coords = {("x", "y"): xy_coords, "time": 3.14}
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # coordinate array for single dimension has wrong dimensionality
        wrong_coords = {("x", "y"): xy_coords, "time": ht.ones((k, 2))}
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # device of a coordinate array does not coincide with device of value array
        # TBD - how to test this?

        # communicator of a coordinate array does not coincide with communicator of value array
        # TBD - how to test this?

        # size of value array in a dimension does not coincide with size of coordinate array in this dimension
        wrong_coords = {("x", "y"): xy_coords, "time": ht.ones(nprocs * k + 1)}
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # value array is split along a dimension, but cooresponding coordinate array is not split along this dimension
        wrong_data = ht.resplit(data, 2)
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                wrong_data, dims=dims, coords=coords, name=name, attrs=attrs
            )

        # value array is not split along a dimension, but cooresponding coordinate array is split along this dimension
        wrong_coords = {("x", "y"): xy_coords, "time": ht.resplit(t, 0)}
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # coordinate array in the case of "physical coordinates" is not a DXarray
        wrong_coords = {("x", "y"): 3.14, "time": t}
        with self.assertRaises(TypeError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # dimension names in coordinate DXarray in the case of "physical coordinates" do not coincide with dimension names of value array
        wrong_coords_xy = ht.dxarray.DXarray(
            xy, dims=["xx", "yy"], attrs=attrs_xy, name="coordinates of space"
        )
        wrong_coords = {("x", "y"): wrong_coords_xy, "time": t}
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # size of values for physical coordinates does not coincide with size of the respective coordinate array
        wrong_xy = ht.random.rand(m + 1, n, split=1)
        wrong_coords_xy = ht.dxarray.DXarray(
            wrong_xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        wrong_coords = {("x", "y"): wrong_coords_xy, "time": t}
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )

        # communicator of coordinate array for physical coordinates does not coincide with communicator of value array
        # TBD - how to test this?

        # device of coordinate array for physical coordinates does not coincide with device of value array
        # TBD - how to test this?

        # coordinate array for physical coordinates is not split along the split dimension of the value array (two cases)
        wrong_data = ht.random.randn(m, n, k, ell)
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                wrong_data, dims=dims, coords=coords, name=name, attrs=attrs
            )
        wrong_xy = ht.random.rand(m, n)
        wrong_coords_xy = ht.dxarray.DXarray(
            wrong_xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        wrong_coords = {("x", "y"): wrong_coords_xy, "time": t}
        with self.assertRaises(ValueError):
            dxarray = ht.dxarray.DXarray(
                data, dims=dims, coords=wrong_coords, name=name, attrs=attrs
            )
            dxarray *= 1

    def test_balanced_and_balancing(self):
        m = 2
        n = 5 * nprocs
        k = 2
        ell = 2

        # create a highly unbalanced array for the values but not for the coordinates
        xy = ht.random.rand(m, n, split=1)
        xy = xy[:, 4:]
        xy.balance_()
        t = ht.linspace(-1, 1, k, split=None)
        attrs_xy = {"units_xy": "meters"}
        xy_coords = ht.dxarray.DXarray(
            xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        data = ht.random.randn(m, n, k, ell, split=1)
        data = data[:, 4:, :, :]
        name = "mytestarray"
        attrs = {
            "units time": "seconds",
            "measured data": "something really random and meaningless",
        }
        dims = ["x", "y", "time", "no_measurements"]
        coords = {("x", "y"): xy_coords, "time": t}

        dxarray = ht.dxarray.DXarray(data, dims=dims, coords=coords, name=name, attrs=attrs)

        # balancedness-status is first unknown, then known as false (if explicitly checked) and finally known as false after this check
        self.assertEqual(dxarray.balanced, None)
        self.assertEqual(dxarray.is_balanced(), False)
        self.assertEqual(dxarray.balanced, False)

        # rebalancing should work
        dxarray.balance_()
        self.assertEqual(dxarray.balanced, True)
        self.assertEqual(dxarray.is_balanced(force_check=True), True)

        # rebalanced array should be equal to original one
        self.assertTrue(ht.allclose(dxarray.values, data))
        self.assertEqual(dxarray.dims, dims)
        self.assertEqual(dxarray.dims_with_coords, ["x", "y", "time"])
        self.assertEqual(dxarray.dims_without_coords, ["no_measurements"])
        self.assertEqual(dxarray.name, name)
        self.assertEqual(dxarray.attrs, attrs)
        # TBD: check for equality of coordinate arrays

    def test_resplit_(self):
        m = 2 * nprocs
        n = 3 * nprocs
        k = 5 * nprocs
        ell = 2

        xy = ht.random.rand(m, n, split=1)
        t = ht.linspace(-1, 1, k, split=None)
        attrs_xy = {"units_xy": "meters"}
        xy_coords = ht.dxarray.DXarray(
            xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        data = ht.random.randn(m, n, k, ell, split=1)
        name = "mytestarray"
        attrs = {
            "units time": "seconds",
            "measured data": "something really random and meaningless",
        }
        dims = ["x", "y", "time", "no_measurements"]
        coords = {("x", "y"): xy_coords, "time": t}

        dxarray = ht.dxarray.DXarray(data, dims=dims, coords=coords, name=name, attrs=attrs)
        for newsplit in ["x", "time", None, "y"]:
            dxarray.resplit_(newsplit)
            self.assertEqual(dxarray.split, newsplit)
            self.assertTrue(ht.allclose(dxarray.values, data))
            self.assertEqual(dxarray.dims, dims)
            self.assertEqual(dxarray.dims_with_coords, ["x", "y", "time"])
            self.assertEqual(dxarray.dims_without_coords, ["no_measurements"])
            self.assertEqual(dxarray.name, name)
            self.assertEqual(dxarray.attrs, attrs)
            # TBD: check for equality of coordinate arrays

    def test_to_and_from_xarray(self):
        m = 2
        n = 3 * nprocs
        k = 10
        ell = 2

        # test constructor in a case that should work and also test if all attributes of the DXarray are set correctly
        # here we include a dimension ("no_measurements") without coordinates and two dimensions ("x", "y") with physical instead of logical coordinates
        xy = ht.random.rand(m, n, split=1)
        t = ht.linspace(-1, 1, k, split=None)
        attrs_xy = {"units_xy": "meters"}
        xy_coords = ht.dxarray.DXarray(
            xy, dims=["x", "y"], attrs=attrs_xy, name="coordinates of space"
        )
        data = ht.random.randn(m, n, k, ell, split=1)
        name = "mytestarray"
        attrs = {
            "units time": "seconds",
            "measured data": "something really random and meaningless",
        }
        dims = ["x", "y", "time", "no_measurements"]
        coords = {("x", "y"): xy_coords, "time": t}

        dxarray = ht.dxarray.DXarray(data, dims=dims, coords=coords, name=name, attrs=attrs)

        xarray = dxarray.xarray()
        print(xarray)
        # TBD convert back and check for equality (or the other way round?)
        # dxarray_from_xarray = ht.dxarray.from_xarray(xarray,split=dxarray.split,device=dxarray.device)
