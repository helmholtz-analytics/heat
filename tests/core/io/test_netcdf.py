"""Tests for reading and writing NetCDF4 files."""

import fnmatch
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import torch

import heat as ht

from ._base import IOTestCase

if ht.io.supports("netcdf"):
    import netCDF4 as nc


@unittest.skipUnless(ht.io.supports("netcdf"), "Requires netcdf")
class TestNetCDF(IOTestCase):
    """Tests for reading and writing NetCDF4 files."""

    def test_load_netcdf(self):
        # netcdf support is optional

        # default parameters
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris.larray.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris.larray).all())

        # positive split axis
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, split=0)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        lshape = iris.lshape
        self.assertLessEqual(lshape[0], self.IRIS.shape[0])
        self.assertEqual(lshape[1], self.IRIS.shape[1])

        # negative split axis
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, split=-1)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        lshape = iris.lshape
        self.assertEqual(lshape[0], self.IRIS.shape[0])
        self.assertLessEqual(lshape[1], self.IRIS.shape[1])

        # different data type
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, dtype=ht.int8)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.int8)
        self.assertEqual(iris.larray.dtype, torch.int8)

    def test_load_netcdf_exception(self):
        # netcdf support is optional

        # improper argument types
        with self.assertRaises(TypeError):
            ht.load_netcdf(1, "data")
        with self.assertRaises(TypeError):
            ht.load_netcdf("iris.nc", variable=1)
        with self.assertRaises(TypeError):
            ht.load_netcdf("iris.nc", variable="data", split=1.0)

        # file or variable does not exist
        with self.assertRaises(IOError):
            ht.load_netcdf("foo.nc", variable="data")
        with self.assertRaises(IOError):
            ht.load_netcdf("iris.nc", variable="foo")

    def test_save_netcdf(self):
        # netcdf support is optional

        # local unsplit data
        local_data = ht.arange(100)
        ht.save_netcdf(local_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if local_data.comm.rank == 0:
            with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:],
                    dtype=torch.int32,
                    device=self.device.torch_device,
                )
            self.assertTrue((local_data.larray == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0)
        ht.save_netcdf(split_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if split_data.comm.rank == 0:
            with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:],
                    dtype=torch.int32,
                    device=self.device.torch_device,
                )
            self.assertTrue((local_data.larray == comparison).all())

    def test_save_netcdf_exception(self):
        # netcdf support is optional

        # dummy data
        data = ht.arange(1)

        with self.assertRaises(TypeError):
            ht.save_netcdf(1, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, 1, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, self.NETCDF_OUT_PATH, 1)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, dimension_names=1)
        with self.assertRaises(ValueError):
            ht.save_netcdf(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, dimension_names=["a", "b"])

    # def test_remove_folder(self):
    # ht.MPI_WORLD.Barrier()
    # try:
    #     os.rmdir(os.getcwd() + '/tmp/')
    # except OSError:
    #     pass
