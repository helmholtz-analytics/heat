"""Tests for the generic ht.load / ht.save entry points."""

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

if ht.io.supports("hdf5"):
    import h5py

if ht.io.supports("netcdf"):
    import netCDF4 as nc


class TestDispatch(IOTestCase):
    """Tests for the generic ht.load / ht.save entry points."""

    def test_load(self):
        # HDF5
        if ht.io.supports("hdf5"):
            iris = ht.load(self.HDF5_PATH, dataset="data", dtype=ht.float32)
            self.assertIsInstance(iris, ht.DNDarray)
            # shape invariant
            self.assertEqual(iris.shape, self.IRIS.shape)
            self.assertEqual(iris.larray.shape, self.IRIS.shape)
            # data type
            self.assertEqual(iris.dtype, ht.float32)
            self.assertEqual(iris.larray.dtype, torch.float32)
            # content
            self.assertTrue((self.IRIS == iris.larray).all())

        else:
            with self.assertRaises(RuntimeError):
                _ = ht.load(self.HDF5_PATH, dataset=self.HDF5_DATASET)

        # netCDF
        if ht.io.supports("netcdf"):
            iris = ht.load(self.NETCDF_PATH, variable=self.NETCDF_VARIABLE)
            self.assertIsInstance(iris, ht.DNDarray)
            # shape invariant
            self.assertEqual(iris.shape, self.IRIS.shape)
            self.assertEqual(iris.larray.shape, self.IRIS.shape)
            # data type
            self.assertEqual(iris.dtype, ht.float32)
            self.assertEqual(iris.larray.dtype, torch.float32)
            # content
            self.assertTrue((self.IRIS == iris.larray).all())
        else:
            with self.assertRaises(RuntimeError):
                _ = ht.load(self.NETCDF_PATH, variable=self.NETCDF_VARIABLE)

    def test_load_exception(self):
        # correct extension, file does not exist
        if ht.io.supports("hdf5"):
            with self.assertRaises(IOError):
                ht.load("foo.h5", "data")
        else:
            with self.assertRaises(RuntimeError):
                ht.load("foo.h5", "data")

        if ht.io.supports("netcdf"):
            with self.assertRaises(IOError):
                ht.load("foo.nc", "data")
        else:
            with self.assertRaises(RuntimeError):
                ht.load("foo.nc", "data")

        # unknown file extension
        with self.assertRaises(ValueError):
            ht.load(os.path.join(os.getcwd(), "heat/datasets/iris.json"), "data")
        with self.assertRaises(ValueError):
            ht.load("iris", "data")

    # catch-all save

    def test_save(self):
        if ht.io.supports("hdf5"):
            # local range
            local_range = ht.arange(100)
            local_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET, dtype=local_range.dtype.char())
            if local_range.comm.rank == 0:
                with h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((local_range.larray == comparison).all())

            # split range
            split_range = ht.arange(100, split=0)
            split_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET, dtype=split_range.dtype.char())
            if split_range.comm.rank == 0:
                with h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((local_range.larray == comparison).all())

        if ht.io.supports("netcdf"):
            # local range
            local_range = ht.arange(100)
            local_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if local_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((local_range.larray == comparison).all())

            # split range
            split_range = ht.arange(100, split=0)
            split_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((local_range.larray == comparison).all())

            # naming dimensions: string
            local_range = ht.arange(100, device=self.device)
            local_range.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, dimension_names=self.NETCDF_DIMENSION
            )
            if local_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = handle[self.NETCDF_VARIABLE].dimensions
                self.assertTrue(self.NETCDF_DIMENSION in comparison)

            # naming dimensions: tuple
            local_range = ht.arange(100, device=self.device)
            local_range.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, dimension_names=(self.NETCDF_DIMENSION,)
            )
            if local_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = handle[self.NETCDF_VARIABLE].dimensions
                self.assertTrue(self.NETCDF_DIMENSION in comparison)

            # appending unlimited variable
            split_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, is_unlimited=True)
            ht.MPI_WORLD.Barrier()
            split_range.save(
                self.NETCDF_OUT_PATH,
                self.NETCDF_VARIABLE,
                mode="r+",
                file_slices=slice(split_range.size, None, None),
                # debug=True,
            )
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue(
                    (ht.concatenate((local_range, local_range)).larray == comparison).all()
                )

            # indexing netcdf file: single index
            ht.MPI_WORLD.Barrier()
            zeros = ht.zeros((20, 1, 20, 2), device=self.device)
            zeros.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w")
            ones = ht.ones(20, device=self.device)
            indices = (-1, 0, slice(None), 1)
            ones.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+", file_slices=indices)
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((ones.larray == comparison).all())

            # indexing netcdf file: multiple indices
            ht.MPI_WORLD.Barrier()
            small_range_split = ht.arange(10, split=0, device=self.device)
            small_range = ht.arange(10, device=self.device)
            indices = [[0, 9, 5, 2, 1, 3, 7, 4, 8, 6]]
            small_range_split.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w", file_slices=indices
            )
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((small_range.larray == comparison).all())

            # slicing netcdf file
            sslice = slice(7, 2, -1)
            range_five_split = ht.arange(5, split=0, device=self.device)
            range_five = ht.arange(5, device=self.device)
            range_five_split.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+", file_slices=sslice
            )
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][sslice],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((range_five.larray == comparison).all())

            # indexing netcdf file: broadcasting array
            zeros = ht.zeros((2, 1, 1, 4), device=self.device)
            zeros.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w")
            ones = ht.ones((4), split=0, device=self.device)
            ones_nosplit = ht.ones((4), split=None, device=self.device)
            indices = (0, slice(None), slice(None))
            ones.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+", file_slices=indices)
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((ones_nosplit.larray == comparison).all())

            # indexing netcdf file: broadcasting var
            ht.MPI_WORLD.Barrier()
            zeros = ht.zeros((2, 2), device=self.device)
            zeros.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w")
            ones = ht.ones((1, 2, 1), split=0, device=self.device)
            ones_nosplit = ht.ones((1, 2, 1), device=self.device)
            indices = (0,)
            ones.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+", file_slices=indices)
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((ones_nosplit.larray == comparison).all())

            # indexing netcdf file: broadcasting ones
            ht.MPI_WORLD.Barrier()
            zeros = ht.zeros((1, 1, 1, 1), device=self.device)
            zeros.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w")
            ones = ht.ones((1, 1), device=self.device)
            ones.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+")
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((ones.larray == comparison).all())

            # different split and dtype
            ht.MPI_WORLD.Barrier()
            zeros = ht.zeros((2, 2), split=1, dtype=ht.int32, device=self.device)
            zeros_nosplit = ht.zeros((2, 2), dtype=ht.int32, device=self.device)
            zeros.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w")
            if split_range.comm.rank == 0:
                with nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((zeros_nosplit.larray == comparison).all())

    def test_save_exception(self):
        data = ht.arange(1)

        if ht.io.supports("hdf5"):
            with self.assertRaises(TypeError):
                ht.save(1, self.HDF5_OUT_PATH, self.HDF5_DATASET)
            with self.assertRaises(TypeError):
                ht.save(data, 1, self.HDF5_DATASET)
            with self.assertRaises(TypeError):
                ht.save(data, self.HDF5_OUT_PATH, 1)
        else:
            with self.assertRaises(RuntimeError):
                ht.save(data, self.HDF5_OUT_PATH, self.HDF5_DATASET)

        if ht.io.supports("netcdf"):
            with self.assertRaises(TypeError):
                ht.save(1, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            with self.assertRaises(TypeError):
                ht.save(data, 1, self.NETCDF_VARIABLE)
            with self.assertRaises(TypeError):
                ht.save(data, self.NETCDF_OUT_PATH, 1)
            with self.assertRaises(ValueError):
                ht.save(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r")
            with self.assertRaises((ValueError, IndexError)):
                ht.save(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
                ht.save(
                    ht.arange(2, split=0),
                    self.NETCDF_OUT_PATH,
                    self.NETCDF_VARIABLE,
                    file_slices=slice(None),
                    mode="a",
                )
            with self.assertRaises((ValueError, IndexError)):
                ht.save(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
                ht.save(
                    ht.arange(2, split=None),
                    self.NETCDF_OUT_PATH,
                    self.NETCDF_VARIABLE,
                    file_slices=slice(None),
                    mode="a",
                )
        else:
            with self.assertRaises(RuntimeError):
                ht.save(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)

        with self.assertRaises(ValueError):
            ht.save(1, "data.dat")
