"""Shared fixtures for the ``heat.core.io`` tests."""

import os
import shutil
from pathlib import Path

import numpy as np
import torch

import heat as ht
from heat.testing.basic_test import TestCase


class IOTestCase(TestCase):
    """Holds the dataset paths and the temporary-file cleanup shared by the io tests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        pwd = os.getcwd()
        cls.HDF5_PATH = str(Path(ht.__file__).parent / "datasets" / "iris.h5")
        cls.HDF5_OUT_PATH = pwd + "/test.h5"
        cls.HDF5_DATASET = "data"

        cls.NETCDF_PATH = str(Path(ht.__file__).parent / "datasets" / "iris.nc")
        cls.NETCDF_OUT_PATH = pwd + "/test.nc"
        cls.NETCDF_VARIABLE = "data"
        cls.NETCDF_DIMENSION = "data"

        # load comparison data from csv
        cls.CSV_PATH = str(Path(ht.__file__).parent / "datasets" / "iris.csv")
        cls.CSV_OUT_PATH = pwd + "/test.csv"
        cls.IRIS = (
            torch.from_numpy(np.loadtxt(cls.CSV_PATH, delimiter=";"))
            .float()
            .to(cls.device.torch_device)
        )

        cls.ZARR_SHAPE = (100, 100)
        cls.ZARR_OUT_PATH = pwd + "/zarr_test_out.zarr"
        cls.ZARR_IN_PATH = pwd + "/zarr_test_in.zarr"
        cls.ZARR_TEMP_PATH = pwd + "/zarr_temp.zarr"
        cls.ZARR_NESTED_PATH = pwd + "/zarr_test_nested.zarr"

        # device-aware dtypes
        testing_types = [ht.int32, ht.int64, ht.float32]
        if not cls.is_mps:
            testing_types.append(ht.float64)
        cls.testing_types = testing_types

        cls.HDF5_MULTIPLE_FOLDER = pwd + "/hdf5_data"
        cls.HDF5_MULTIPLE_FILE_PREFIX = "data_"
        cls.HDF5_MULTIPLE_FILE_ENDING = ".h5"
        cls.HDF5_MULTIPLE_DATASET = "data"

    def tearDown(self):
        # synchronize all processes
        ht.MPI_WORLD.Barrier()

        # clean up of temporary files
        if ht.io.supports("hdf5"):
            try:
                os.remove(self.HDF5_OUT_PATH)
            except FileNotFoundError:
                pass

            try:
                shutil.rmtree(self.HDF5_MULTIPLE_FOLDER)
            except FileNotFoundError:
                pass

        if ht.io.supports("netcdf"):
            try:
                os.remove(self.NETCDF_OUT_PATH)
            except FileNotFoundError:
                pass

        if ht.io.supports("zarr"):
            if ht.MPI_WORLD.rank == 0:
                for file in [
                    self.ZARR_TEMP_PATH,
                    self.ZARR_IN_PATH,
                    self.ZARR_OUT_PATH,
                    self.ZARR_NESTED_PATH,
                ]:
                    try:
                        shutil.rmtree(file)
                    except FileNotFoundError:
                        pass

        ht.MPI_WORLD.Barrier()
