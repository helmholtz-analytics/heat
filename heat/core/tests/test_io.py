import numpy as np
import os
import torch
import tempfile
import time
import random
import shutil
import fnmatch

import heat as ht
from .test_suites.basic_test import TestCase


class TestIO(TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestIO, cls).setUpClass()
        pwd = os.getcwd()
        cls.HDF5_PATH = os.path.join(os.getcwd(), "heat/datasets/iris.h5")
        cls.HDF5_OUT_PATH = pwd + "/test.h5"
        cls.HDF5_DATASET = "data"

        cls.NETCDF_PATH = os.path.join(os.getcwd(), "heat/datasets/iris.nc")
        cls.NETCDF_OUT_PATH = pwd + "/test.nc"
        cls.NETCDF_VARIABLE = "data"
        cls.NETCDF_DIMENSION = "data"

        # load comparison data from csv
        cls.CSV_PATH = os.path.join(os.getcwd(), "heat/datasets/iris.csv")
        cls.CSV_OUT_PATH = pwd + "/test.csv"
        cls.IRIS = (
            torch.from_numpy(np.loadtxt(cls.CSV_PATH, delimiter=";"))
            .float()
            .to(cls.device.torch_device)
        )

    def tearDown(self):
        # synchronize all nodes
        ht.MPI_WORLD.Barrier()

        # clean up of temporary files
        if ht.io.supports_hdf5():
            try:
                os.remove(self.HDF5_OUT_PATH)
            except FileNotFoundError:
                pass

        if ht.io.supports_netcdf():
            try:
                os.remove(self.NETCDF_OUT_PATH)
            except FileNotFoundError:
                pass
        # if ht.MPI_WORLD.rank == 0:

        # synchronize all nodes
        ht.MPI_WORLD.Barrier()

    # catch-all loading
    def test_load(self):
        # HDF5
        if ht.io.supports_hdf5():
            iris = ht.load(self.HDF5_PATH, dataset="data")
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
        if ht.io.supports_netcdf():
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

    def test_load_csv(self):
        csv_file_length = 150
        csv_file_cols = 4
        first_value = torch.tensor(
            [5.1, 3.5, 1.4, 0.2], dtype=torch.float32, device=self.device.torch_device
        )
        tenth_value = torch.tensor(
            [4.9, 3.1, 1.5, 0.1], dtype=torch.float32, device=self.device.torch_device
        )

        a = ht.load_csv(self.CSV_PATH, sep=";")
        self.assertEqual(len(a), csv_file_length)
        self.assertEqual(a.shape, (csv_file_length, csv_file_cols))
        self.assertTrue(torch.equal(a.larray[0], first_value))
        self.assertTrue(torch.equal(a.larray[9], tenth_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", split=0)
        rank = a.comm.Get_rank()
        expected_gshape = (csv_file_length, csv_file_cols)
        self.assertEqual(a.gshape, expected_gshape)

        counts, _, _ = a.comm.counts_displs_shape(expected_gshape, 0)
        expected_lshape = (counts[rank], csv_file_cols)
        self.assertEqual(a.lshape, expected_lshape)

        if rank == 0:
            self.assertTrue(torch.equal(a.larray[0], first_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", header_lines=9, dtype=ht.float32, split=0)
        expected_gshape = (csv_file_length - 9, csv_file_cols)
        counts, _, _ = a.comm.counts_displs_shape(expected_gshape, 0)
        expected_lshape = (counts[rank], csv_file_cols)

        self.assertEqual(a.gshape, expected_gshape)
        self.assertEqual(a.lshape, expected_lshape)
        self.assertEqual(a.dtype, ht.float32)
        if rank == 0:
            self.assertTrue(torch.equal(a.larray[0], tenth_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", split=1)
        self.assertEqual(a.shape, (csv_file_length, csv_file_cols))
        self.assertEqual(a.lshape[0], csv_file_length)

        a = ht.load_csv(self.CSV_PATH, sep=";", split=0)
        b = ht.load(self.CSV_PATH, sep=";", split=0)
        self.assertTrue(ht.equal(a, b))

        # Test for csv where header is longer then the first process`s share of lines
        a = ht.load_csv(self.CSV_PATH, sep=";", header_lines=100, split=0)
        self.assertEqual(a.shape, (50, 4))

        with self.assertRaises(TypeError):
            ht.load_csv(12314)
        with self.assertRaises(TypeError):
            ht.load_csv(self.CSV_PATH, sep=11)
        with self.assertRaises(TypeError):
            ht.load_csv(self.CSV_PATH, header_lines="3", sep=";", split=0)

    def test_save_csv(self):
        for rnd_type in [
            (ht.random.randint, ht.types.int32),
            (ht.random.randint, ht.types.int64),
            (ht.random.rand, ht.types.float32),
            (ht.random.rand, ht.types.float64),
        ]:
            for separator in [",", ";", "|"]:
                for split in [None, 0, 1]:
                    for headers in [None, ["# This", "# is a", "# test."]]:
                        for shape in [(1, 1), (10, 10), (20, 1), (1, 20), (25, 4), (4, 25)]:
                            if rnd_type[0] == ht.random.randint:
                                data = rnd_type[0](
                                    -1000, 1000, size=shape, dtype=rnd_type[1], split=split
                                )
                            else:
                                data = rnd_type[0](
                                    shape[0],
                                    shape[1],
                                    split=split,
                                    dtype=rnd_type[1],
                                )

                            if data.comm.rank == 0:
                                tmpfile = tempfile.NamedTemporaryFile(
                                    prefix="test_io_", suffix=".csv", delete=False
                                )
                                tmpfile.close()
                                filename = tmpfile.name
                            else:
                                filename = None
                            filename = data.comm.handle.bcast(filename, root=0)

                            data.save(
                                filename,
                                header_lines=headers,
                                sep=separator,
                            )
                            comparison = ht.load_csv(
                                filename,
                                # split=split,
                                header_lines=0 if headers is None else len(headers),
                                sep=separator,
                            ).reshape(shape)
                            resid = data - comparison
                            self.assertTrue(
                                ht.max(resid).item() < 0.00001 and ht.min(resid).item() > -0.00001
                            )
                            data.comm.handle.Barrier()
                            if data.comm.rank == 0:
                                os.unlink(filename)

        # Test vector
        data = ht.random.randint(0, 100, size=(150,))
        if data.comm.rank == 0:
            tmpfile = tempfile.NamedTemporaryFile(prefix="test_io_", suffix=".csv", delete=False)
            tmpfile.close()
            filename = tmpfile.name
        else:
            filename = None
        filename = data.comm.handle.bcast(filename, root=0)
        data.save(filename)
        comparison = ht.load(filename).reshape((150,))
        self.assertTrue((data == comparison).all())
        data.comm.handle.Barrier()
        if data.comm.rank == 0:
            os.unlink(filename)

        # Test 0 matrix
        data = ht.zeros((10, 10))
        if data.comm.rank == 0:
            tmpfile = tempfile.NamedTemporaryFile(prefix="test_io_", suffix=".csv", delete=False)
            tmpfile.close()
            filename = tmpfile.name
        else:
            filename = None
        filename = data.comm.handle.bcast(filename, root=0)
        data.save(filename)
        comparison = ht.load(filename)
        self.assertTrue((data == comparison).all())
        data.comm.handle.Barrier()
        if data.comm.rank == 0:
            os.unlink(filename)

        # Test negative float values
        data = ht.random.rand(100, 100)
        data = data - 500
        if data.comm.rank == 0:
            tmpfile = tempfile.NamedTemporaryFile(prefix="test_io_", suffix=".csv", delete=False)
            tmpfile.close()
            filename = tmpfile.name
        else:
            filename = None
        filename = data.comm.handle.bcast(filename, root=0)
        data.save(filename)
        comparison = ht.load(filename)
        self.assertTrue((data == comparison).all())
        data.comm.handle.Barrier()
        if data.comm.rank == 0:
            os.unlink(filename)

    def test_load_exception(self):
        # correct extension, file does not exist
        if ht.io.supports_hdf5():
            with self.assertRaises(IOError):
                ht.load("foo.h5", "data")
        else:
            with self.assertRaises(RuntimeError):
                ht.load("foo.h5", "data")

        if ht.io.supports_netcdf():
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
        if ht.io.supports_hdf5():
            # local range
            local_range = ht.arange(100)
            local_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET, dtype=local_range.dtype.char())
            if local_range.comm.rank == 0:
                with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
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
                with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((local_range.larray == comparison).all())

        if ht.io.supports_netcdf():
            # local range
            local_range = ht.arange(100)
            local_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if local_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = handle[self.NETCDF_VARIABLE].dimensions
                self.assertTrue(self.NETCDF_DIMENSION in comparison)

            # naming dimensions: tuple
            local_range = ht.arange(100, device=self.device)
            local_range.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, dimension_names=(self.NETCDF_DIMENSION,)
            )
            if local_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:],
                        dtype=torch.int32,
                        device=self.device.torch_device,
                    )
                self.assertTrue((zeros_nosplit.larray == comparison).all())

    def test_save_exception(self):
        data = ht.arange(1)

        if ht.io.supports_hdf5():
            with self.assertRaises(TypeError):
                ht.save(1, self.HDF5_OUT_PATH, self.HDF5_DATASET)
            with self.assertRaises(TypeError):
                ht.save(data, 1, self.HDF5_DATASET)
            with self.assertRaises(TypeError):
                ht.save(data, self.HDF5_OUT_PATH, 1)
        else:
            with self.assertRaises(RuntimeError):
                ht.save(data, self.HDF5_OUT_PATH, self.HDF5_DATASET)

        if ht.io.supports_netcdf():
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

    def test_load_hdf5(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            self.skipTest("Requires HDF5")

        # default parameters
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris.larray.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris.larray).all())

        # cropped load
        iris_cropped = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=0, load_fraction=0.5)
        self.assertEqual(iris_cropped.shape[0], iris.shape[0] // 2)

        # positive split axis
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=0)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        lshape = iris.lshape
        self.assertLessEqual(lshape[0], self.IRIS.shape[0])
        self.assertEqual(lshape[1], self.IRIS.shape[1])

        # negative split axis
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=-1)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        lshape = iris.lshape
        self.assertEqual(lshape[0], self.IRIS.shape[0])
        self.assertLessEqual(lshape[1], self.IRIS.shape[1])

        # different data type
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, dtype=ht.int8)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.int8)
        self.assertEqual(iris.larray.dtype, torch.int8)

    def test_load_hdf5_exception(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            self.skipTest("Requires HDF5")

        # improper argument types
        with self.assertRaises(TypeError):
            ht.load_hdf5(1, "data")
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", 1)
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", dataset="data", split=1.0)
        with self.assertRaises(TypeError):
            ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, load_fraction="a")
        with self.assertRaises(ValueError):
            ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, load_fraction=0.0, split=0)

        # file or dataset does not exist
        with self.assertRaises(IOError):
            ht.load_hdf5("foo.h5", dataset="data")
        with self.assertRaises(IOError):
            ht.load_hdf5("iris.h5", dataset="foo")

    def test_save_hdf5(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # local unsplit data
        local_data = ht.arange(100)
        ht.save_hdf5(
            local_data, self.HDF5_OUT_PATH, self.HDF5_DATASET, dtype=local_data.dtype.char()
        )
        if local_data.comm.rank == 0:
            with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=self.device.torch_device
                )
            self.assertTrue((local_data.larray == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0)
        ht.save_hdf5(
            split_data, self.HDF5_OUT_PATH, self.HDF5_DATASET, dtype=split_data.dtype.char()
        )
        if split_data.comm.rank == 0:
            with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=self.device.torch_device
                )
            self.assertTrue((local_data.larray == comparison).all())

    def test_save_hdf5_exception(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            self.skipTest("Requires HDF5")

        # dummy data
        data = ht.arange(1)

        with self.assertRaises(TypeError):
            ht.save_hdf5(1, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        with self.assertRaises(TypeError):
            ht.save_hdf5(data, 1, self.HDF5_DATASET)
        with self.assertRaises(TypeError):
            ht.save_hdf5(data, self.HDF5_OUT_PATH, 1)

    def test_load_netcdf(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            self.skipTest("Requires NetCDF")

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
        if not ht.io.supports_netcdf():
            self.skipTest("Requires NetCDF")

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
        if not ht.io.supports_netcdf():
            self.skipTest("Requires NetCDF")

        # local unsplit data
        local_data = ht.arange(100)
        ht.save_netcdf(local_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if local_data.comm.rank == 0:
            with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
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
            with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:],
                    dtype=torch.int32,
                    device=self.device.torch_device,
                )
            self.assertTrue((local_data.larray == comparison).all())

    def test_save_netcdf_exception(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            self.skipTest("Requires NetCDF")

        # dummy data
        data = ht.arange(1)

        with self.assertRaises(TypeError):
            ht.save_netcdf(1, self.NETCDF_PATH, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, 1, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, self.NETCDF_PATH, 1)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, self.NETCDF_PATH, self.NETCDF_VARIABLE, dimension_names=1)
        with self.assertRaises(ValueError):
            ht.save_netcdf(data, self.NETCDF_PATH, self.NETCDF_VARIABLE, dimension_names=["a", "b"])

    # def test_remove_folder(self):
    # ht.MPI_WORLD.Barrier()
    # try:
    #     os.rmdir(os.getcwd() + '/tmp/')
    # except OSError:
    #     pass

    def test_load_npy_int(self):
        # testing for int arrays
        if ht.MPI_WORLD.rank == 0:
            crea_array = []
            for i in range(0, ht.MPI_WORLD.size * 5):
                x = np.random.randint(1000, size=(random.randint(0, 30), 6, 11))
                np.save(os.path.join(os.getcwd(), "heat/datasets", "int_data") + str(i), x)
                crea_array.append(x)
            int_array = np.concatenate(crea_array)
        ht.MPI_WORLD.Barrier()
        load_array = ht.load_npy_from_path(
            os.path.join(os.getcwd(), "heat/datasets"), dtype=ht.int32, split=0
        )
        load_array_npy = load_array.numpy()

        self.assertIsInstance(load_array, ht.DNDarray)
        self.assertEqual(load_array.dtype, ht.int32)
        if ht.MPI_WORLD.rank == 0:
            self.assertTrue((load_array_npy == int_array).all)
            for file in os.listdir(os.path.join(os.getcwd(), "heat/datasets")):
                if fnmatch.fnmatch(file, "*.npy"):
                    os.remove(os.path.join(os.getcwd(), "heat/datasets", file))

    def test_load_npy_float(self):
        # testing for float arrays and split dimension other than 0
        if ht.MPI_WORLD.rank == 0:
            crea_array = []
            for i in range(0, ht.MPI_WORLD.size * 5 + 1):
                x = np.random.rand(2, random.randint(1, 10), 11)
                np.save(os.path.join(os.getcwd(), "heat/datasets", "float_data") + str(i), x)
                crea_array.append(x)
            float_array = np.concatenate(crea_array, 1)
        ht.MPI_WORLD.Barrier()

        load_array = ht.load_npy_from_path(
            os.path.join(os.getcwd(), "heat/datasets"), dtype=ht.float64, split=1
        )
        load_array_npy = load_array.numpy()
        self.assertIsInstance(load_array, ht.DNDarray)
        self.assertEqual(load_array.dtype, ht.float64)
        if ht.MPI_WORLD.rank == 0:
            self.assertTrue((load_array_npy == float_array).all)
            for file in os.listdir(os.path.join(os.getcwd(), "heat/datasets")):
                if fnmatch.fnmatch(file, "*.npy"):
                    os.remove(os.path.join(os.getcwd(), "heat/datasets", file))

    def test_load_npy_exception(self):
        with self.assertRaises(TypeError):
            ht.load_npy_from_path(path=1, split=0)
        with self.assertRaises(TypeError):
            ht.load_npy_from_path("heat/datasets", split="ABC")
        with self.assertRaises(ValueError):
            ht.load_npy_from_path(path="heat", dtype=ht.int64, split=0)
        if ht.MPI_WORLD.size > 1:
            if ht.MPI_WORLD.rank == 0:
                x = np.random.rand(2, random.randint(1, 10), 11)
                np.save(os.path.join(os.getcwd(), "heat/datasets", "float_data"), x)
            ht.MPI_WORLD.Barrier()
            with self.assertRaises(RuntimeError):
                ht.load_npy_from_path("heat/datasets", dtype=ht.int64, split=0)
            ht.MPI_WORLD.Barrier()
            if ht.MPI_WORLD.rank == 0:
                os.remove(os.path.join(os.getcwd(), "heat/datasets", "float_data.npy"))

    def test_load_multiple_csv(self):
        if not ht.io.supports_pandas():
            self.skipTest("Requires pandas")

        import pandas as pd

        csv_path = os.path.join(os.getcwd(), "heat/datasets/csv_tests")
        if ht.MPI_WORLD.rank == 0:
            nplist = []
            npdroplist = []
            os.mkdir(csv_path)
            for i in range(0, ht.MPI_WORLD.size * 5 + 1):
                a = np.random.randint(100, size=(5))
                b = np.random.randint(100, size=(5))
                c = np.random.randint(100, size=(5))

                data = {"A": a, "B": b, "C": c}
                data2 = {"B": b, "C": c}
                df = pd.DataFrame(data)  # noqa F821
                df2 = pd.DataFrame(data2)  # noqa F821
                nplist.append(df.to_numpy())
                npdroplist.append(df2.to_numpy())
                df.to_csv((os.path.join(csv_path, f"csv_test_{i}.csv")), index=False)

            nparray = np.concatenate(nplist)
            npdroparray = np.concatenate(npdroplist)
        ht.MPI_WORLD.Barrier()

        def delete_first_col(dataf):
            dataf.drop(dataf.columns[0], axis=1, inplace=True)
            return dataf

        load_array = ht.load_csv_from_folder(csv_path, dtype=ht.int32, split=0)
        load_func_array = ht.load_csv_from_folder(
            csv_path, dtype=ht.int32, split=0, func=delete_first_col
        )
        load_array_float = ht.load_csv_from_folder(csv_path, dtype=ht.float32, split=0)

        load_array_npy = load_array.numpy()
        load_func_array_npy = load_func_array.numpy()

        self.assertIsInstance(load_array, ht.DNDarray)
        self.assertEqual(load_array.dtype, ht.int32)
        self.assertEqual(load_array_float.dtype, ht.float32)

        if ht.MPI_WORLD.rank == 0:
            self.assertTrue((load_array_npy == nparray).all)
            self.assertTrue((load_func_array_npy == npdroparray).all)
            shutil.rmtree(csv_path)

    def test_load_multiple_csv_exception(self):
        if not ht.io.supports_pandas():
            self.skipTest("Requires pandas")

        import pandas as pd

        with self.assertRaises(TypeError):
            ht.load_csv_from_folder(path=1, split=0)
        with self.assertRaises(TypeError):
            ht.load_csv_from_folder("heat/datasets", split="ABC")
        with self.assertRaises(TypeError):
            ht.load_csv_from_folder(path="heat/datasets", func=1)
        with self.assertRaises(ValueError):
            ht.load_csv_from_folder(path="heat", dtype=ht.int64, split=0)
        if ht.MPI_WORLD.size > 1:
            if ht.MPI_WORLD.rank == 0:
                os.mkdir(os.path.join(os.getcwd(), "heat/datasets/csv_tests"))
                df = pd.DataFrame({"A": [0, 0, 0]})  # noqa F821
                df.to_csv(
                    (os.path.join(os.getcwd(), "heat/datasets/csv_tests", "fail.csv")),
                    index=False,
                )
            ht.MPI_WORLD.Barrier()

            with self.assertRaises(RuntimeError):
                ht.load_csv_from_folder("heat/datasets/csv_tests", dtype=ht.int64, split=0)
            ht.MPI_WORLD.Barrier()
            if ht.MPI_WORLD.rank == 0:
                shutil.rmtree(os.path.join(os.getcwd(), "heat/datasets/csv_tests"))
