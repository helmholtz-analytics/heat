import numpy as np
import os
import tempfile
import torch
import unittest

import heat as ht

if os.environ.get("HEAT_USE_DEVICE") == 'cpu':
    ht.use_device("cpu")
    torch_device = ht.get_device().torch_device
    heat_device = None
elif os.environ.get("HEAT_USE_DEVICE") == 'gpu' and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
    torch_device = ht.get_device().torch_device
    heat_device = None
elif os.environ.get("HEAT_USE_DEVICE") == 'lcpu' and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
    torch_device = ht.cpu.torch_device
    heat_device = ht.cpu
elif os.environ.get("HEAT_USE_DEVICE") == 'lgpu' and torch.cuda.is_available():
    ht.use_device("cpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
    torch_device = ht.cpu.torch_device
    heat_device = ht.cpu


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.HDF5_PATH = os.path.join(os.getcwd(), "heat/datasets/data/iris.h5")
        cls.HDF5_OUT_PATH = os.path.join(tempfile.gettempdir(), "test.h5")
        cls.HDF5_DATASET = "data"

        cls.NETCDF_PATH = os.path.join(os.getcwd(), "heat/datasets/data/iris.nc")
        cls.NETCDF_OUT_PATH = os.path.join(tempfile.gettempdir(), "test.nc")
        cls.NETCDF_VARIABLE = "data"

        # load comparison data from csv
        cls.CSV_PATH = os.path.join(os.getcwd(), "heat/datasets/data/iris.csv")
        cls.IRIS = torch.from_numpy(np.loadtxt(cls.CSV_PATH, delimiter=";")).float().to(torch_device)

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

        # synchronize all nodes
        ht.MPI_WORLD.Barrier()

    # catch-all loading
    def test_load(self):
        # HDF5
        if ht.io.supports_hdf5():
            iris = ht.load(self.HDF5_PATH, dataset="data", device=heat_device)
            self.assertIsInstance(iris, ht.DNDarray)
            # shape invariant
            self.assertEqual(iris.shape, self.IRIS.shape)
            self.assertEqual(iris._DNDarray__array.shape, self.IRIS.shape)
            # data type
            self.assertEqual(iris.dtype, ht.float32)
            self.assertEqual(iris._DNDarray__array.dtype, torch.float32)
            # content
            self.assertTrue((self.IRIS == iris._DNDarray__array).all())
        else:
            with self.assertRaises(ValueError):
                _ = ht.load(self.HDF5_PATH, dataset=self.HDF5_DATASET, device=heat_device)

        # netCDF
        if ht.io.supports_netcdf():
            iris = ht.load(self.NETCDF_PATH, variable=self.NETCDF_VARIABLE, device=heat_device)
            self.assertIsInstance(iris, ht.DNDarray)
            # shape invariant
            self.assertEqual(iris.shape, self.IRIS.shape)
            self.assertEqual(iris._DNDarray__array.shape, self.IRIS.shape)
            # data type
            self.assertEqual(iris.dtype, ht.float32)
            self.assertEqual(iris._DNDarray__array.dtype, torch.float32)
            # content
            self.assertTrue((self.IRIS == iris._DNDarray__array).all())
        else:
            with self.assertRaises(ValueError):
                _ = ht.load(self.NETCDF_PATH, variable=self.NETCDF_VARIABLE, device=heat_device)

    def test_load_csv(self):
        csv_file_length = 150
        csv_file_cols = 4
        first_value = torch.tensor([5.1, 3.5, 1.4, 0.2], dtype=torch.float32, device=torch_device)
        tenth_value = torch.tensor([4.9, 3.1, 1.5, 0.1], dtype=torch.float32, device=torch_device)

        a = ht.load_csv(self.CSV_PATH, sep=";", device=heat_device)
        self.assertEqual(len(a), csv_file_length)
        self.assertEqual(a.shape, (csv_file_length, csv_file_cols))
        self.assertTrue(torch.equal(a._DNDarray__array[0], first_value))
        self.assertTrue(torch.equal(a._DNDarray__array[9], tenth_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", split=0, device=heat_device)
        rank = a.comm.Get_rank()
        expected_gshape = (csv_file_length, csv_file_cols)
        self.assertEqual(a.gshape, expected_gshape)

        counts, _, _ = a.comm.counts_displs_shape(expected_gshape, 0)
        expected_lshape = (counts[rank], csv_file_cols)
        self.assertEqual(a.lshape, expected_lshape)

        if rank == 0:
            self.assertTrue(torch.equal(a._DNDarray__array[0], first_value))

        a = ht.load_csv(
            self.CSV_PATH, sep=";", header_lines=9, dtype=ht.float32, split=0, device=heat_device
        )
        expected_gshape = (csv_file_length - 9, csv_file_cols)
        counts, _, _ = a.comm.counts_displs_shape(expected_gshape, 0)
        expected_lshape = (counts[rank], csv_file_cols)

        self.assertEqual(a.gshape, expected_gshape)
        self.assertEqual(a.lshape, expected_lshape)
        self.assertEqual(a.dtype, ht.float32)
        if rank == 0:
            self.assertTrue(torch.equal(a._DNDarray__array[0], tenth_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", split=1, device=heat_device)
        self.assertEqual(a.shape, (csv_file_length, csv_file_cols))
        self.assertEqual(a.lshape[0], csv_file_length)

        a = ht.load_csv(self.CSV_PATH, sep=";", split=0, device=heat_device)
        b = ht.load(self.CSV_PATH, sep=";", split=0, device=heat_device)
        self.assertTrue(ht.equal(a, b))

        # Test for csv where header is longer then the first process`s share of lines
        a = ht.load_csv(self.CSV_PATH, sep=";", header_lines=100, split=0, device=heat_device)
        self.assertEqual(a.shape, (50, 4))

        with self.assertRaises(TypeError):
            ht.load_csv(12314, device=heat_device)
        with self.assertRaises(TypeError):
            ht.load_csv(self.CSV_PATH, sep=11, device=heat_device)
        with self.assertRaises(TypeError):
            ht.load_csv(self.CSV_PATH, header_lines="3", sep=";", split=0, device=heat_device)

    def test_load_exception(self):
        # correct extension, file does not exist
        if ht.io.supports_hdf5():
            with self.assertRaises(IOError):
                ht.load("foo.h5", "data", device=heat_device)
        else:
            with self.assertRaises(ValueError):
                ht.load("foo.h5", "data", device=heat_device)

        if ht.io.supports_netcdf():
            with self.assertRaises(IOError):
                ht.load("foo.nc", "data", device=heat_device)
        else:
            with self.assertRaises(ValueError):
                ht.load("foo.nc", "data", device=heat_device)

        # unknown file extension
        with self.assertRaises(ValueError):
            ht.load(
                os.path.join(os.getcwd(), "heat/datasets/data/iris.json"), "data", device=heat_device
            )
        with self.assertRaises(ValueError):
            ht.load("iris", "data", device=heat_device)

    # catch-all save
    def test_save(self):
        if ht.io.supports_hdf5():
            # local range
            local_range = ht.arange(100, device=heat_device)
            local_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET)
            if local_range.comm.rank == 0:
                with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET], dtype=torch.int32, device=torch_device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

            # split range
            split_range = ht.arange(100, split=0, device=heat_device)
            split_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET)
            if split_range.comm.rank == 0:
                with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET], dtype=torch.int32, device=torch_device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

        if ht.io.supports_netcdf():
            # local range
            local_range = ht.arange(100, device=heat_device)
            local_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if local_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=torch_device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

            # split range
            split_range = ht.arange(100, split=0, device=heat_device)
            split_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if split_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=torch_device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

    def test_save_exception(self):
        data = ht.arange(1, device=heat_device)

        if ht.io.supports_hdf5():
            with self.assertRaises(TypeError):
                ht.save(1, self.HDF5_OUT_PATH, self.HDF5_DATASET)
            with self.assertRaises(TypeError):
                ht.save(data, 1, self.HDF5_DATASET)
            with self.assertRaises(TypeError):
                ht.save(data, self.HDF5_OUT_PATH, 1)
        else:
            with self.assertRaises(ValueError):
                ht.save(data, self.HDF5_OUT_PATH, self.HDF5_DATASET)

        if ht.io.supports_netcdf():
            with self.assertRaises(TypeError):
                ht.save(1, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            with self.assertRaises(TypeError):
                ht.save(data, 1, self.NETCDF_VARIABLE)
            with self.assertRaises(TypeError):
                ht.save(data, self.NETCDF_OUT_PATH, 1)
        else:
            with self.assertRaises(ValueError):
                ht.save(data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)

    def test_load_hdf5(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # default parameters
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, device=heat_device)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris._DNDarray__array.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris._DNDarray__array).all())

        # positive split axis
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=0, device=heat_device)
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
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, dtype=ht.int8, device=heat_device)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.int8)
        self.assertEqual(iris._DNDarray__array.dtype, torch.int8)

    def test_load_hdf5_exception(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # improper argument types
        with self.assertRaises(TypeError):
            ht.load_hdf5(1, "data", device=heat_device)
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", 1, device=heat_device)
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", dataset="data", split=1.0, device=heat_device)

        # file or dataset does not exist
        with self.assertRaises(IOError):
            ht.load_hdf5("foo.h5", dataset="data", device=heat_device)
        with self.assertRaises(IOError):
            ht.load_hdf5("iris.h5", dataset="foo", device=heat_device)

    def test_save_hdf5(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # local unsplit data
        local_data = ht.arange(100, device=heat_device)
        ht.save_hdf5(local_data, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        if local_data.comm.rank == 0:
            with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=torch_device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0, device=heat_device)
        ht.save_hdf5(split_data, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        if split_data.comm.rank == 0:
            with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=torch_device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

    def test_save_hdf5_exception(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # dummy data
        data = ht.arange(1, device=heat_device)

        with self.assertRaises(TypeError):
            ht.save_hdf5(1, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        with self.assertRaises(TypeError):
            ht.save_hdf5(data, 1, self.HDF5_DATASET)
        with self.assertRaises(TypeError):
            ht.save_hdf5(data, self.HDF5_OUT_PATH, 1)

    def test_load_netcdf(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            return

        # default parameters
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, device=heat_device)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris._DNDarray__array.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris._DNDarray__array).all())

        # positive split axis
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, split=0, device=heat_device)
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
        iris = ht.load_netcdf(
            self.NETCDF_PATH, self.NETCDF_VARIABLE, dtype=ht.int8, device=heat_device
        )
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.int8)
        self.assertEqual(iris._DNDarray__array.dtype, torch.int8)

    def test_load_netcdf_exception(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            return

        # improper argument types
        with self.assertRaises(TypeError):
            ht.load_netcdf(1, "data", device=heat_device)
        with self.assertRaises(TypeError):
            ht.load_netcdf("iris.nc", variable=1, device=heat_device)
        with self.assertRaises(TypeError):
            ht.load_netcdf("iris.nc", variable="data", split=1.0, device=heat_device)

        # file or variable does not exist
        with self.assertRaises(IOError):
            ht.load_netcdf("foo.nc", variable="data", device=heat_device)
        with self.assertRaises(IOError):
            ht.load_netcdf("iris.nc", variable="foo", device=heat_device)

    def test_save_netcdf(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            return

        # local unsplit data
        local_data = ht.arange(100, device=heat_device)
        ht.save_netcdf(local_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if local_data.comm.rank == 0:
            with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=torch_device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0, device=heat_device)
        ht.save_netcdf(split_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if split_data.comm.rank == 0:
            with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=torch_device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

    def test_save_netcdf_exception(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            return

        # dummy data
        data = ht.arange(1, device=heat_device)

        with self.assertRaises(TypeError):
            ht.save_netcdf(1, self.NETCDF_PATH, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, 1, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, self.NETCDF_PATH, 1)
