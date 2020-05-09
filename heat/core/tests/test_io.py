import numpy as np
import os
import tempfile
import torch
import unittest

import heat as ht

if os.environ.get("DEVICE") == "gpu" and torch.cuda.is_available():
    ht.use_device("gpu")
    torch.cuda.set_device(torch.device(ht.get_device().torch_device))
else:
    ht.use_device("cpu")
device = ht.get_device().torch_device
ht_device = None
if os.environ.get("DEVICE") == "lgpu" and torch.cuda.is_available():
    device = ht.gpu.torch_device
    ht_device = ht.gpu
    torch.cuda.set_device(device)


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.HDF5_PATH = os.path.join(os.getcwd(), "heat/datasets/data/iris.h5")
        cls.HDF5_OUT_PATH = os.path.join(tempfile.gettempdir(), "test.h5")
        cls.HDF5_DATASET = "data"

        cls.NETCDF_PATH = os.path.join(os.getcwd(), "heat/datasets/data/iris.nc")
        cls.NETCDF_OUT_PATH = os.path.join(tempfile.gettempdir(), "test.nc")
        cls.NETCDF_VARIABLE = "data"
        cls.NETCDF_DIMENSION = "data"

        # load comparison data from csv
        cls.CSV_PATH = os.path.join(os.getcwd(), "heat/datasets/data/iris.csv")
        cls.IRIS = torch.from_numpy(np.loadtxt(cls.CSV_PATH, delimiter=";")).float().to(device)

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
            iris = ht.load(self.HDF5_PATH, dataset="data", device=ht_device)
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
                _ = ht.load(self.HDF5_PATH, dataset=self.HDF5_DATASET, device=ht_device)

        # netCDF
        if ht.io.supports_netcdf():
            iris = ht.load(self.NETCDF_PATH, variable=self.NETCDF_VARIABLE, device=ht_device)
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
                _ = ht.load(self.NETCDF_PATH, variable=self.NETCDF_VARIABLE, device=ht_device)

    def test_load_csv(self):
        csv_file_length = 150
        csv_file_cols = 4
        first_value = torch.tensor([5.1, 3.5, 1.4, 0.2], dtype=torch.float32, device=device)
        tenth_value = torch.tensor([4.9, 3.1, 1.5, 0.1], dtype=torch.float32, device=device)

        a = ht.load_csv(self.CSV_PATH, sep=";", device=ht_device)
        self.assertEqual(len(a), csv_file_length)
        self.assertEqual(a.shape, (csv_file_length, csv_file_cols))
        self.assertTrue(torch.equal(a._DNDarray__array[0], first_value))
        self.assertTrue(torch.equal(a._DNDarray__array[9], tenth_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", split=0, device=ht_device)
        rank = a.comm.Get_rank()
        expected_gshape = (csv_file_length, csv_file_cols)
        self.assertEqual(a.gshape, expected_gshape)

        counts, _, _ = a.comm.counts_displs_shape(expected_gshape, 0)
        expected_lshape = (counts[rank], csv_file_cols)
        self.assertEqual(a.lshape, expected_lshape)

        if rank == 0:
            self.assertTrue(torch.equal(a._DNDarray__array[0], first_value))

        a = ht.load_csv(
            self.CSV_PATH, sep=";", header_lines=9, dtype=ht.float32, split=0, device=ht_device
        )
        expected_gshape = (csv_file_length - 9, csv_file_cols)
        counts, _, _ = a.comm.counts_displs_shape(expected_gshape, 0)
        expected_lshape = (counts[rank], csv_file_cols)

        self.assertEqual(a.gshape, expected_gshape)
        self.assertEqual(a.lshape, expected_lshape)
        self.assertEqual(a.dtype, ht.float32)
        if rank == 0:
            self.assertTrue(torch.equal(a._DNDarray__array[0], tenth_value))

        a = ht.load_csv(self.CSV_PATH, sep=";", split=1, device=ht_device)
        self.assertEqual(a.shape, (csv_file_length, csv_file_cols))
        self.assertEqual(a.lshape[0], csv_file_length)

        a = ht.load_csv(self.CSV_PATH, sep=";", split=0, device=ht_device)
        b = ht.load(self.CSV_PATH, sep=";", split=0, device=ht_device)
        self.assertTrue(ht.equal(a, b))

        # Test for csv where header is longer then the first process`s share of lines
        a = ht.load_csv(self.CSV_PATH, sep=";", header_lines=100, split=0, device=ht_device)
        self.assertEqual(a.shape, (50, 4))

        with self.assertRaises(TypeError):
            ht.load_csv(12314, device=ht_device)
        with self.assertRaises(TypeError):
            ht.load_csv(self.CSV_PATH, sep=11, device=ht_device)
        with self.assertRaises(TypeError):
            ht.load_csv(self.CSV_PATH, header_lines="3", sep=";", split=0, device=ht_device)

    def test_load_exception(self):
        # correct extension, file does not exist
        if ht.io.supports_hdf5():
            with self.assertRaises(IOError):
                ht.load("foo.h5", "data", device=ht_device)
        else:
            with self.assertRaises(ValueError):
                ht.load("foo.h5", "data", device=ht_device)

        if ht.io.supports_netcdf():
            with self.assertRaises(IOError):
                ht.load("foo.nc", "data", device=ht_device)
        else:
            with self.assertRaises(ValueError):
                ht.load("foo.nc", "data", device=ht_device)

        # unknown file extension
        with self.assertRaises(ValueError):
            ht.load(
                os.path.join(os.getcwd(), "heat/datasets/data/iris.json"), "data", device=ht_device
            )
        with self.assertRaises(ValueError):
            ht.load("iris", "data", device=ht_device)

    # catch-all save
    def test_save(self):
        if ht.io.supports_hdf5():
            # local range
            local_range = ht.arange(100, device=ht_device)
            local_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET)
            if local_range.comm.rank == 0:
                with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET], dtype=torch.int32, device=device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

            # split range
            split_range = ht.arange(100, split=0, device=ht_device)
            split_range.save(self.HDF5_OUT_PATH, self.HDF5_DATASET)
            if split_range.comm.rank == 0:
                with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.HDF5_DATASET], dtype=torch.int32, device=device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

        if ht.io.supports_netcdf():
            # local range
            local_range = ht.arange(100, device=ht_device)
            local_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if local_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

            # split range
            split_range = ht.arange(100, split=0, device=ht_device)
            split_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
            if split_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=device
                    )
                self.assertTrue((local_range._DNDarray__array == comparison).all())

            # naming dimensions
            local_range = ht.arange(100, device=ht_device)
            local_range.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, dimension_names=self.NETCDF_DIMENSION
            )
            if local_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = handle[self.NETCDF_VARIABLE].dimensions
                self.assertTrue(self.NETCDF_DIMENSION in comparison)

            # appending unlimited variable
            split_range.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, is_unlimited=True)
            split_range.save(
                self.NETCDF_OUT_PATH,
                self.NETCDF_VARIABLE,
                mode="r+",
                file_slices=slice(split_range.size, None, None),
            )
            if split_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=device
                    )
                self.assertTrue(
                    (
                        ht.concatenate((local_range, local_range))._DNDarray__array == comparison
                    ).all()
                )

            # indexing netcdf file: single index
            zeros = ht.zeros((20,1,20,2), device=ht_device)
            zeros.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w")
            ones = ht.ones(20, device=ht_device)
            indices = (-1,0,slice(None),1)
            ones.save(self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+", file_slices=indices)
            if split_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices], dtype=torch.int32, device=device
                    )
                self.assertTrue((ones._DNDarray__array == comparison).all())

            # indexing netcdf file: multiple indices
            small_range_split = ht.arange(10, split=0, device=ht_device)
            small_range = ht.arange(10, device=ht_device)
            indices = [[0, 9, 5, 2, 1, 3, 7, 4, 8, 6]]
            small_range_split.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="w", file_slices=indices
            )
            if split_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][indices], dtype=torch.int32, device=device
                    )
                self.assertTrue((small_range._DNDarray__array == comparison).all())

            # slicing netcdf file
            sslice = slice(7, 2, -1)
            range_five_split = ht.arange(5, split=0, device=ht_device)
            range_five = ht.arange(5, device=ht_device)
            range_five_split.save(
                self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE, mode="r+", file_slices=sslice
            )
            if split_range.comm.rank == 0:
                with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                    comparison = torch.tensor(
                        handle[self.NETCDF_VARIABLE][sslice], dtype=torch.int32, device=device
                    )
                self.assertTrue((range_five._DNDarray__array == comparison).all())

    def test_save_exception(self):
        data = ht.arange(1, device=ht_device)

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
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, device=ht_device)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris._DNDarray__array.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris._DNDarray__array).all())

        # positive split axis
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=0, device=ht_device)
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
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, dtype=ht.int8, device=ht_device)
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
            ht.load_hdf5(1, "data", device=ht_device)
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", 1, device=ht_device)
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", dataset="data", split=1.0, device=ht_device)

        # file or dataset does not exist
        with self.assertRaises(IOError):
            ht.load_hdf5("foo.h5", dataset="data", device=ht_device)
        with self.assertRaises(IOError):
            ht.load_hdf5("iris.h5", dataset="foo", device=ht_device)

    def test_save_hdf5(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # local unsplit data
        local_data = ht.arange(100, device=ht_device)
        ht.save_hdf5(local_data, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        if local_data.comm.rank == 0:
            with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0, device=ht_device)
        ht.save_hdf5(split_data, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        if split_data.comm.rank == 0:
            with ht.io.h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

    def test_save_hdf5_exception(self):
        # HDF5 support is optional
        if not ht.io.supports_hdf5():
            return

        # dummy data
        data = ht.arange(1, device=ht_device)

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
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, device=ht_device)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris._DNDarray__array.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris._DNDarray__array).all())

        # positive split axis
        iris = ht.load_netcdf(self.NETCDF_PATH, self.NETCDF_VARIABLE, split=0, device=ht_device)
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
            self.NETCDF_PATH, self.NETCDF_VARIABLE, dtype=ht.int8, device=ht_device
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
            ht.load_netcdf(1, "data", device=ht_device)
        with self.assertRaises(TypeError):
            ht.load_netcdf("iris.nc", variable=1, device=ht_device)
        with self.assertRaises(TypeError):
            ht.load_netcdf("iris.nc", variable="data", split=1.0, device=ht_device)

        # file or variable does not exist
        with self.assertRaises(IOError):
            ht.load_netcdf("foo.nc", variable="data", device=ht_device)
        with self.assertRaises(IOError):
            ht.load_netcdf("iris.nc", variable="foo", device=ht_device)

    def test_save_netcdf(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            return

        # local unsplit data
        local_data = ht.arange(100, device=ht_device)
        ht.save_netcdf(local_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if local_data.comm.rank == 0:
            with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0, device=ht_device)
        ht.save_netcdf(split_data, self.NETCDF_OUT_PATH, self.NETCDF_VARIABLE)
        if split_data.comm.rank == 0:
            with ht.io.nc.Dataset(self.NETCDF_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.NETCDF_VARIABLE][:], dtype=torch.int32, device=device
                )
            self.assertTrue((local_data._DNDarray__array == comparison).all())

    def test_save_netcdf_exception(self):
        # netcdf support is optional
        if not ht.io.supports_netcdf():
            return

        # dummy data
        data = ht.arange(1, device=ht_device)

        with self.assertRaises(TypeError):
            ht.save_netcdf(1, self.NETCDF_PATH, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, 1, self.NETCDF_VARIABLE)
        with self.assertRaises(TypeError):
            ht.save_netcdf(data, self.NETCDF_PATH, 1)
