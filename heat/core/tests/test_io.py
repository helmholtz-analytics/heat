import os
import tempfile
import unittest

import heat as ht


class TestIO(unittest.TestCase):
    try:
        import h5py
    except ImportError:
        # HDF5 support is optional
        pass
    else:
        def test_load_hdf5(self):
            # default parameters
            iris = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data')
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # positive split axis
            iris = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=1)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # negative split axis
            iris = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=-1)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # different data type
            iris = ht.load_hdf5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', dtype=ht.int8)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.int8)

        def test_load_hdf5_exception(self):
            # improper argument types
            with self.assertRaises(TypeError):
                ht.load_hdf5(1, 'data')
            with self.assertRaises(TypeError):
                ht.load_hdf5('iris.h5', 1)
            with self.assertRaises(TypeError):
                ht.load_hdf5('iris.h5', dataset='data', split=1.0)

            # file or dataset does not exist
            with self.assertRaises(IOError):
                ht.load_hdf5('foo.h5', dataset='data')
            with self.assertRaises(IOError):
                ht.load_hdf5('iris.h5', dataset='foo')

        def test_save_hdf5(self):
            data = ht.arange(100)
            ht.save_hdf5(data, os.path.join(tempfile.gettempdir(), 'test0.h5'), 'data')
            # ht.save_hdf5(data, os.path.join(os.getcwd(), 'test0.h5'), 'data')

        def test_save_hdf5_exception(self):
            # dummy data
            data = ht.arange(1)

            with self.assertRaises(TypeError):
                ht.save_hdf5(1, os.path.join(tempfile.gettempdir(), 'test.h5'), 'data')
            with self.assertRaises(TypeError):
                ht.save_hdf5(data, 1, 'data')
            with self.assertRaises(TypeError):
                ht.save_hdf5(data, os.path.join(tempfile.gettempdir(), 'test.h5'), 1)

    try:
        import netCDF4 as nc
    except ImportError:
        # netCDF is optional
        pass
    else:
        def test_load_netcdf(self):
            # default parameters
            iris = ht.load_netcdf(os.path.join(os.getcwd(), 'heat/datasets/data/iris.nc'), 'data')
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # positive split axis
            iris = ht.load_netcdf(os.path.join(os.getcwd(), 'heat/datasets/data/iris.nc'), 'data', split=1)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # negative split axis
            iris = ht.load_netcdf(os.path.join(os.getcwd(), 'heat/datasets/data/iris.nc'), 'data', split=-1)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # different data type
            iris = ht.load_netcdf(os.path.join(os.getcwd(), 'heat/datasets/data/iris.nc'), 'data', dtype=ht.int8)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.int8)

        def test_load_netcdf_exception(self):
            # improper argument types
            with self.assertRaises(TypeError):
                ht.load_netcdf(1, 'data')
            with self.assertRaises(TypeError):
                ht.load_netcdf('iris.nc', variable=1)
            with self.assertRaises(TypeError):
                ht.load_netcdf('iris.nc', variable='data', split=1.0)

            # file or variable does not exist
            with self.assertRaises(IOError):
                ht.load_netcdf('foo.nc', variable='data')
            with self.assertRaises(IOError):
                ht.load_netcdf('iris.nc', variable='foo')

    def test_load(self):
        # HDF5
        try:
            iris = ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), dataset='data')
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)
        except ValueError:
            # HDF5 is optional
            pass

        # NetCDF4
        try:
            iris = ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.nc'), variable='data')
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)
        except ValueError:
            # NetCDF4 is optional
            pass

    def test_load_exception(self):
        # correct extension file does not exist
        try:
            with self.assertRaises(IOError):
                ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/foo.h5'), 'data')
        except ValueError:
            # HDF5 is optional
            pass

        try:
            with self.assertRaises(IOError):
                ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/foo.nc'), 'data')
        except ValueError:
            # netCDF4 is optional
            pass

        # unknown file extension
        with self.assertRaises(ValueError):
            ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris'), 'data')
        with self.assertRaises(ValueError):
            ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.csv'), 'data')
