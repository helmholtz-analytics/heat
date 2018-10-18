import os
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

    try:
        import netCDF4
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
        # default parameters
        iris = ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), dataset='data')
        self.assertIsInstance(iris, ht.tensor)
        self.assertEqual(iris.shape, (150, 4,))
        self.assertEqual(iris.dtype, ht.float32)

        # default parameters
        iris = ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.nc'), variable='data')
        self.assertIsInstance(iris, ht.tensor)
        self.assertEqual(iris.shape, (150, 4,))
        self.assertEqual(iris.dtype, ht.float32)

    def test_load_exception(self):
        # correct extension file does not exist
        with self.assertRaises(IOError):
            ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/foo.h5'), 'data')
        with self.assertRaises(IOError):
            ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/foo.nc'), 'data')

        # unknown file extension
        with self.assertRaises(ValueError):
            ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris'), 'data')
        with self.assertRaises(ValueError):
            ht.load(os.path.join(os.getcwd(), 'heat/datasets/data/iris.csv'), 'data')
