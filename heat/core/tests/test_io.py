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
        def test_loadh5(self):
            # default parameters
            iris = ht.loadh5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data')
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # positive split axis
            iris = ht.loadh5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=1)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # negative split axis
            iris = ht.loadh5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', split=-1)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.float32)

            # different data type
            iris = ht.loadh5(os.path.join(os.getcwd(), 'heat/datasets/data/iris.h5'), 'data', dtype=ht.int8)
            self.assertIsInstance(iris, ht.tensor)
            self.assertEqual(iris.shape, (150, 4,))
            self.assertEqual(iris.dtype, ht.int8)
