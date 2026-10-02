"""Tests for reading and writing HDF5 files."""

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


@unittest.skipUnless(ht.io.supports("hdf5"), "Requires hdf5")
class TestHDF5(IOTestCase):
    """Tests for reading and writing HDF5 files."""

    def test_load_hdf5(self):
        # HDF5 support is optional

        # default parameters
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, dtype=ht.float32)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float32)
        self.assertEqual(iris.larray.dtype, torch.float32)
        self.assertTrue((self.IRIS == iris.larray).all())

        # positive split axis
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=0)
        self.assertIsInstance(iris, ht.DNDarray)
        self.assertEqual(iris.shape, self.IRIS.shape)
        self.assertEqual(iris.dtype, ht.float64)
        lshape = iris.lshape
        self.assertLessEqual(lshape[0], self.IRIS.shape[0])
        self.assertEqual(lshape[1], self.IRIS.shape[1])

        # negative split axis
        iris = ht.load_hdf5(self.HDF5_PATH, self.HDF5_DATASET, split=-1, dtype=ht.float32)
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

        # improper argument types
        with self.assertRaises(TypeError):
            ht.load_hdf5(1, "data")
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", 1)
        with self.assertRaises(TypeError):
            ht.load_hdf5("iris.h5", dataset="data", split=1.0)

        # file or dataset does not exist
        with self.assertRaises(IOError):
            ht.load_hdf5("foo.h5", dataset="data")
        with self.assertRaises(IOError):
            ht.load_hdf5("iris.h5", dataset="foo")

    def test_save_hdf5(self):
        # HDF5 support is optional
        if not ht.io.supports("hdf5"):
            return

        # local unsplit data
        local_data = ht.arange(100)
        ht.save_hdf5(
            local_data, self.HDF5_OUT_PATH, self.HDF5_DATASET, dtype=torch.int32
        )
        if local_data.comm.rank == 0:
            with h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=self.device.torch_device
                )
            self.assertTrue((local_data.larray == comparison).all())

        # distributed data range
        split_data = ht.arange(100, split=0)
        ht.save_hdf5(
            split_data, self.HDF5_OUT_PATH, self.HDF5_DATASET
        )
        if split_data.comm.rank == 0:
            with h5py.File(self.HDF5_OUT_PATH, "r") as handle:
                comparison = torch.tensor(
                    handle[self.HDF5_DATASET], dtype=torch.int32, device=self.device.torch_device
                )
            self.assertTrue((local_data.larray == comparison).all())

    def test_save_hdf5_exception(self):
        # HDF5 support is optional

        # dummy data
        data = ht.arange(1)

        with self.assertRaises(TypeError):
            ht.save_hdf5(1, self.HDF5_OUT_PATH, self.HDF5_DATASET)
        with self.assertRaises(TypeError):
            ht.save_hdf5(data, 1, self.HDF5_DATASET)
        with self.assertRaises(TypeError):
            ht.save_hdf5(data, self.HDF5_OUT_PATH, 1)

    @unittest.skipIf(not ht.io.supports("hdf5"), reason="Requires HDF5")
    def test_load_partial_hdf5(self):
        test_axis = [None, 0, 1]
        test_slices = [
            (slice(0, 50, None), slice(None, None, None)),
            (slice(0, 50, None), slice(0, 2, None)),
            (slice(50, 100, None), slice(None, None, None)),
            (slice(None, None, None), slice(2, 4, None)),
            (slice(50), None),
            (None, slice(0, 3, 2)),
            (slice(50),),
            (slice(50, 100),),
        ]
        test_cases = [(a, s) for a in test_axis for s in test_slices]

        for axis, slices in test_cases:
            with self.subTest(axis=axis, slices=slices):
                HDF5_DATASET = "data"
                expect_error = False
                for s in slices:
                    if s and s.step not in [None, 1]:
                        expect_error = True
                        break

                if expect_error:
                    with self.assertRaises(ValueError):
                        sliced_iris = ht.load_hdf5(
                            self.HDF5_PATH, HDF5_DATASET, split=axis, slices=slices
                        )
                else:
                    original_iris = ht.load_hdf5(self.HDF5_PATH, HDF5_DATASET, split=axis)
                    tmp_slices = tuple(slice(None) if s is None else s for s in slices)
                    expected_iris = original_iris[tmp_slices]
                    sliced_iris = ht.load_hdf5(self.HDF5_PATH, HDF5_DATASET, split=axis, slices=slices)
                    self.assertTrue(ht.equal(sliced_iris, expected_iris))

    def test_load_multiple_hdf5_even(self):

        import h5py

        N_FILES = 11
        N_ROWS = 4
        N_COLUMNS = 5
        G_SHAPE = (N_FILES * N_ROWS, N_COLUMNS)
        ELEMS = G_SHAPE[0] * G_SHAPE[1]
        comm = ht.MPI_WORLD

        original_data = torch.arange(0, ELEMS, dtype=torch.int64).view(G_SHAPE)

        rank_slices = [comm.chunk(G_SHAPE, split=0, rank=i)[-1][0] for i in range(N_FILES)]  # all row slices
        local_slice = rank_slices[comm.rank]

        Path(self.HDF5_MULTIPLE_FOLDER).mkdir(exist_ok=True)

        if comm.rank == 0:
            for n in range(N_FILES):
                file_path = Path(self.HDF5_MULTIPLE_FOLDER, self.HDF5_MULTIPLE_FILE_PREFIX+str(n)+self.HDF5_MULTIPLE_FILE_ENDING)
                with h5py.File(str(file_path), "w") as file:
                    file[self.HDF5_MULTIPLE_DATASET] = original_data[rank_slices[n]].numpy()

        comm.Barrier()

        dndarray = ht.io.load_multiple_hdf5(self.HDF5_MULTIPLE_FOLDER, self.HDF5_MULTIPLE_DATASET, dtype=torch.int64)
        dndarray_np = dndarray.numpy()
        original_data_np = original_data.numpy()
        self.assertTrue((dndarray_np == original_data_np).all())

    def test_load_multiple_hdf5_uneven(self):

        import h5py

        N_FILES = 9
        N_ROWS = [2, 3, 1, 4, 6, 2, 1, 9, 2]
        TOTAL_ROWS = sum(N_ROWS)
        N_COLUMNS = 2
        G_SHAPE = (TOTAL_ROWS, N_COLUMNS)
        ELEMS = G_SHAPE[0] * G_SHAPE[1]
        comm = ht.MPI_WORLD

        original_data = torch.arange(0, ELEMS, dtype=torch.float32).view(G_SHAPE)

        Path(self.HDF5_MULTIPLE_FOLDER).mkdir(exist_ok=True)

        if comm.rank == 0:
            for n in range(N_FILES):
                file_path = Path(self.HDF5_MULTIPLE_FOLDER, self.HDF5_MULTIPLE_FILE_PREFIX+str(n)+self.HDF5_MULTIPLE_FILE_ENDING)
                written_rows = sum(N_ROWS[:n])
                to_write = N_ROWS[n]
                with h5py.File(str(file_path), "w") as file:
                    file[self.HDF5_MULTIPLE_DATASET] = original_data[written_rows:written_rows+to_write].numpy()

        comm.Barrier()

        dndarray = ht.io.load_multiple_hdf5(self.HDF5_MULTIPLE_FOLDER, self.HDF5_MULTIPLE_DATASET)
        dndarray_np = dndarray.numpy()
        original_data_np = original_data.numpy()
        self.assertTrue((dndarray_np == original_data_np).all())

    @unittest.skipIf(not ht.io.supports("hdf5"), reason="Requires HDF5")
    def test_load_multiple_hdf5_exceptions(self):
        # wrong type for folder path
        with self.assertRaises(TypeError):
            ht.io.load_multiple_hdf5(1, "my_dataset_name")

        # wrong type for dataset name
        with self.assertRaises(TypeError):
            ht.io.load_multiple_hdf5("/my_folder_name", 3.14)

        # wrong type for sorting function
        with self.assertRaises(TypeError):
            ht.io.load_multiple_hdf5("/my_folder_name", "my_dataset_name", sorting_func=5)

        # folder does not exist
        with self.assertRaises(ValueError):
            ht.io.load_multiple_hdf5("/this/folder/does/not/exist", "my_dataset_name")

        import h5py
        comm = ht.MPI_WORLD

        # folder is empty
        empty_folder = Path(self.HDF5_MULTIPLE_FOLDER, "empty_test_folder")
        if comm.rank == 0:
            empty_folder.mkdir(parents=True,exist_ok=True)
        comm.Barrier()
        with self.assertRaises(ValueError):
            ht.io.load_multiple_hdf5(str(empty_folder), "my_dataset_name")
        comm.Barrier()
        if comm.rank == 0:
            empty_folder.rmdir()
        comm.Barrier()

        # Amount of dimensions of all hdf5 files must be the same
        inconsistent_folder = Path(self.HDF5_MULTIPLE_FOLDER, "inconsistent_test_folder")
        if comm.rank == 0:
            inconsistent_folder.mkdir(exist_ok=True)
            # create first file with dataset of shape (4, 5)
            with h5py.File(str(Path(inconsistent_folder, "file_0.h5")), "w") as file:
                file["my_dataset"] = np.random.rand(4, 5)
            # create second file with dataset of shape (4, 5, 6)
            with h5py.File(str(Path(inconsistent_folder, "file_1.h5")), "w") as file:
                file["my_dataset"] = np.random.rand(4, 5, 6)
        comm.Barrier()
        with self.assertRaises(ValueError):
            ht.io.load_multiple_hdf5(str(inconsistent_folder), "my_dataset")
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(inconsistent_folder)

        # Dimension missmatch on ndim
        missmatch_folder = Path(self.HDF5_MULTIPLE_FOLDER, "missmatch_test_folder")
        if comm.rank == 0:
            missmatch_folder.mkdir(exist_ok=True)
            # create first file with dataset of shape (4, 5)
            with h5py.File(str(Path(missmatch_folder, "file_0.h5")), "w") as file:
                file["my_dataset"] = np.random.rand(4, 5)
            # create second file with dataset of shape (6, 5)
            with h5py.File(str(Path(missmatch_folder, "file_1.h5")), "w") as file:
                file["my_dataset"] = np.random.rand(6, 7)
        comm.Barrier()
        with self.assertRaises(ValueError):
            ht.io.load_multiple_hdf5(str(missmatch_folder), "my_dataset", dtype=ht.float32)
        comm.Barrier()
        if comm.rank == 0:
            shutil.rmtree(missmatch_folder)
