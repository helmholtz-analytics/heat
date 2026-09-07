"""Tests for loading a folder of NumPy .npy files."""

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


class TestNPY(IOTestCase):
    """Tests for loading a folder of NumPy .npy files."""

    def test_load_npy_int(self):
        temp_path = Path.cwd()
        # testing for int arrays
        if ht.MPI_WORLD.rank == 0:
            crea_array = []
            for i in range(0, ht.MPI_WORLD.size * 5):
                x = np.random.randint(1000, size=(random.randint(0, 30), 6, 11))
                np.save(str(temp_path / "int_data") + str(i), x)
                crea_array.append(x)
            int_array = np.concatenate(crea_array)
        ht.MPI_WORLD.Barrier()
        load_array = ht.load_npy_from_path(
            str(temp_path), dtype=ht.int32, split=0
        )
        load_array_npy = load_array.numpy()

        self.assertIsInstance(load_array, ht.DNDarray)
        self.assertEqual(load_array.dtype, ht.int32)
        if ht.MPI_WORLD.rank == 0:
            self.assertTrue((load_array_npy == int_array).all)
            for file in os.listdir(str(temp_path)):
                if fnmatch.fnmatch(file, "*.npy"):
                    os.remove(str(temp_path / file))

    def test_load_npy_float(self):
        temp_path = Path.cwd()
        # testing for float arrays and split dimension other than 0
        if ht.MPI_WORLD.rank == 0:
            crea_array = []
            for i in range(0, ht.MPI_WORLD.size * 5 + 1):
                x = np.random.rand(2, random.randint(1, 10), 11)
                np.save(str(temp_path / "float_data") + str(i), x)
                crea_array.append(x)
            float_array = np.concatenate(crea_array, 1)
        ht.MPI_WORLD.Barrier()

        if not self.is_mps:
            # float64 not supported in MPS
            load_array = ht.load_npy_from_path(
                str(temp_path), dtype=ht.float64, split=1
            )
            load_array_npy = load_array.numpy()
            self.assertIsInstance(load_array, ht.DNDarray)
            self.assertEqual(load_array.dtype, ht.float64)
            if ht.MPI_WORLD.rank == 0:
                self.assertTrue((load_array_npy == float_array).all)
        if ht.MPI_WORLD.rank == 0:
            for file in os.listdir(str(temp_path)):
                if fnmatch.fnmatch(file, "*.npy"):
                    os.remove(str(temp_path / file))

    def test_load_npy_exception(self):
        with self.assertRaises(TypeError):
            ht.load_npy_from_path(path=1, split=0)
        with self.assertRaises(TypeError):
            ht.load_npy_from_path("heat/datasets", split="ABC")
        with self.assertRaises(ValueError):
            ht.load_npy_from_path(path="heat", dtype=ht.int64, split=0)
        if ht.MPI_WORLD.size > 1:
            path, tmpdir = self.get_tmpdir()
            if ht.MPI_WORLD.rank == 0:
                x = np.random.rand(2, random.randint(1, 10), 11)
                np.save(os.path.join(path, "float_data"), x)
            ht.MPI_WORLD.Barrier()
            with self.assertRaises(RuntimeError):
                ht.load_npy_from_path(path, dtype=ht.int64, split=0)
            ht.MPI_WORLD.Barrier()
