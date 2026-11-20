"""Tests for loading a folder of CSV files via pandas."""

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


@unittest.skipUnless(ht.io.supports("pandas"), "Requires pandas")
class TestCSVFolder(IOTestCase):
    """Tests for loading a folder of CSV files via pandas."""

    def test_load_multiple_csv(self):

        import pandas as pd

        csv_path, tmpdir = self.get_tmpdir()

        if ht.MPI_WORLD.rank == 0:
            nplist = []
            npdroplist = []
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

    def test_load_multiple_csv_exception(self):

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
            path, tmpdir = self.get_tmpdir()
            if ht.MPI_WORLD.rank == 0:
                df = pd.DataFrame({"A": [0, 0, 0]})  # noqa F821
                df.to_csv(
                    (os.path.join(os.getcwd(), path, "fail.csv")),
                    index=False,
                )
            ht.MPI_WORLD.Barrier()

            with self.assertRaises(RuntimeError):
                ht.load_csv_from_folder(path, dtype=ht.int64, split=0)
            ht.MPI_WORLD.Barrier()
