"""Tests for reading and writing CSV files."""

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

from heat.testing.basic_test import TestCase

from ._base import IOTestCase


class TestCSV(IOTestCase):
    """Tests for reading and writing CSV files."""

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

    @unittest.skipIf(
        len(TestCase.get_hostnames()) > 1 and not os.environ.get("TMPDIR"),
        "Requires the environment variable 'TMPDIR' to point to a globally accessible path. Otherwise the test will be skiped on multi-node setups.",
    )

    def test_save_csv(self):
        # Test for different random types
        # include float64 only if device is not MPS
        data = None
        if self.is_mps:
            rnd_types = [
                (ht.random.randint, ht.types.int32),
                (ht.random.randint, ht.types.int64),
                (ht.random.rand, ht.types.float32),
            ]
        else:
            rnd_types = [
                (ht.random.randint, ht.types.int32),
                (ht.random.randint, ht.types.int64),
                (ht.random.rand, ht.types.float32),
                (ht.random.rand, ht.types.float64),
            ]
        for rnd_type in rnd_types:
            for separator in [",", ";", "|"]:
                for split in [None, 0, 1]:
                    for headers in [None, ["# This", "# is a", "# test."]]:
                        for shape in [(1, 1), (10, 10), (20, 1), (1, 20), (25, 4), (4, 25)]:
                            if rnd_type[0] == ht.random.randint:
                                data: ht.DNDarray = rnd_type[0](
                                    -1000, 1000, size=shape, dtype=rnd_type[1], split=split
                                )
                            else:
                                data: ht.DNDarray = rnd_type[0](
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
