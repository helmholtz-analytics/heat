"""Tests for the helpers shared by the file-format modules of ``heat.core.io``."""

import os
import unittest

import heat as ht
from heat.core.io import utils
from heat.testing.basic_test import TestCase


class TestIOUtils(TestCase):
    """Tests for :mod:`heat.core.io.utils`."""

    def test_size_from_slice(self):
        test_cases = [
            (1000, slice(500), 500, 0),
            (10, slice(0, 10, 2), 5, 0),
            (100, slice(10, 50, 3), 14, 10),
            (1000, slice(0), 0, 0),
            (0, slice(0), 0, 0),
        ]
        for size, slice_obj, expected_new_size, expected_offset in test_cases:
            with self.subTest(size=size, slice=slice_obj):
                new_size, offset = utils.size_from_slice(size, slice_obj)
                self.assertEqual(new_size, expected_new_size)
                self.assertEqual(offset, expected_offset)

    def test_sanitize_path(self):
        self.assertEqual(utils.sanitize_path("some/path"), "some/path")
        with self.assertRaises(TypeError):
            utils.sanitize_path(1)
        with self.assertRaises(TypeError):
            utils.sanitize_path(None)

    def test_extension_of(self):
        self.assertEqual(utils.extension_of("data.h5"), ".h5")
        self.assertEqual(utils.extension_of("a/b/DATA.HDF5"), ".hdf5")
        self.assertEqual(utils.extension_of("data.nc "), ".nc")
        self.assertEqual(utils.extension_of("noextension"), "")

    def test_sanitize_extension(self):
        self.assertEqual(utils.sanitize_extension("d.h5", {".h5", ".hdf5"}, "HDF5"), "d.h5")
        with self.assertRaises(ValueError):
            utils.sanitize_extension("d.nc", {".h5", ".hdf5"}, "HDF5")

    def test_sanitize_write_mode(self):
        for mode in ("w", "a", "r+"):
            self.assertEqual(utils.sanitize_write_mode(mode), mode)
        with self.assertRaises(ValueError):
            utils.sanitize_write_mode("x")

    def test_split_file_indices(self):
        for n_files in range(1, 12):
            for size in range(1, min(n_files, 5) + 1):
                with self.subTest(n_files=n_files, size=size):
                    covered = []
                    for rank in range(size):
                        idx, count = utils.split_file_indices(n_files, rank, size)
                        covered.extend(range(idx, idx + count))
                    # every file is read exactly once, and the load is balanced
                    self.assertEqual(covered, list(range(n_files)))

    def test_serialized_write_orders_processes(self):
        comm = ht.MPI_WORLD
        path = os.path.join(os.getcwd(), "test_serialized_write.txt")

        def write_root():
            with open(path, "w") as handle:
                handle.write(f"{comm.rank}\n")

        def write_other():
            with open(path, "a") as handle:
                handle.write(f"{comm.rank}\n")

        utils.serialized_write(comm, True, write_root, write_other)
        comm.Barrier()
        with open(path) as handle:
            ranks = [int(line) for line in handle if line.strip()]
        self.assertEqual(ranks, list(range(comm.size)))
        comm.Barrier()
        if comm.rank == 0:
            os.remove(path)
        comm.Barrier()

    def test_serialized_write_skips_others_when_not_split(self):
        comm = ht.MPI_WORLD
        called = []
        utils.serialized_write(
            comm, False, lambda: called.append("root"), lambda: called.append("other")
        )
        self.assertEqual(called, ["root"] if comm.rank == 0 else [])

    def test_serialized_write_propagates_root_error(self):
        comm = ht.MPI_WORLD

        def boom():
            raise ValueError("boom on root")

        # every process must see the failure, not just the one that raised it
        with self.assertRaises(ValueError):
            utils.serialized_write(comm, True, boom, lambda: None)

    def test_serialized_write_propagates_non_root_error(self):
        """A failure on a rank other than 0 must not deadlock the ring."""
        comm = ht.MPI_WORLD
        if comm.size < 2:
            self.skipTest("Requires at least 2 processes")
        failing_rank = comm.size - 1

        def write_other():
            if comm.rank == failing_rank:
                raise ValueError("boom on a non-root rank")

        with self.assertRaises(ValueError):
            utils.serialized_write(comm, True, lambda: None, write_other)


if __name__ == "__main__":
    unittest.main()
