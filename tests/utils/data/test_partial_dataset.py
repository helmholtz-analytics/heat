import heat as ht
import torch
import unittest
from flaky import flaky

from pathlib import Path


@unittest.skipIf(torch.cuda.is_available() and torch.version.hip, "not supported for HIP")
@unittest.skipUnless(ht.supports_hdf5(), "Requires HDF5")
@flaky
class TestPartialDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.HDF5_PATH = str(Path(ht.__file__).parent / "datasets" / "iris.h5")

    def _create_test_dataset(self, file, comm, initial_load, load_length, use_gpu=False):
        """Helper method to create a TestDataset instance."""
        class TestDataset(ht.utils.data.partial_dataset.PartialH5Dataset):
            def __init__(self, file, comm, load, load_len, use_gpus=False):
                super(TestDataset, self).__init__(
                    file, comm=comm, initial_load=load, load_length=load_len, use_gpu=use_gpus
                )

            def __getitem__(self, item):
                return self.data[item]

        return TestDataset(file, comm, initial_load, load_length, use_gpu)

    def test_dataset_initialization(self):
        """Test that PartialH5Dataset initializes correctly."""
        full_data = ht.load(self.HDF5_PATH, dataset="data", split=None)

        # Test basic initialization
        partial_dset = self._create_test_dataset(self.HDF5_PATH, full_data.comm, 30, 20)
        self.assertEqual(partial_dset.total_size, full_data.shape[0])
        self.assertTrue(partial_dset.partial_dataset)

    def test_batch_shape_consistency(self):
        """Test that all batches have the expected shape across device configurations."""
        full_data = ht.load(self.HDF5_PATH, dataset="data", split=None)
        target_shape = full_data.shape
        expected_batch_shape = (7, 4)

        # Test with different device configurations
        device_configs = [
            {"use_gpu": False, "pin_memory": False},
        ]
        if torch.cuda.is_available():
            device_configs.append({"use_gpu": True, "pin_memory": True})

        for config in device_configs:
            with self.subTest(use_gpu=config["use_gpu"], pin_memory=config["pin_memory"]):
                partial_dset = self._create_test_dataset(
                    self.HDF5_PATH, full_data.comm, 30, 20, use_gpu=config["use_gpu"]
                )
                dl = ht.utils.data.DataLoader(
                    dataset=partial_dset,
                    batch_size=7,
                    pin_memory=config["pin_memory"],
                )

                for batch in dl:
                    self.assertEqual(batch.shape, expected_batch_shape)
                    break  # Just check first batch for this test

    def test_consecutive_batches_differ(self):
        """Test that consecutive batches within an epoch are different."""
        full_data = ht.load(self.HDF5_PATH, dataset="data", split=None)

        device_configs = [
            {"use_gpu": False, "pin_memory": False},
        ]
        if torch.cuda.is_available():
            device_configs.append({"use_gpu": True, "pin_memory": True})

        for config in device_configs:
            with self.subTest(use_gpu=config["use_gpu"]):
                partial_dset = self._create_test_dataset(
                    self.HDF5_PATH, full_data.comm, 30, 20, use_gpu=config["use_gpu"]
                )
                dl = ht.utils.data.DataLoader(
                    dataset=partial_dset,
                    batch_size=7,
                    pin_memory=config["pin_memory"],
                )

                last_batch = None
                batch_count = 0
                for batch in dl:
                    if last_batch is not None:
                        self.assertFalse(
                            torch.allclose(last_batch, batch),
                            "Consecutive batches should differ"
                        )
                    last_batch = batch
                    batch_count += 1
                    if batch_count >= 3:  # Only check first few batches
                        break

    def test_element_count_per_epoch(self):
        """Test that the total element count is within expected bounds."""
        full_data = ht.load(self.HDF5_PATH, dataset="data", split=None)
        target_shape = full_data.shape

        device_configs = [
            {"use_gpu": False, "pin_memory": False},
        ]
        if torch.cuda.is_available():
            device_configs.append({"use_gpu": True, "pin_memory": True})

        for config in device_configs:
            with self.subTest(use_gpu=config["use_gpu"]):
                partial_dset = self._create_test_dataset(
                    self.HDF5_PATH, full_data.comm, 30, 20, use_gpu=config["use_gpu"]
                )
                dl = ht.utils.data.DataLoader(
                    dataset=partial_dset,
                    batch_size=7,
                    pin_memory=config["pin_memory"],
                )

                elems = 0
                for batch in dl:
                    elems += batch.shape[0]

                expected_min = (target_shape[0] - 7) // full_data.comm.size
                self.assertGreaterEqual(
                    elems, expected_min,
                    f"Element count {elems} should be >= {expected_min}"
                )

    def test_data_varies_between_epochs(self):
        """Test that data differs between consecutive epochs due to shuffling."""
        full_data = ht.load(self.HDF5_PATH, dataset="data", split=None)

        device_configs = [
            {"use_gpu": False, "pin_memory": False},
        ]
        if torch.cuda.is_available():
            device_configs.append({"use_gpu": True, "pin_memory": True})

        for config in device_configs:
            with self.subTest(use_gpu=config["use_gpu"]):
                partial_dset = self._create_test_dataset(
                    self.HDF5_PATH, full_data.comm, 30, 20, use_gpu=config["use_gpu"]
                )
                dl = ht.utils.data.DataLoader(
                    dataset=partial_dset,
                    batch_size=7,
                    pin_memory=config["pin_memory"],
                )

                epoch_data = []
                for epoch in range(2):
                    epoch_batches = None
                    for batch in dl:
                        if epoch_batches is None:
                            epoch_batches = batch
                        else:
                            epoch_batches = torch.cat((epoch_batches, batch), dim=0)
                    epoch_data.append(epoch_batches)

                # Ensure we collected data from both epochs
                self.assertEqual(len(epoch_data), 2)

                # Check that the two epochs have different data
                self.assertFalse(
                    torch.allclose(epoch_data[0], epoch_data[1]),
                    "Data should vary between epochs"
                )

    def test_partial_h5_dataset_integration(self):
        """Integration test: verify the complete workflow with both CPU and GPU."""
        full_data = ht.load(self.HDF5_PATH, dataset="data", split=None)
        target_shape = full_data.shape

        device_configs = [
            {"use_gpu": False, "pin_memory": False},
        ]
        if torch.cuda.is_available():
            device_configs.append({"use_gpu": True, "pin_memory": True})

        for config in device_configs:
            with self.subTest(use_gpu=config["use_gpu"], pin_memory=config["pin_memory"]):
                partial_dset = self._create_test_dataset(
                    self.HDF5_PATH, full_data.comm, 30, 20, use_gpu=config["use_gpu"]
                )
                dl = ht.utils.data.DataLoader(
                    dataset=partial_dset,
                    batch_size=7,
                    pin_memory=config["pin_memory"],
                )

                for epoch in range(2):
                    elems = 0
                    last_batch = None
                    for batch in dl:
                        elems += batch.shape[0]
                        # Check batch shape
                        self.assertEqual(batch.shape, (7, 4))

                        # Check that consecutive batches are different
                        if last_batch is not None:
                            self.assertFalse(torch.allclose(last_batch, batch))
                        last_batch = batch

                    # Check element count
                    expected_min = (target_shape[0] - 7) // full_data.comm.size
                    self.assertGreaterEqual(elems, expected_min)
