import pytest
import heat as ht
import torch
import os

from pathlib import Path

HDF5_PATH = str(Path(ht.__file__).parent / "datasets" / "iris.h5")
USE_GPU = torch.cuda.is_available() and os.getenv("HEAT_TEST_USE_GPU") == "gpu"

@pytest.mark.skipif(torch.cuda.is_available() and torch.version.hip, reason="Not supported for ROCM/HIP")
@pytest.mark.skipif(not ht.supports_hdf5(), reason="Requires HDF5")
class TestPartialDataset:
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
        full_data = ht.load(HDF5_PATH, dataset="data", split=None)

        # Test basic initialization
        initial_load = 30
        load_length = 20
        partial_dset = self._create_test_dataset(HDF5_PATH, full_data.comm, initial_load, load_length)
        assert partial_dset.total_size == full_data.shape[0]

        rows = full_data.shape[0]
        if initial_load > rows // full_data.comm.size:
            assert partial_dset.partial_dataset == False
        else:
            assert partial_dset.partial_dataset == True

    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_batch_shape_consistency(self, pin_memory):
        """Test that all batches have the expected shape across device configurations."""
        full_data = ht.load(HDF5_PATH, dataset="data", split=None)
        expected_batch_shape = (7, 4)

        # Test with different device configurations

        partial_dset = self._create_test_dataset(
            HDF5_PATH, full_data.comm, 30, 20, use_gpu=USE_GPU
        )
        dl = ht.utils.data.DataLoader(
            dataset=partial_dset,
            batch_size=7,
            pin_memory=pin_memory,
        )

        for batch in dl:
            assert batch.shape == expected_batch_shape
            break  # Just check first batch for this test

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_consecutive_batches_differ(self, pin_memory):
        """Test that consecutive batches within an epoch are different."""
        full_data = ht.load(HDF5_PATH, dataset="data", split=None)

        partial_dset = self._create_test_dataset(
            HDF5_PATH, full_data.comm, 30, 20, use_gpu=USE_GPU
        )
        dl = ht.utils.data.DataLoader(
            dataset=partial_dset,
            batch_size=7,
            pin_memory=pin_memory,
        )

        last_batch = None
        batch_count = 0
        for batch in dl:
            if last_batch is not None:
                assert not torch.allclose(last_batch, batch)
            last_batch = batch
            batch_count += 1
            if batch_count >= 3:  # Only check first few batches
                break

    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_element_count_per_epoch(self, pin_memory):
        """Test that the total element count is within expected bounds."""
        full_data = ht.load(HDF5_PATH, dataset="data", split=None)
        target_shape = full_data.shape

        partial_dset = self._create_test_dataset(
            HDF5_PATH, full_data.comm, 30, 20, use_gpu=USE_GPU
        )
        dl = ht.utils.data.DataLoader(
            dataset=partial_dset,
            batch_size=7,
            pin_memory=pin_memory,
        )

        elems = 0
        for batch in dl:
            elems += batch.shape[0]

        expected_min = (target_shape[0] - 7) // full_data.comm.size
        assert elems >= expected_min, f"Element count {elems} should be >= {expected_min}"

    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_data_varies_between_epochs(self, pin_memory):
        """Test that data differs between consecutive epochs due to shuffling."""
        full_data = ht.load(HDF5_PATH, dataset="data", split=None)

        partial_dset = self._create_test_dataset(
            HDF5_PATH, full_data.comm, 30, 20, use_gpu=USE_GPU
        )
        dl = ht.utils.data.DataLoader(
            dataset=partial_dset,
            batch_size=7,
            pin_memory=pin_memory,
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
        assert len(epoch_data) == 2, f"Expected data from 2 epochs, got {len(epoch_data)}"

        # Check that the two epochs have different data
        assert not torch.allclose(epoch_data[0], epoch_data[1]), "Data should vary between epochs"

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize("pin_memory", [False, True])
    def test_partial_h5_dataset_integration(self, pin_memory):
        """Integration test: verify the complete workflow with both CPU and GPU."""
        full_data = ht.load(HDF5_PATH, dataset="data", split=None)
        target_shape = full_data.shape

        partial_dset = self._create_test_dataset(
            HDF5_PATH, full_data.comm, 30, 20, use_gpu=USE_GPU
        )
        dl = ht.utils.data.DataLoader(
            dataset=partial_dset,
            batch_size=7,
            pin_memory=pin_memory,
        )

        for epoch in range(2):
            elems = 0
            last_batch = None
            for batch in dl:
                elems += batch.shape[0]
                # Check batch shape
                assert batch.shape == (7, 4), f"Expected batch shape (7, 4), got {batch.shape}"

                # Check that consecutive batches are different
                if last_batch is not None:
                    assert not torch.allclose(last_batch, batch), "Consecutive batches should differ"
                last_batch = batch

            # Check element count
            expected_min = (target_shape[0] - 7) // full_data.comm.size
            assert elems >= expected_min, f"Element count {elems} should be >= {expected_min}"
