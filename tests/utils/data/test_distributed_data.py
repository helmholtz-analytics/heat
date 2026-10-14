import heat as ht
from heat.utils.data.datatools import DistributedDataset, DistributedDataLoader, _batch_indices, _create_distributed_indices
import torch
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataloader
import pytest

from heat.testing.basic_test import TestCase

class SeedEnviroment:
    """
    Class to be used in a `with` Environment.
    Changes the torch seed to the given and then resets it to the previous one when exiting.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def __enter__(self):
        self.state = torch.random.get_rng_state()

        if self.seed is not None:
            torch.random.manual_seed(self.seed)

    def __exit__(self, *args, **kwargs):
        torch.random.set_rng_state(self.state)


class TestDatatools:
    @classmethod
    def setup_class(cls):
        TestCase.setUpClass()

    @pytest.mark.parametrize("N", [20])
    @pytest.mark.parametrize("split", [None, 0])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("batch_size", [3])
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_distributed_dataset(self, N, split, shuffle, batch_size, drop_last):

        features = ht.random.randn(N, 3, 244, 244, dtype=ht.float32, split=split)
        labels = ht.arange(N, split=split).reshape(N, 1)

        features_torch = ht.resplit(features, None).larray
        labels_torch = ht.resplit(labels, None).larray

        dataset = DistributedDataset(features)
        torch_dataset = TensorDataset(features_torch)

        seed = torch.Generator().initial_seed()

        with SeedEnviroment(seed):
            ht_data = list(TorchDataloader(dataset, batch_size, shuffle=shuffle, collate_fn=lambda x: x, drop_last=drop_last))
        with SeedEnviroment(seed):
            torch_data = list(TorchDataloader(torch_dataset, batch_size, shuffle=shuffle, drop_last=drop_last))

        for ht_batch, (torch_batch, ) in zip(ht_data, torch_data):
            print(type(ht_batch), type(torch_batch))
            assert torch.allclose(ht_batch, torch_batch)

        dataset = DistributedDataset(features, labels)
        torch_dataset = TensorDataset(features_torch, labels_torch)


        with SeedEnviroment(seed):
            ht_data = list(TorchDataloader(dataset, batch_size, shuffle=shuffle, collate_fn=lambda x: x, drop_last=drop_last))
        with SeedEnviroment(seed):
            torch_data = list(TorchDataloader(torch_dataset, batch_size, shuffle=shuffle, drop_last=drop_last))

        for (ht_features, ht_labels), (torch_features, torch_labels) in zip(ht_data, torch_data):
            print(type(ht_batch), type(torch_batch))
            assert torch.allclose(ht_features, torch_features)
            assert torch.allclose(ht_labels, torch_labels)

    @pytest.mark.parametrize("N", [20])
    @pytest.mark.parametrize("split", [None, 0])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("batch_size", [3])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("prefetch_factor", [0, 1, 2])
    def test_distributed_dataloader(self, N, split, shuffle, batch_size, drop_last, prefetch_factor):
        features = ht.random.randn(N, 3, 244, 244, dtype=ht.float32, split=split)
        labels = ht.arange(N, split=split).reshape(N, 1)

        features_torch = ht.resplit(features, None).larray
        labels_torch = ht.resplit(labels, None).larray

        feature_data = list(TorchDataloader(TensorDataset(features_torch), batch_size, shuffle=False, drop_last=drop_last))
        labels_data = list(TorchDataloader(TensorDataset(labels_torch), batch_size, shuffle=False, drop_last=drop_last))

        feature_ht_data = list(DistributedDataLoader(features, batch_size=batch_size, shuffle=False, drop_last=drop_last, prefetch_factor=prefetch_factor, seed=0))
        labels_ht_data = list(DistributedDataLoader(labels, batch_size=batch_size, shuffle=False, drop_last=drop_last, prefetch_factor=prefetch_factor, seed=0))

        for feature_batch, (torch_batch,) in zip(feature_ht_data, feature_data):
            assert torch.allclose(feature_batch, torch_batch)

        for labels_batch, (torch_batch,) in zip(labels_ht_data, labels_data):
            assert torch.allclose(labels_batch, torch_batch)

        seed = torch.Generator().initial_seed()
        labels_ht_data = list(DistributedDataLoader(labels, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, prefetch_factor=prefetch_factor, seed=seed))
        double_loader = DistributedDataLoader(features, labels, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, prefetch_factor=prefetch_factor, seed=seed)

        for (feature_batch, label_batch), indices in zip(double_loader, labels_ht_data):
            assert torch.allclose(label_batch, indices)
            assert torch.allclose(feature_batch, features_torch[indices.flatten()])

    def test_distributed_dataset_params(self):
        ...

    def test_distributed_dataloader_params(self):
        ...

    def test_batch_indices(self):
        ...

    def test_create_distributed_indices(self):
        ...