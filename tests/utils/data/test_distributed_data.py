import heat as ht
from heat.utils.data.datatools import (
    DistributedDataset,
    DistributedDataLoader,
    _batch_indices,
    _create_distributed_indices,
)
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataloader, DistributedSampler as TorchDistributedSampler
import pytest

from heat.testing.basic_test import TestCase


class SeedEnvironment:
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

    @pytest.mark.parametrize("N", [0, 1, 11])
    @pytest.mark.parametrize("split", [None, 0])
    @pytest.mark.parametrize("p", [0.0, 0.6, 0.95, 1.0])
    @pytest.mark.parametrize("seed", [None, 42])
    def test_create_train_val_split(
        self, N: int, split: int | None, p: float, seed: int | None
    ) -> None:
        features = torch.arange(N * 3).reshape(N, 3)
        labels = torch.arange(N)

        X = ht.array(features, split=split)
        y = ht.array(labels, split=split)

        rng_state = torch.random.get_rng_state()
        result = ht.utils.data.create_train_val_split(X, y, p=p, seed=seed)
        assert torch.equal(torch.random.get_rng_state(), rng_state)

        X_train, X_val, y_train, y_val = result
        n_train = int(N * p)
        n_val = N - n_train

        assert X_train.gshape == (n_train, 3)
        assert X_val.gshape == (n_val, 3)
        assert y_train.gshape == (n_train,)
        assert y_val.gshape == (n_val,)

        for output, original in zip(result, (X, X, y, y)):
            assert output.split == original.split
            assert output.dtype == original.dtype
            assert output.device == original.device

        train_X, val_X, train_y, val_y = (ht.resplit(array, None).larray for array in result)

        assert torch.equal(train_X, features[train_y])
        assert torch.equal(val_X, features[val_y])

        combined = torch.cat((train_y, val_y))
        assert torch.equal(combined.sort().values, labels)

        orders = ht.MPI_WORLD.handle.allgather((train_y.tolist(), val_y.tolist()))
        assert all(order == orders[0] for order in orders)

        if seed is not None:
            repeated = ht.utils.data.create_train_val_split(X, y, p=p, seed=seed)
            for first, second in zip(result, repeated):
                assert torch.equal(first.larray, second.larray)

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

        comm = ht.get_comm()

        with SeedEnvironment(seed):
            ht_data = list(
                TorchDataloader(
                    dataset,
                    batch_size,
                    collate_fn=lambda x: x,
                    sampler=TorchDistributedSampler(dataset, num_replicas=comm.size, rank=comm.rank, drop_last=drop_last, shuffle=shuffle)
                )
            )
        with SeedEnvironment(seed):
            torch_data = list(
                TorchDataloader(
                    torch_dataset,
                    batch_size,
                    sampler=TorchDistributedSampler(dataset, num_replicas=comm.size, rank=comm.rank, drop_last=drop_last, shuffle=shuffle)
                )
            )

        for ht_batch, (torch_batch,) in zip(ht_data, torch_data):
            assert torch.allclose(ht_batch, torch_batch)

        dataset = DistributedDataset(features, labels)
        torch_dataset = TensorDataset(features_torch, labels_torch)

        with SeedEnvironment(seed):
            ht_data = list(
                TorchDataloader(
                    dataset,
                    batch_size,
                    collate_fn=lambda x: x,
                    sampler=TorchDistributedSampler(dataset, num_replicas=comm.size, rank=comm.rank, drop_last=drop_last, shuffle=shuffle)
                )
            )
        with SeedEnvironment(seed):
            torch_data = list(
                TorchDataloader(
                    torch_dataset,
                    batch_size,
                    sampler=TorchDistributedSampler(dataset, num_replicas=comm.size, rank=comm.rank, drop_last=drop_last, shuffle=shuffle)
                )
            )

        for (ht_features, ht_labels), (torch_features, torch_labels) in zip(ht_data, torch_data):
            assert torch.allclose(ht_features, torch_features)
            assert torch.allclose(ht_labels, torch_labels)

    @pytest.mark.parametrize("N", [0, 1, 11, 24])
    @pytest.mark.parametrize("split", [None, 0])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("batch_size", [3])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("prefetch_factor", [0, 1, 2])
    def test_distributed_dataloader(self, N, split, shuffle, batch_size, drop_last, prefetch_factor):
        comm = ht.get_comm()
        features_torch = torch.arange(N * 3, dtype=torch.float32).reshape(N, 3)
        labels_torch = torch.arange(N)
        features = ht.array(features_torch, split=split)
        labels = ht.array(labels_torch, split=split)
        seed = 42

        order = (
            torch.randperm(N, generator=torch.Generator().manual_seed(seed)).tolist()
            if shuffle
            else list(range(N))
        )
        if split == 0:
            samples_per_rank = N // comm.size if drop_last else -(-N // comm.size)
            order = [order[i % N] for i in range(samples_per_rank * comm.size)]
            start = comm.rank * samples_per_rank
            order = order[start : start + samples_per_rank]

        reference = list(
            TorchDataloader(
                TensorDataset(features_torch[order], labels_torch[order]),
                batch_size=batch_size,
                drop_last=drop_last,
            )
        )
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        feature_loader = DistributedDataLoader(features, **kwargs)
        label_loader = DistributedDataLoader(labels, **kwargs)
        double_loader = DistributedDataLoader(features, labels, **kwargs)

        rng_state = torch.random.get_rng_state()
        feature_batches = list(feature_loader)
        label_batches = list(label_loader)
        double_batches = list(double_loader)
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        assert len(feature_loader) == len(feature_batches) == len(reference)
        assert len(label_loader) == len(label_batches) == len(reference)
        assert len(double_loader) == len(double_batches) == len(reference)

        for features_batch, labels_batch, pair, expected in zip(
            feature_batches, label_batches, double_batches, reference
        ):
            assert torch.equal(features_batch, expected[0])
            assert torch.equal(labels_batch, expected[1])
            assert torch.equal(pair[0], expected[0])
            assert torch.equal(pair[1], expected[1])

        local_labels = torch.cat(label_batches).tolist() if label_batches else []
        rank_labels = comm.allgather(local_labels)
        assert all(len(indices) == len(local_labels) for indices in rank_labels)
        if split == 0:
            combined = [label for indices in rank_labels for label in indices]
            if drop_last:
                assert len(combined) == len(set(combined))
            else:
                assert set(combined) == set(range(N))
                assert len(combined) == samples_per_rank * comm.size

    @pytest.mark.parametrize("split", [None, 0])
    @pytest.mark.parametrize("prefetch_factor", [0, 2])
    def test_distributed_dataloader_epochs(self, split, prefetch_factor):
        comm = ht.get_comm()
        data = ht.arange(48, split=split)
        reference_seed = 42
        seed = reference_seed + comm.rank

        def make_loader():
            return DistributedDataLoader(
                data,
                batch_size=3,
                shuffle=True,
                seed=seed,
                prefetch_factor=prefetch_factor,
            )

        loader = make_loader()
        rng_state = torch.random.get_rng_state()
        first = torch.cat(list(loader))
        second = torch.cat(list(loader))
        assert loader.epoch == 2
        assert not torch.equal(first, second)
        assert torch.equal(first, torch.cat(list(make_loader())))

        expected_seed = reference_seed if data.is_distributed() else seed
        expected = torch.randperm(48, generator=torch.Generator().manual_seed(expected_seed))
        if data.is_distributed():
            expected = expected.chunk(comm.size)[comm.rank]
        assert torch.equal(first, expected)

        loader.seed = seed
        assert loader.epoch == 0
        assert torch.equal(first, torch.cat(list(loader)))
        assert loader.epoch == 1

        with iter(loader) as batches:
            next(batches)
        assert loader.epoch == 2
        assert torch.equal(torch.random.get_rng_state(), rng_state)

    @pytest.mark.parametrize("N", [0, 1, 11, 24])
    @pytest.mark.parametrize("distributed", [False, True])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_distributed_indices_partition(self, N, distributed, shuffle, drop_last):
        comm = ht.get_comm()
        reference_seed = 42
        seed = reference_seed + comm.rank
        rng_state = torch.random.get_rng_state()
        indices = _create_distributed_indices(
            comm, N, shuffle, seed, distributed=distributed, drop_last=drop_last
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        assert indices.dtype == torch.int64
        assert indices.device.type == "cpu"
        assert indices.is_contiguous()

        expected_seed = reference_seed if distributed else seed
        expected = (
            torch.randperm(N, generator=torch.Generator().manual_seed(expected_seed))
            if shuffle
            else torch.arange(N)
        )
        if not distributed:
            assert indices.shape == (1, N)
            assert torch.equal(indices[0], expected)
            return

        orders = comm.allgather(indices.tolist())
        assert all(order == orders[0] for order in orders)
        count = N // comm.size if drop_last else -(-N // comm.size)
        assert indices.shape == (comm.size, count)
        flat = indices.flatten()
        if drop_last:
            assert torch.equal(flat, expected[: count * comm.size])
            assert flat.unique().numel() == flat.numel()
        else:
            assert torch.equal(flat[:N], expected)
            assert flat.unique().numel() == N
            if N:
                assert torch.equal(flat, expected[torch.arange(flat.numel()) % N])

    def test_distributed_dataset_params(self) -> None:
        data = ht.arange(8)
        tensor = torch.arange(8)
        scalar = ht.array(1)
        distributed = ht.arange(8, split=0)
        shorter = ht.arange(7)
        unsupported_split = ht.zeros((8, 2), split=1)

        with pytest.raises(ValueError):
            DistributedDataset()
        with pytest.raises(TypeError):
            DistributedDataset(tensor)
        with pytest.raises(TypeError):
            DistributedDataset(data, tensor)
        with pytest.raises(ValueError):
            DistributedDataset(scalar)
        with pytest.raises(ValueError):
            DistributedDataset(data, distributed)
        with pytest.raises(ValueError):
            DistributedDataset(data, shorter)
        with pytest.raises(ValueError):
            DistributedDataset(unsupported_split)
        with pytest.raises(TypeError):
            DistributedDataset(data, post_processing="invalid")
        with pytest.raises(TypeError):
            DistributedDataset(data, comm="invalid")

    def test_distributed_dataloader_params(self) -> None:
        data = ht.arange(8)
        tensor = torch.arange(8)
        scalar = ht.array(1)
        distributed = ht.arange(8, split=0)
        shorter = ht.arange(7)
        unsupported_split = ht.zeros((8, 2), split=1)

        with pytest.raises(ValueError):
            DistributedDataLoader(batch_size=2)
        with pytest.raises(TypeError):
            DistributedDataLoader(tensor, batch_size=2)
        with pytest.raises(TypeError):
            DistributedDataLoader(data, tensor, batch_size=2)
        with pytest.raises(ValueError):
            DistributedDataLoader(scalar, batch_size=2)
        with pytest.raises(ValueError):
            DistributedDataLoader(data, distributed, batch_size=2)
        with pytest.raises(ValueError):
            DistributedDataLoader(data, shorter, batch_size=2)
        with pytest.raises(ValueError):
            DistributedDataLoader(unsupported_split, batch_size=2)
        with pytest.raises(TypeError):
            DistributedDataLoader(data, batch_size=2, post_processing="invalid")
        with pytest.raises(TypeError):
            DistributedDataLoader(data, batch_size=2, comm="invalid")

        for batch_size in (0, -1, 1.5, "invalid", None, True):
            with pytest.raises(ValueError):
                DistributedDataLoader(data, batch_size=batch_size)

        for prefetch_factor in (-1, 1.5, "invalid", None):
            with pytest.raises(ValueError):
                DistributedDataLoader(data, batch_size=2, prefetch_factor=prefetch_factor)

        for drop_last in (0, "invalid", None):
            with pytest.raises(ValueError):
                DistributedDataLoader(data, batch_size=2, drop_last=drop_last)

        loader = DistributedDataLoader(data, batch_size=2)
        for seed in (1.5, "invalid", None, True):
            with pytest.raises(ValueError):
                DistributedDataLoader(data, batch_size=2, seed=seed)
            with pytest.raises(ValueError):
                loader.seed = seed

    def test_batch_indices(self) -> None:
        for indices in (torch.tensor(0), torch.arange(8)):
            with pytest.raises(IndexError):
                list(_batch_indices(indices, batch_size=2, drop_last=False))

        indices = torch.arange(8).reshape(1, 8)
        with pytest.raises(ValueError):
            list(_batch_indices(indices, batch_size=0, drop_last=False))
        with pytest.raises(TypeError):
            list(_batch_indices(indices, batch_size=1.5, drop_last=False))

    def test_create_distributed_indices(self) -> None:
        for shuffle in (False, True):
            with pytest.raises(RuntimeError):
                _create_distributed_indices(
                    ht.MPI_WORLD, num_elements=-1, shuffle=shuffle, seed=0, distributed=False
                )

        with pytest.raises(RuntimeError):
            _create_distributed_indices(
                ht.MPI_WORLD, num_elements=8, shuffle=True, seed="invalid", distributed=False
            )
        with pytest.raises(AttributeError):
            _create_distributed_indices(
                object(), num_elements=8, shuffle=False, seed=0, distributed=False
            )
