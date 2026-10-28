import heat as ht
from heat.utils.data.datatools import (
    DistributedDataset,
    DistributedDataLoader,
    _batch_indices,
    _create_distributed_indices,
    _sanitize_communicators,
)
import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader as TorchDataloader,
    DistributedSampler as TorchDistributedSampler,
)
import pytest

from heat.testing.basic_test import TestCase


# Sample axis, feature split axis, and label split axis.
LAYOUTS = [(0, None, None), (0, 0, 0), (1, 1, 0), (1, 0, 0), (-1, 0, None), (0, None, 0)]


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

    @pytest.mark.parametrize("N", [1, 11])
    @pytest.mark.parametrize("sample_axis,feature_split,label_split", LAYOUTS)
    @pytest.mark.parametrize("p", [0.0, 0.6, 0.95, 1.0])
    @pytest.mark.parametrize("seed", [None, 42])
    def test_create_train_val_split(
        self, N, sample_axis, feature_split, label_split, p, seed
    ) -> None:
        features = torch.arange(N * 3).reshape(N, 3)
        labels = torch.arange(N)

        X = ht.array(features.movedim(0, sample_axis), split=feature_split)
        y = ht.array(labels, split=label_split)
        original_X, original_y = X.larray.clone(), y.larray.clone()
        axis_kwargs = {} if sample_axis == 0 else {"sample_axis": (sample_axis, 0)}

        rng_state = torch.random.get_rng_state()
        result = ht.utils.data.create_train_val_split(X, y, p=p, seed=seed, **axis_kwargs)
        assert torch.equal(torch.random.get_rng_state(), rng_state)

        X_train, X_val, y_train, y_val = result
        n_train = int(N * p)
        n_val = N - n_train

        assert X_train.gshape == tuple(features[:n_train].movedim(0, sample_axis).shape)
        assert X_val.gshape == tuple(features[:n_val].movedim(0, sample_axis).shape)
        assert y_train.gshape == (n_train,)
        assert y_val.gshape == (n_val,)

        for output, original in zip(result, (X, X, y, y)):
            assert output.split == original.split
            assert output.dtype == original.dtype
            assert output.device == original.device
            assert output.comm is original.comm

        train_X, val_X, train_y, val_y = (ht.resplit(array, None).larray for array in result)
        train_X = train_X.movedim(sample_axis, 0)
        val_X = val_X.movedim(sample_axis, 0)

        assert torch.equal(train_X, features[train_y])
        assert torch.equal(val_X, features[val_y])

        combined = torch.cat((train_y, val_y))
        assert torch.equal(combined.sort().values, labels)

        orders = ht.MPI_WORLD.handle.allgather((train_y.tolist(), val_y.tolist()))
        assert all(order == orders[0] for order in orders)

        if seed is not None:
            repeated = ht.utils.data.create_train_val_split(X, y, p=p, seed=seed, **axis_kwargs)
            for first, second in zip(result, repeated):
                assert torch.equal(first.larray, second.larray)
        assert X.split == feature_split
        assert y.split == label_split
        assert torch.equal(X.larray, original_X)
        assert torch.equal(y.larray, original_y)

    @pytest.mark.parametrize("N", [20])
    @pytest.mark.parametrize("sample_axis,feature_split,label_split", LAYOUTS)
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("batch_size", [3])
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_distributed_dataset(
        self, N, sample_axis, feature_split, label_split, shuffle, batch_size, drop_last
    ):

        features = ht.random.randn(N, 3, 244, 244, dtype=ht.float32, split=feature_split)
        labels = ht.arange(N, split=label_split).reshape(N, 1)

        features_torch = ht.resplit(features, None).larray
        labels_torch = ht.resplit(labels, None).larray
        features = ht.moveaxis(features, 0, sample_axis)
        # Set the requested physical split after moving the sample axis.
        if features.split != feature_split:
            features.resplit_(feature_split)
        original_features, original_labels = features.larray.clone(), labels.larray.clone()

        dataset = DistributedDataset(features, sample_axis=sample_axis)
        torch_dataset = TensorDataset(features_torch)

        seed = torch.Generator().initial_seed()

        comm = ht.get_comm()

        with SeedEnvironment(seed):
            ht_data = list(
                TorchDataloader(
                    dataset,
                    batch_size,
                    collate_fn=lambda x: x,
                    sampler=TorchDistributedSampler(
                        dataset,
                        num_replicas=comm.size,
                        rank=comm.rank,
                        drop_last=drop_last,
                        shuffle=shuffle,
                    ),
                )
            )
        with SeedEnvironment(seed):
            torch_data = list(
                TorchDataloader(
                    torch_dataset,
                    batch_size,
                    sampler=TorchDistributedSampler(
                        dataset,
                        num_replicas=comm.size,
                        rank=comm.rank,
                        drop_last=drop_last,
                        shuffle=shuffle,
                    ),
                )
            )

        for ht_batch, (torch_batch,) in zip(ht_data, torch_data):
            assert torch.allclose(ht_batch, torch_batch)

        dataset = DistributedDataset(features, labels, sample_axis=(sample_axis, 0))
        torch_dataset = TensorDataset(features_torch, labels_torch)

        with SeedEnvironment(seed):
            ht_data = list(
                TorchDataloader(
                    dataset,
                    batch_size,
                    collate_fn=lambda x: x,
                    sampler=TorchDistributedSampler(
                        dataset,
                        num_replicas=comm.size,
                        rank=comm.rank,
                        drop_last=drop_last,
                        shuffle=shuffle,
                    ),
                )
            )
        with SeedEnvironment(seed):
            torch_data = list(
                TorchDataloader(
                    torch_dataset,
                    batch_size,
                    sampler=TorchDistributedSampler(
                        dataset,
                        num_replicas=comm.size,
                        rank=comm.rank,
                        drop_last=drop_last,
                        shuffle=shuffle,
                    ),
                )
            )

        for (ht_features, ht_labels), (torch_features, torch_labels) in zip(ht_data, torch_data):
            assert torch.allclose(ht_features, torch_features)
            assert torch.allclose(ht_labels, torch_labels)
        assert features.split == feature_split
        assert labels.split == label_split
        assert torch.equal(features.larray, original_features)
        assert torch.equal(labels.larray, original_labels)

    @pytest.mark.parametrize("N", [1, 11, 24])
    @pytest.mark.parametrize("sample_axis,feature_split,label_split", LAYOUTS)
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("batch_size", [3])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("prefetch_factor", [0, 1, 2])
    def test_distributed_dataloader(
        self,
        N,
        sample_axis,
        feature_split,
        label_split,
        shuffle,
        batch_size,
        drop_last,
        prefetch_factor,
    ):
        features_torch = torch.arange(N * 3, dtype=torch.float32).reshape(N, 3)
        labels_torch = torch.arange(N)
        features = ht.array(features_torch.movedim(0, sample_axis), split=feature_split)
        labels = ht.array(labels_torch, split=label_split)
        original_features, original_labels = features.larray.clone(), labels.larray.clone()
        seed = 42

        comm = _sanitize_communicators(features.comm, labels.comm)

        if shuffle:
            order = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).tolist()
        else:
            order = list(range(N))
        distributed = features.is_distributed() or labels.is_distributed()

        if drop_last:
            samples_per_rank = N // comm.size
        else:
            samples_per_rank = (N + comm.size - 1) // comm.size

        distributed_order = order
        if distributed:
            total_samples = samples_per_rank * comm.size
            if drop_last:
                adjusted_order = order[:total_samples]
            else:
                adjusted_order = [order[i % N] for i in range(total_samples)]
            start = comm.rank * samples_per_rank
            distributed_order = adjusted_order[start : start + samples_per_rank]

        feature_order = order
        if features.is_distributed():
            feature_order = distributed_order
        label_order = order
        if labels.is_distributed():
            label_order = distributed_order

        feature_reference = list(
            TorchDataloader(
                features_torch[feature_order], batch_size=batch_size, drop_last=drop_last
            )
        )
        label_reference = list(
            TorchDataloader(labels_torch[label_order], batch_size=batch_size, drop_last=drop_last)
        )
        # The paired loader partitions both inputs whenever either is distributed.
        pair_reference = list(
            TorchDataloader(
                TensorDataset(features_torch[distributed_order], labels_torch[distributed_order]),
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
        feature_loader = DistributedDataLoader(features, sample_axis=sample_axis, **kwargs)
        label_loader = DistributedDataLoader(labels, **kwargs)
        double_loader = DistributedDataLoader(
            features, labels, sample_axis=(sample_axis, 0), **kwargs
        )

        rng_state = torch.random.get_rng_state()
        feature_batches = list(feature_loader)
        label_batches = list(label_loader)
        double_batches = list(double_loader)
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        assert len(feature_loader) == len(feature_batches) == len(feature_reference)
        assert len(label_loader) == len(label_batches) == len(label_reference)
        assert len(double_loader) == len(double_batches) == len(pair_reference)

        for features_batch, expected in zip(feature_batches, feature_reference):
            assert torch.equal(features_batch, expected)
        for labels_batch, expected in zip(label_batches, label_reference):
            assert torch.equal(labels_batch, expected)
        for (features_batch, labels_batch), (expected_features, expected_labels) in zip(
            double_batches, pair_reference
        ):
            assert torch.equal(features_batch, expected_features)
            assert torch.equal(labels_batch, expected_labels)

        local_labels = (
            torch.cat([pair[1] for pair in double_batches]).tolist() if double_batches else []
        )
        rank_labels = comm.allgather(local_labels)
        assert all(len(indices) == len(local_labels) for indices in rank_labels)
        if distributed:
            combined = [label for indices in rank_labels for label in indices]
            if drop_last:
                assert len(combined) == len(set(combined))
            else:
                assert set(combined) == set(range(N))
                assert len(combined) == samples_per_rank * comm.size
        assert features.split == feature_split
        assert labels.split == label_split
        assert torch.equal(features.larray, original_features)
        assert torch.equal(labels.larray, original_labels)

    @pytest.mark.parametrize("split", [None, 0])
    @pytest.mark.parametrize("prefetch_factor", [0, 2])
    def test_distributed_dataloader_epochs(self, split, prefetch_factor):
        data = ht.arange(48, split=split)
        comm = data.comm
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

    @pytest.mark.parametrize("N", [1, 11, 24])
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
        feature_split = ht.zeros((8, 2), split=1)

        with pytest.raises(ValueError):
            DistributedDataset()
        with pytest.raises(TypeError):
            DistributedDataset(tensor)
        with pytest.raises(TypeError):
            DistributedDataset(data, tensor)
        with pytest.raises(ValueError):
            DistributedDataset(scalar)
        assert len(DistributedDataset(data, distributed)) == 8
        with pytest.raises(ValueError):
            DistributedDataset(data, shorter)
        assert len(DistributedDataset(feature_split)) == 8
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
        feature_split = ht.zeros((8, 2), split=1)

        with pytest.raises(ValueError):
            DistributedDataLoader(batch_size=2)
        with pytest.raises(TypeError):
            DistributedDataLoader(tensor, batch_size=2)
        with pytest.raises(TypeError):
            DistributedDataLoader(data, tensor, batch_size=2)
        with pytest.raises(ValueError):
            DistributedDataLoader(scalar, batch_size=2)
        DistributedDataLoader(data, distributed, batch_size=2)
        with pytest.raises(ValueError):
            DistributedDataLoader(data, shorter, batch_size=2)
        DistributedDataLoader(feature_split, batch_size=2)
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

    @pytest.mark.parametrize(
        "factory", [DistributedDataset, DistributedDataLoader, ht.utils.data.create_train_val_split]
    )
    @pytest.mark.parametrize(
        "sample_axis,error",
        [
            (True, TypeError),
            (1.5, TypeError),
            (None, TypeError),
            ([0], TypeError),
            ((False,), TypeError),
            ((0.5,), TypeError),
            (1, ValueError),
            (-2, ValueError),
            ((), ValueError),
            ((0, 0), ValueError),
        ],
    )
    def test_sample_axis_params(self, factory, sample_axis, error):
        kwargs = {"batch_size": 2} if factory is DistributedDataLoader else {}
        with pytest.raises(error):
            factory(ht.arange(8), sample_axis=sample_axis, **kwargs)

    @pytest.mark.parametrize(
        "factory", [DistributedDataset, DistributedDataLoader, ht.utils.data.create_train_val_split]
    )
    def test_sample_count_mismatch(self, factory):
        kwargs = {"batch_size": 2} if factory is DistributedDataLoader else {}
        with pytest.raises(ValueError):
            factory(ht.zeros((3, 8)), ht.arange(7), sample_axis=(1, 0), **kwargs)

    @pytest.mark.parametrize(
        "factory", [DistributedDataset, DistributedDataLoader, ht.utils.data.create_train_val_split]
    )
    @pytest.mark.parametrize("sample_axis,feature_split,label_split", LAYOUTS)
    def test_empty_input(self, factory, sample_axis, feature_split, label_split):
        """Reject zero global samples regardless of their axes or distribution."""
        features = ht.array(torch.empty((0, 3)).movedim(0, sample_axis), split=feature_split)
        labels = ht.arange(0, split=label_split)
        kwargs = {"batch_size": 2} if factory is DistributedDataLoader else {}
        with pytest.raises(ValueError, match="at least one sample"):
            factory(features, labels, sample_axis=(sample_axis, 0), **kwargs)

    def test_sample_axes_custom_comm(self, monkeypatch):
        world = ht.MPI_WORLD
        comm = world.Split(world.rank % 2, key=-world.rank)
        try:
            rows = torch.arange(11 * 3 * 2).reshape(11, 3, 2)
            features = ht.array(rows.movedim(0, -1), split=0, comm=comm)
            labels = ht.arange(11, comm=comm)
            original_features = features.larray.clone()
            original_labels = labels.larray.clone()

            def post_processing(i, batch):
                assert batch.shape[1:] == ((3, 2) if i == 0 else ())
                return batch + 100 * i

            dataset = DistributedDataset(
                features, labels, sample_axis=(-1, 0), post_processing=post_processing
            )
            indices = (torch.arange(2) + 2 * comm.rank) % 11
            batch, targets = dataset.__getitems__(indices.tolist())
            assert torch.equal(batch.cpu(), rows[indices])
            assert torch.equal(targets.cpu(), indices + 100)
            empty_batch, empty_targets = dataset.__getitems__([])
            assert empty_batch.shape == (0, 3, 2)
            assert empty_targets.shape == (0,)
            assert len(dataset) == 11

            loader = DistributedDataLoader(
                features,
                labels,
                sample_axis=(-1, 0),
                batch_size=2,
                shuffle=True,
                seed=42 + comm.rank,
                prefetch_factor=2,
                post_processing=post_processing,
            )
            order = torch.randperm(11, generator=torch.Generator().manual_seed(42))
            count = -(-11 // comm.size)
            padded = order[torch.arange(count * comm.size) % 11]
            expected = padded[comm.rank * count : (comm.rank + 1) * count]
            batches = list(loader)
            assert len(batches) == len(loader)
            assert torch.equal(torch.cat([b[0] for b in batches]).cpu(), rows[expected])
            assert torch.equal(torch.cat([b[1] for b in batches]).cpu(), expected + 100)

            monkeypatch.setattr(
                "heat.utils.data.datatools.random.randint", lambda low, high: 100 + world.rank
            )
            seed = comm.bcast(100 + world.rank, root=0)
            order = torch.randperm(11, generator=torch.Generator().manual_seed(seed))
            parts = ht.utils.data.create_train_val_split(
                features, labels, sample_axis=(-1, 0), p=0.6
            )
            for array, original in zip(parts, (features, features, labels, labels)):
                assert array.comm is comm
                assert array.split == original.split
            train, val, train_labels, val_labels = (ht.resplit(a, None).larray for a in parts)
            assert torch.equal(train.movedim(-1, 0).cpu(), rows[order[:6]])
            assert torch.equal(val.movedim(-1, 0).cpu(), rows[order[6:]])
            assert torch.equal(train_labels.cpu(), order[:6])
            assert torch.equal(val_labels.cpu(), order[6:])

            single_train, single_val = ht.utils.data.create_train_val_split(
                features, sample_axis=-1, p=0.6, seed=seed
            )
            assert torch.equal(single_train.larray, parts[0].larray)
            assert torch.equal(single_val.larray, parts[1].larray)
            assert features.gshape == (3, 2, 11)
            assert features.split == 0
            assert labels.split is None
            assert torch.equal(features.larray, original_features)
            assert torch.equal(labels.larray, original_labels)
        finally:
            comm.Free()

    def test_batch_indices(self) -> None:
        for indices in (torch.tensor(0), torch.arange(8)):
            with pytest.raises(IndexError):
                list(_batch_indices(indices, batch_size=2, drop_last=False))

        indices = torch.arange(8).reshape(1, 8)
        with pytest.raises(ValueError):
            list(_batch_indices(indices, batch_size=0, drop_last=False))
        with pytest.raises(TypeError):
            list(_batch_indices(indices, batch_size=1.5, drop_last=False))

    @pytest.mark.parametrize("num_elements", [0, -1])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("distributed", [False, True])
    def test_create_distributed_indices_nonpositive(self, num_elements, shuffle, distributed):
        """Reject nonpositive counts before accessing even an invalid communicator."""
        for comm in (ht.MPI_WORLD, object()):
            with pytest.raises(ValueError, match="greater than 0"):
                _create_distributed_indices(
                    comm,
                    num_elements=num_elements,
                    shuffle=shuffle,
                    seed=0,
                    distributed=distributed,
                )

    def test_create_distributed_indices(self) -> None:
        with pytest.raises(RuntimeError):
            _create_distributed_indices(
                ht.MPI_WORLD, num_elements=8, shuffle=True, seed="invalid", distributed=False
            )

        # Local index creation does not access the communicator.
        indices = _create_distributed_indices(
            object(), num_elements=8, shuffle=False, seed=0, distributed=False
        )
        assert torch.equal(indices, torch.arange(8).unsqueeze(0))

        with pytest.raises(AttributeError):
            _create_distributed_indices(
                object(), num_elements=8, shuffle=False, seed=0, distributed=True
            )
