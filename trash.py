from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator
import time
import numpy as np
import torch
from tqdm import tqdm

from mpi4py import MPI


import heat as ht


class DistributedDataset(torch.utils.data.Dataset):
    def __init__(self, a: ht.DNDarray) -> None:
        self.__a = a
        self.__comm: ht.MPICommunication = a.comm

    def __getitems__(self, indices: list[int]) -> Any:
        N = len(indices)
        index_buffer = torch.empty(N * self.__comm.size, dtype=torch.int64)
        self.__comm.Allgather(torch.tensor(indices), index_buffer)

        return ht.take(self.__a, index_buffer, axis=0).larray

    def __len__(self):
        return len(self.__a)


@dataclass
class _PendingBatch:
    """
    A batch whose Itake operations have been started but may not
    have completed yet.

    Keeping `indices` here also guarantees that the index tensor
    stays alive until all corresponding requests are finished.
    """

    indices: torch.Tensor
    requests: tuple


class AsyncDataLoader:
    def __init__(
        self,
        data: ht.DNDarray | tuple[ht.DNDarray, ...],
        batch_size: int,
        axis: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        prefetch_factor: int = 0,
        seed: int = 0,
    ) -> None:
        if isinstance(data, ht.DNDarray):
            self._arrays = (data,)
            self._single_array = True
        elif isinstance(data, tuple):
            if len(data) == 0:
                raise ValueError("'data' cannot be an empty tuple.")

            if not all(isinstance(x, ht.DNDarray) for x in data):
                raise TypeError("Every element of 'data' must be a DNDarray.")

            self._arrays = data
            self._single_array = False
        else:
            raise TypeError("'data' must be a DNDarray or a tuple of DNDarrays.")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("'batch_size' must be a positive integer.")
        if not isinstance(prefetch_factor, int) or prefetch_factor < 0:
            raise ValueError("'prefetch_factor' must be a non-negative integer.")

        if not isinstance(axis, int):
            raise TypeError("'axis' must be an integer.")
        if axis != 0:
            raise NotImplementedError("Data not split across")

        self.batch_size = batch_size
        self.axis = axis
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.seed = seed

        lengths = []

        for array in self._arrays:
            normalized_axis = axis

            if normalized_axis < 0:
                normalized_axis += array.ndim

            if not 0 <= normalized_axis < array.ndim:
                raise ValueError(f"{axis=} does not exist for array with {array.ndim} dimensions.")

            lengths.append(array.gshape[normalized_axis])

        if len(set(lengths)) != 1:
            raise ValueError(
                f"All DNDarrays must have the same global length along axis {axis}. Got {lengths}."
            )

        self._num_samples = lengths[0]
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self._num_samples // self.batch_size

        return (self._num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "_AsyncDataLoaderIter":
        # Increment immediately so creating a new iterator corresponds
        # to creating a new epoch.
        epoch = self._epoch
        self._epoch += 1

        return _AsyncDataLoaderIter(
            loader=self,
            epoch=epoch,
        )


class _AsyncDataLoaderIter:
    def __init__(self, loader: AsyncDataLoader, epoch: int):
        self.loader = loader
        self.epoch = epoch

        self._batch_iterator = iter(self._batch_indices())

        self._pending = deque()
        self._input_exhausted = False
        self._closed = False

    def _batch_indices(self) -> Iterator[torch.Tensor]:
        """
        Generate global indices for every batch.

        Every MPI rank generates exactly the same sequence.
        """
        n = self.loader._num_samples
        batch_size = self.loader.batch_size

        if self.loader.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.loader.seed + self.epoch)

            indices = torch.randperm(n, generator=generator, device="cpu")

            for start in range(0, n, batch_size):
                batch = indices[start : start + batch_size]

                if self.loader.drop_last and batch.numel() < batch_size:
                    return

                yield batch
        else:
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)

                if self.loader.drop_last and stop - start < batch_size:
                    return

                yield torch.arange(start, stop, device="cpu")

    def _start_batch(self, indices: torch.Tensor) -> _PendingBatch:
        """
        Start all Itake operations belonging to one batch.
        """
        requests = []

        # Reuse index tensors for arrays living on the same device.
        indices_by_device = {}

        for array in self.loader._arrays:
            device = array.larray.device

            if device not in indices_by_device:
                indices_by_device[device] = indices.to(device)

            device_indices = indices_by_device[device]

            requests.append(ht.Itake(array, device_indices, axis=self.loader.axis))

        return _PendingBatch(indices=indices, requests=tuple(requests))

    def _finish_batch(self, pending: _PendingBatch):
        """
        Wait for all arrays making up a batch.
        """
        results = tuple(request.Wait() for request in pending.requests)

        if self.loader._single_array:
            return results[0]

        return results

    def _start_next(self) -> bool:
        """
        Start one additional batch.

        Returns False when there are no batches remaining.
        """
        if self._input_exhausted:
            return False

        try:
            indices = next(self._batch_iterator)
        except StopIteration:
            self._input_exhausted = True
            return False

        self._pending.append(self._start_batch(indices))

        return True

    def _fill_prefetch_queue(self):
        """
        Keep exactly `prefetch_factor` future batches in flight,
        where possible.
        """
        while len(self._pending) < self.loader.prefetch_factor:
            if not self._start_next():
                break

    def __next__(self):
        if self._closed:
            raise StopIteration

        if not self._pending:
            if not self._start_next():
                self._closed = True
                raise StopIteration

        current = self._pending.popleft()
        batch = self._finish_batch(current)
        self._fill_prefetch_queue()
        return batch

    def close(self):
        """
        Complete any outstanding MPI collectives.

        Nonblocking MPI collective operations cannot simply be
        abandoned/cancelled.
        """
        if self._closed:
            return

        while self._pending:
            pending = self._pending.popleft()

            # Complete the request even though its result is discarded.
            self._finish_batch(pending)

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    arr = ht.random.randn(1000, 10_000, split=0)
    dataset = DistributedDataset(arr)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=50, shuffle=True, num_workers=0, collate_fn=lambda x: x
    )

    for batch in dataloader:
        ...

    # dataloader = AsyncDataLoader(arr, 50, axis=0, shuffle=True, drop_last=False, prefetch_factor=0)
    dataloader = ht.utils.data.DistributedDataLoader(
        data=arr, batch_size=50, shuffle=True, drop_last=False, prefetch_factor=10
    )

    comm: ht.MPICommunication = arr.comm
    rank = comm.rank
    size = comm.size

    rates = list()

    for _ in range(100):
        with tqdm(dataloader) as pbar:
            for batch in pbar:
                ...
            # print(f"[{rank}/{size-1}]", 1.0 / pbar.format_dict["elapsed"] * pbar.format_dict["n"])
            rates.append(1.0 / pbar.format_dict["elapsed"] * pbar.format_dict["n"])

    mean = np.mean(rates)
    median = np.median(rates)
    p99 = np.percentile(rates, 99)
    p1 = np.percentile(rates, 1)

    mean = comm.allreduce(mean, op=MPI.SUM) / size
    median = comm.allreduce(median, op=MPI.SUM) / size
    p99 = comm.allreduce(p99, op=MPI.SUM) / size
    p1 = comm.allreduce(p1, op=MPI.SUM) / size

    if rank == 0:
        print(f"{mean=}, it/s")
        print(f"{median=}, it/s")
        print(f"{p99=}, it/s")
        print(f"{p1=}, it/s")


if __name__ == "__main__":
    main()
