# flake8: noqa
import heat as ht
from mpi4py import MPI
from perun import monitor
from heat.classification import LocalOutlierFactor


@monitor()
def lof_replicated_indexing(data, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, fully_distributed=False)
    lof.fit(data)


@monitor()
def lof_distributed_indexing(data, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, fully_distributed=True)
    lof.fit(data)


@monitor()
def lof_non_distributed(data, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit(data)

@monitor()
def lof_distributed_chunked(data, n_neighbors, chunks=4):
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, chunks=chunks, fully_distributed=True
    )
    lof.fit(data)


@monitor()
def lof_non_distributed_chunked(data, n_neighbors, chunks=4):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, chunks=chunks)
    lof.fit(data)


def run_classification_benchmarks():
    n_samples = 20000
    n_features = 32
    n_neighbors = 20

    # Setup distributed dataset partitioned along samples (axis 0)
    data = ht.random.random((n_samples, n_features), split=0, dtype=ht.float64)

    # Benchmark memory-efficient path (MPI Alltoallv advanced indexing)
    lof_distributed_indexing(data, n_neighbors)

    # Benchmark runtime-efficient path (replicated A)
    lof_replicated_indexing(data, n_neighbors)

    # Distributed with chunking (reduced peak memory during top-k distance calculation)
    lof_distributed_chunked(data, n_neighbors, chunks=4)

    del data

    # Setup non-distributed dataset (replicated across all processes)
    data_non_dist = ht.random.random((n_samples, n_features), split=None, dtype=ht.float64)

    # Benchmark fully non-distributed execution
    lof_non_distributed(data_non_dist, n_neighbors)

    # Non-distributed with chunking
    lof_non_distributed_chunked(data_non_dist, n_neighbors, chunks=4)

    del data_non_dist
