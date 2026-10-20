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


def run_classification_benchmarks():
    # Setup distributed dataset partitioned along samples (axis 0)
    n_samples = 4000
    n_features = 32
    n_neighbors = 20

    data = ht.random.random((n_samples, n_features), split=0, dtype=ht.float64)

    # Benchmark runtime-efficient path (replicated A)
    lof_replicated_indexing(data, n_neighbors)

    # Benchmark memory-efficient path (MPI Alltoallv advanced indexing)
    lof_distributed_indexing(data, n_neighbors)

    del data
