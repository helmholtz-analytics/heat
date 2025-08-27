# flake8: noqa
import heat as ht
from mpi4py import MPI
from perun import monitor
from heat.decomposition import IncrementalPCA


@monitor()
def incremental_pca_split0(list_of_X, n_components):
    ipca = IncrementalPCA(n_components=n_components)
    for X in list_of_X:
        ipca.partial_fit(X)


def run_decomposition_benchmarks():
    list_of_X = [ht.random.rand(50000, 500, split=0) for _ in range(10)]
    incremental_pca_split0(list_of_X, 50)
