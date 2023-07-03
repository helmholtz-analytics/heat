# flake8: noqa
import heat as ht
from heat.utils.data.mnist import MNISTDataset
from perun.decorator import monitor

mnist_data = MNISTDataset("../../heat/datasets", train=True, split=0)
data = mnist_data.htdata.reshape((28 * 28, -1)).T / 255
print(data.min(), data.max())


@monitor()
def hierachical_svd_rank(data, r):
    approx_svd = ht.linalg.hsvd_rank(data, maxrank=r, compute_sv=True, silent=True)


@monitor()
def hierachical_svd_tol(data, tol):
    approx_svd = ht.linalg.hsvd_rtol(data, rtol=tol, compute_sv=True, silent=True)


hierachical_svd_rank(data, 10)
hierachical_svd_tol(data, 1e-2)
