# flake8: noqa
import heat as ht
from heat.utils.data.mnist import MNISTDataset
from perun.decorator import monitor

mnist_data = MNISTDataset("../../heat/datasets", train=True, split=0)
data_1 = mnist_data.htdata.reshape((28 * 28, -1)).T / 255
data = data_1.T @ data_1
del data_1


@monitor
def lanczos_split_0(mat):
    ht.linalg.lanczos(mat, m=100)


lanczos_split_0(data)
