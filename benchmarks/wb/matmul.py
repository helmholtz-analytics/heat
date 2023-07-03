# flake8: noqa
import heat as ht
from heat.utils.data.mnist import MNISTDataset
from perun.decorator import monitor

mnist_data = MNISTDataset("../../heat/datasets", train=True, split=0)
data_1 = mnist_data.htdata.reshape((28 * 28, -1)).T / 255
data_0 = data_1.resplit_(0)


@monitor()
def matmul_split_01(mat1, mat2):
    mat = mat1 @ mat2


@monitor()
def matmul_split_10(mat1, mat2):
    mat = mat1 @ mat2


@monitor()
def matmul_split_00(mat1, mat2):
    mat = mat1 @ mat2


@monitor()
def matmul_split_11(mat1, mat2):
    mat = mat1 @ mat2


matmul_split_01(data_1.T, data_1)
matmul_split_10(data_0.T, data_0)
matmul_split_00(data_1.T, data_0)
matmul_split_11(data_0.T, data_1)
