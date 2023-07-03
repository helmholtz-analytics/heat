# flake8: noqa
import heat as ht
from heat.utils.data.mnist import MNISTDataset
from perun.decorator import monitor

mnist_data = MNISTDataset("../../heat/datasets", train=True, split=0)
data_0 = mnist_data.htdata.reshape((-1, 28 * 28)) / 255


@monitor()
def qr_split_0(mat):
    ht.linalg.qr(mat)


qr_split_0(data_0)
