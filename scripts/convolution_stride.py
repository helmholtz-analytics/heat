import heat as ht
from heat import manipulations
import numpy as np
import torch

full_even_stride2 = ht.array([0, 3, 10, 18, 26, 34, 42, 50, 42, 15]).astype(ht.int)
full_odd_stride2 = ht.array([0, 3, 9, 15, 21, 27, 33, 39, 29]).astype(ht.int)
valid_even_stride2 = ht.array([6, 14, 22, 30, 38, 46, 54]).astype(ht.int)

dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
signal = ht.arange(0, 16).astype(ht.int)
full_ones = ht.ones(7, split=0).astype(ht.int)
kernel_odd = ht.ones(3).astype(ht.int)
kernel_even = [1, 1, 1, 1]
dis_kernel_odd = ht.ones(3, split=0).astype(ht.int)
dis_kernel_even = ht.ones(4, split=0).astype(ht.int)

modes = ["full", "valid"]
for i, mode in enumerate(modes):
    print(mode)
    np.random.seed(12)
    np_a = np.random.randint(1000, size=4418)
    np_b = np.random.randint(1000, size=1543)

    stride = np.random.randint(1, high=len(np_a), size=1)[0]
    print(stride)
    t_a = torch.asarray(np_a, dtype=torch.int64).reshape([1, 1, len(np_a)])
    t_b = torch.asarray(np_b, dtype=torch.int64).reshape([1, 1, len(np_b)])
    t_b = torch.flip(t_b, [2])
    if mode == "full":
        torch_conv = torch.conv1d(t_a, t_b, stride=stride, padding=len(np_b) - 1)
    else:
        torch_conv = torch.conv1d(t_a, t_b, stride=stride, padding=0)

    torch_conv = torch.squeeze(torch_conv)

    a = ht.array(np_a, split=0, dtype=ht.int32)
    b = ht.array(np_b, split=0, dtype=ht.int32)
    conv = ht.convolve(a, b, mode=mode, stride=stride)
    print(conv)
    print(torch_conv.type(torch.int32))
    print(ht.equal(conv, ht.array(torch_conv.type(torch.int32))))
