"""Test script for MPI & Heat Convolution"""

import heat as ht
import torch
import numpy as np

ht_dtype = ht.int

mode = "valid"
stride = 2
solution = ht.array([6, 14, 22, 30, 38, 46, 54]).astype(ht_dtype)

dis_signal = ht.arange(0, 16, split=0).astype(ht_dtype)
signal = ht.arange(0, 16).astype(ht_dtype)
kernel = [1, 1, 1, 1]
dis_kernel = ht.ones(4, split=0).astype(ht_dtype)


conv = ht.convolve(dis_signal, dis_kernel, mode=mode, stride=stride)

print(conv)
print(solution)
# np.random.seed(12)
# np_a = np.random.randint(1000, size=4418)
# np_b = np.random.randint(1000, size=154)
# # torch convolution does not support int on MPS
# ht_dtype = ht.int64
# stride = np.random.randint(1, high=len(np_a), size=1)[0]
#
# # stride = 3
# # generate solution
# t_a = torch.asarray(np_a, dtype=torch.int64).reshape([1, 1, len(np_a)])
# t_b = torch.asarray(np_b, dtype=torch.int64).reshape([1, 1, len(np_b)])
# t_b = torch.flip(t_b, [2])
#
# for mode in ["full", "valid"]:
#     if mode == "full":
#         solution = torch.conv1d(t_a, t_b, stride=stride, padding=len(np_b) - 1)
#     else:
#         solution = torch.conv1d(t_a, t_b, stride=stride, padding=0)
#         solution = torch.squeeze(solution).numpy()
#     if solution.shape == ():
#         solution = solution.reshape((1,))
#
#         # test
#     a = ht.array(np_a, split=0, dtype=ht_dtype)
#     b = ht.array(np_b, split=None, dtype=ht_dtype)
#
#     conv = ht.convolve(a, b, mode=mode, stride=stride)

# print((solution==conv).all())
