import heat as ht
from heat import manipulations
import numpy as np
import torch


print(ht.convolve(ht.array([1, 2]), ht.array([1, 1]), mode="valid", stride=2))
print(ht.convolve(ht.array([1, 2, 3]), ht.array([1, 1]), mode="valid", stride=2))
