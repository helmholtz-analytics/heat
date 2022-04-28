import heat as ht 
import numpy as np
from scipy import signal

a = ht.ones((5, 5), split=0)
v = ht.ones((3, 3))


gt = [[4., 6., 6., 6., 4.],
      [6., 9., 9., 9., 6.],
      [6., 9., 9., 9., 6.],
      [6., 9., 9., 9., 6.],
      [4., 6., 6., 6., 4.]]


na = np.ones((5,5))
nv = np.ones((3,3))

grad = signal.convolve2d(na, nv, mode='same')

gt = ht.array(grad)


result = ht.convolve2d(a, v, mode='same')


print(result.shape, a.shape)







