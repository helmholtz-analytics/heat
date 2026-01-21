import heat as ht
import numpy as np
from perun import monitor

@monitor
def conv2d_fixed_kernel(image, kernel, mode):
    return ht.convolve2d(image, kernel, mode=mode)

# Only image distributed
# Scale 2D image, convolution fixed
current_size = 100
kernel = ht.random.random((3,3), split=None)
for n in range(0,10):
    image = ht.random.random((current_size, current_size), split=0)

    convolved_image = conv2d(image, kernel, mode="full")
    current_size = 2*current_size

# Both distributed
# Scale 2D image + kernel the same way (keep window the same)

# Kernel distributed
# Scaled kernel, keep 2D image the same
