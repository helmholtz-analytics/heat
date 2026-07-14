import heat as ht
from perun import monitor

# 1D
@monitor()
def convolution_array_distributed(signal, kernel):
    ht.convolve(signal, kernel, mode="full")


@monitor()
def convolution_kernel_distributed(signal, kernel):
    ht.convolve(signal, kernel, mode="full")


@monitor()
def convolution_distributed(signal, kernel):
    ht.convolve(signal, kernel, mode="full")


@monitor()
def convolution_batch_processing(signal, kernel):
    ht.convolve(signal, kernel, mode="full")


@monitor()
def convolution_array_distributed_stride(signal, kernel, stride):
    ht.convolve(signal, kernel, mode="full", stride=stride)


@monitor()
def convolution_kernel_distributed_stride(signal, kernel, stride):
    ht.convolve(signal, kernel, mode="full", stride=stride)


@monitor()
def convolution_distributed_stride(signal, kernel, stride):
    ht.convolve(signal, kernel, mode="full", stride=stride)


@monitor()
def convolution_batch_processing_stride(signal, kernel, stride):
    ht.convolve(signal, kernel, mode="full", stride=stride)

# 2D

@monitor()
def convolution2d_array_distributed(signal, kernel):
    ht.convolve2d(signal, kernel, mode="full")


@monitor()
def convolution2d_kernel_distributed(signal, kernel):
    ht.convolve2d(signal, kernel, mode="full")


@monitor()
def convolution2d_distributed(signal, kernel):
    ht.convolve2d(signal, kernel, mode="full")


@monitor()
def convolution2d_batch_processing(signal, kernel):
    ht.convolve2d(signal, kernel, mode="full")


@monitor()
def convolution2d_array_distributed_stride(signal, kernel, stride):
    ht.convolve2d(signal, kernel, mode="full", stride=stride)


@monitor()
def convolution2d_kernel_distributed_stride(signal, kernel, stride):
    ht.convolve2d(signal, kernel, mode="full", stride=stride)


@monitor()
def convolution2d_distributed_stride(signal, kernel, stride):
    ht.convolve2d(signal, kernel, mode="full", stride=stride)


@monitor()
def convolution2d_batch_processing_stride(signal, kernel, stride):
    ht.convolve2d(signal, kernel, mode="full", stride=stride)

def create_signal_kernel(conv_dim, split_signal, split_kernel):
    n_s = 2000000
    n_k = 9999
    stride = 3

    signal = ht.random.random((n_s,), split=split_signal)
    kernel = ht.random.random_integer(0, 2, (n_k,), split=split_kernel)
    if conv_dim == 2:
        n_s_1 = n_s // 1000
        n_s_2 = n_s // n_s_1
        signal = ht.reshape(signal, (n_s_1, n_s_2))

        n_k_1 = n_k // 101
        n_k_2 = n_k // n_k_1
        kernel = ht.reshape(kernel, (n_k_1, n_k_2))

        stride = (stride, stride)

    return signal, kernel, stride

def create_signal_kernel_batch(conv_dim):
    n_s = 50000
    n_b = 1000
    n_k = 525
    stride = 3

    signal = ht.random.random((n_b, n_s), split=0)
    kernel = ht.random.random_integer(0, 1, (n_b, n_k), split=0)

    if conv_dim == 2:
        n_s_1 = n_s // 250
        n_s_2 = n_s // n_s_1
        signal = ht.reshape(signal, (n_b,n_s_1, n_s_2))

        n_k_1 = n_k // 25
        n_k_2 = n_k // n_k_1
        kernel = ht.reshape(kernel, (n_b, n_k_1, n_k_2))

        stride = (stride, stride)

    return signal, kernel, stride

def run_signal_benchmarks():

    for conv_dim in [1, 2]:
        # signal distributed
        signal, kernel, stride = create_signal_kernel(conv_dim, 0, None)

        if conv_dim == 1:
            convolution_array_distributed(signal, kernel)
            convolution_array_distributed_stride(signal, kernel, stride)
        elif conv_dim == 2:
            convolution2d_array_distributed(signal, kernel)
            convolution2d_array_distributed_stride(signal, kernel, stride)

        del signal, kernel

        # kernel distributed
        signal, kernel, stride = create_signal_kernel(conv_dim, None, 0)

        if conv_dim == 1:
            convolution_kernel_distributed(signal, kernel)
            convolution_kernel_distributed_stride(signal, kernel, stride)
        elif conv_dim == 2:
            convolution2d_kernel_distributed(signal, kernel)
            convolution2d_kernel_distributed_stride(signal, kernel, stride)

        del signal, kernel

        # signal and kernel distributed
        signal, kernel, stride = create_signal_kernel(conv_dim, 0, 0)

        if conv_dim == 1:
            convolution_distributed(signal, kernel)
            convolution_distributed_stride(signal, kernel, stride)
        elif conv_dim == 2:
            convolution2d_distributed(signal, kernel)
            convolution2d_distributed_stride(signal, kernel, stride)

        del signal, kernel

        # batch processing
        signal, kernel, stride = create_signal_kernel_batch(conv_dim)

        if conv_dim == 1:
            convolution_batch_processing(signal, kernel)
            convolution_batch_processing_stride(signal, kernel, stride)
        elif conv_dim == 2:
            convolution2d_batch_processing(signal, kernel)
            convolution2d_batch_processing_stride(signal, kernel, stride)
