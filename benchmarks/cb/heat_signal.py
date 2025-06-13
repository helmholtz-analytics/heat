import heat as ht
from perun import monitor


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


def run_signal_benchmarks():
    n_s = 1000000
    n_k = 1000
    stride = 3

    # signal distributed
    signal = ht.random.random((n_s,), split=0)
    kernel = ht.random.random((n_k,), split=None)

    convolution_array_distributed(signal, kernel)
    convolution_array_distributed_stride(signal, kernel, stride)

    del signal, kernel

    # kernel distributed
    signal = ht.random.random((n_s,), split=None)
    kernel = ht.random.random((n_k,), split=0)

    convolution_kernel_distributed(signal, kernel)
    convolution_kernel_distributed_stride(signal, kernel, stride)

    del signal, kernel

    # signal and kernel distributed
    signal = ht.random.random((n_s,), split=0)
    kernel = ht.random.random((n_k,), split=0)

    convolution_distributed(signal, kernel)
    convolution_distributed_stride(signal, kernel, stride)

    del signal, kernel

    # batch processing
    n_s = 5000
    n_b = 9000
    n_k = 503
    signal = ht.random.random((n_b, n_s), split=0)
    kernel = ht.random.random((n_b, n_k), split=0)

    convolution_batch_processing(signal, kernel)
    convolution_batch_processing_stride(signal, kernel, stride)
