import heat as ht
import torch

# ## Parallel Processing
# ---
#
# Heat's actual power lies in the possibility to exploit the processing performance of modern accelerator hardware (GPUs) as well as distributed (high-performance) cluster systems. All operations executed on CPUs are, to a large extent, vectorized (AVX) and thread-parallelized (OpenMP). Heat builds on PyTorch, so it supports GPU acceleration on Nvidia and AMD GPUs.
#
# For distributed computations, your system or laptop needs to have Message Passing Interface (MPI) installed. For GPU computations, your system needs to have one or more suitable GPUs and (MPI-aware) CUDA/ROCm ecosystem.
#
# **NOTE:** The GPU examples below will only properly execute on a computer with a GPU. Make sure to either start the notebook on an appropriate machine or copy and paste the examples into a script and execute it on a suitable device.

# ### GPUs
#
# Heat's array creation functions all support an additional parameter that which places the data on a specific device. By default, the CPU is selected, but it is also possible to directly allocate the data on a GPU.


if torch.cuda.is_available():
    ht.zeros(
        (
            3,
            4,
        ),
        device="gpu",
    )

# Arrays on the same device can be seamlessly used in any Heat operation.

if torch.cuda.is_available():
    a = ht.zeros(
        (
            3,
            4,
        ),
        device="gpu",
    )
    b = ht.ones(
        (
            3,
            4,
        ),
        device="gpu",
    )
    print(a + b)


# However, performing operations on arrays with mismatching devices will purposefully result in an error (due to potentially large copy overhead).

if torch.cuda.is_available():
    a = ht.full(
        (
            3,
            4,
        ),
        4,
        device="cpu",
    )
    b = ht.ones(
        (
            3,
            4,
        ),
        device="gpu",
    )
    print(a + b)

# It is possible to explicitly move an array from one device to the other and back to avoid this error.

if torch.cuda.is_available():
    a = ht.full(
        (
            3,
            4,
        ),
        4,
        device="gpu",
    )
    a.cpu()
    print(a + b)
