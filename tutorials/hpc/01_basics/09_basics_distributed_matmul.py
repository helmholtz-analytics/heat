# As we saw earlier, because the underlying data objects are PyTorch tensors, we can easily create DNDarrays on GPUs or move DNDarrays to GPUs. This allows us to perform distributed array operations on multi-GPU systems.
#
# So far we have demostrated small, easy-to-parallelize arithmetical operations. Let's move to linear algebra. Heat's `linalg` module supports a wide range of linear algebra operations, including matrix multiplication. Matrix multiplication is a very common operation data analysis, it is computationally intensive, and not trivial to parallelize.
#
# With Heat, you can perform matrix multiplication on distributed DNDarrays, and the operation will be parallelized across the MPI processes. Here on 4 GPUs:

import heat as ht
import torch

if torch.cuda.is_available():
    device = "gpu"
else:
    device = "cpu"

n, m = 400, 400
x = ht.random.randn(n, m, split=0, device=device)  # distributed RNG
y = ht.random.randn(m, n, split=None, device=device)
z = x @ y
print(z)

# `ht.linalg.matmul` or `@` breaks down the matrix multiplication into a series of smaller `torch` matrix multiplications, which are then distributed across the MPI processes. This operation can be very communication-intensive on huge matrices that both require distribution, and users should choose the `split` axis carefully to minimize communication overhead.

# You can experiment with sizes and the `split` parameter (distribution axis) for both matrices and time the result. Note that:
# - If you set **`split=None` for both matrices**, each process (in this case, each GPU) will attempt to multiply the entire matrices. Depending on the matrix sizes, the GPU memory might be insufficient. (And if you can multiply the matrices on a single GPU, it's much more efficient to stick to PyTorch's `torch.linalg.matmul` function.)
# - If **`split` is not None for both matrices**, each process will only hold a slice of the data, and will need to communicate data with other processes in order to perform the multiplication. This **introduces huge communication overhead**, but allows you to perform the multiplication on larger matrices than would fit in the memory of a single GPU.
# - If **`split` is None for one matrix and not None for the other**, the multiplication does not require communication, and the result will be distributed. If your data size allows it, you should always favor this option.
#
# Time the multiplication for different split parameters and see how the performance changes.
#
#


import time

start = time.time()
z = x @ y
end = time.time()
print("runtime: ", end - start)


# Heat supports many linear algebra operations:
# ```bash
# >>> ht.linalg.
# ht.linalg.basics        ht.linalg.hsvd_rtol(    ht.linalg.projection(   ht.linalg.triu(
# ht.linalg.cg(           ht.linalg.inv(          ht.linalg.qr(           ht.linalg.vdot(
# ht.linalg.cross(        ht.linalg.lanczos(      ht.linalg.solver        ht.linalg.vecdot(
# ht.linalg.det(          ht.linalg.matmul(       ht.linalg.svdtools      ht.linalg.vector_norm(
# ht.linalg.dot(          ht.linalg.matrix_norm(  ht.linalg.trace(
# ht.linalg.hsvd(         ht.linalg.norm(         ht.linalg.transpose(
# ht.linalg.hsvd_rank(    ht.linalg.outer(        ht.linalg.tril(
# ```
#
# and a lot more is in the works, including distributed eigendecompositions, SVD, and more. If the operation you need is not yet supported, leave us a note [here](tinyurl.com/demoissues) and we'll get back to you.

# You can of course perform all operations on CPUs. You can leave out the `device` attribute entirely.
