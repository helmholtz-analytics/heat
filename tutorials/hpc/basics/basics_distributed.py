import heat as ht

# ### Distributed Computing
#
# Heat is also able to make use of distributed processing capabilities such as those in high-performance cluster systems. For this, Heat exploits the fact that the operations performed on a multi-dimensional array are usually identical for all data items. Hence, a data-parallel processing strategy can be chosen, where the total number of data items is equally divided among all processing nodes. An operation is then performed individually on the local data chunks and, if necessary, communicates partial results behind the scenes. A Heat array assumes the role of a virtual overlay of the local chunks and realizes and coordinates the computations - see the figure below for a visual representation of this concept.
#
# <img src="https://github.com/helmholtz-analytics/heat/blob/main/doc/images/split_array.png?raw=true" width="100%"></img>
#
# The chunks are always split along a singular dimension (i.e. 1-D domain decomposition) of the array. You can specify this in Heat by using the `split` paramter. This parameter is present in all relevant functions, such as array creation (`zeros(), ones(), ...`) or I/O (`load()`) functions.
#
#
#
#
# Examples are provided below. The result of an operation on a Heat tensor will in most cases preserve the split of the respective operands. However, in some cases the split axis might change. For example, a transpose of a Heat array will equally transpose the split axis. Furthermore, a reduction operations, e.g. `sum()` that is performed across the split axis, might remove data partitions entirely. The respective function behaviors can be found in Heat's documentation.
#
# You may also modify the data partitioning of a Heat array by using the `resplit()` function. This allows you to repartition the data as you so choose. Please note, that this should be used sparingly and for small data amounts only, as it entails significant data copying across the network. Finally, a Heat array without any split, i.e. `split=None` (default), will result in redundant copies of data on each computation node.
#
# On a technical level, Heat follows the so-called [Bulk Synchronous Parallel (BSP)](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel) processing model. For the network communication, Heat utilizes the [Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/), a *de facto* standard on modern high-performance computing systems. It is also possible to use MPI on your laptop or desktop computer. Respective software packages are available for all major operating systems. In order to run a Heat script, you need to start it slightly differently than you are probably used to. This
#
# ```bash
# python ./my_script.py
# ```
#
# becomes this instead:
#
# ```bash
# mpirun -n <number_of_processors> python ./my_script.py
# ```
# On an HPC cluster you'll of course use SBATCH or similar.
#
#
# Let's see some examples of working with distributed Heat:

# In the following examples, we'll recreate the array shown in the figure, a 3-dimensional DNDarray of integers ranging from 0 to 59 (5 matrices of size (4,3)).


dndarray = ht.arange(60).reshape(5, 4, 3)
if dndarray.comm.rank == 0:
    print(f"3-dimensional DNDarray of integers ranging from 0 to 59: {dndarray}")


# Notice the additional metadata printed with the DNDarray. With respect to a numpy ndarray, the DNDarray has additional information on the device (in this case, the CPU) and the `split` axis. In the example above, the split axis is `None`, meaning that the DNDarray is not distributed and each MPI process has a full copy of the data.
#
# Let's experiment with a distributed DNDarray: we'll split the same DNDarray as above, but distributed along the major axis.


dndarray = ht.arange(60, split=0).reshape(5, 4, 3)
if dndarray.comm.rank == 0:
    print(f"3-dimensional DNDarray splitted across dim 0: {dndarray}")


# The `split` axis is now 0, meaning that the DNDarray is distributed along the first axis. Each MPI process has a slice of the data along the first axis. In order to see the data on each process, we can print the "local array" via the `larray` attribute.


if dndarray.comm.rank == 0:
    print(f"data on each process: {dndarray.larray}")


# Note that the `larray` is a `torch.Tensor` object. This is the underlying tensor that holds the data. The `dndarray` object is an MPI-aware wrapper around these process-local tensors, providing memory-distributed functionality and information.

# The DNDarray can be distributed along any axis. Modify the `split` attribute when creating the DNDarray in the cell above, to distribute it along a different axis, and see how the `larray`s change. You'll notice that the distributed arrays are always load-balanced, meaning that the data are distributed as evenly as possible across the MPI processes.

# The `DNDarray` object has a number of methods and attributes that are useful for distributed computing. In particular, it keeps track of its global and local (on a given process) shape through distributed operations and array manipulations. The DNDarray is also associated to a `comm` object, the MPI communicator.
#
# (In MPI, the *communicator* is a group of processes that can communicate with each other. The `comm` object is a `MPI.COMM_WORLD` communicator, which is the default communicator that includes all the processes. The `comm` object is used to perform collective operations, such as reductions, scatter, gather, and broadcast. The `comm` object is also used to perform point-to-point communication between processes.)


print(f"Global shape on rank {dndarray.comm.rank}: {dndarray.shape}")
print(f"Local shape on rank: {dndarray.comm.rank}: {dndarray.lshape}")


# You can perform a vast number of operations on DNDarrays distributed over multi-node and/or multi-GPU resources. Check out our [Numpy coverage tables](https://github.com/helmholtz-analytics/heat/blob/main/coverage_tables.md) to see what operations are already supported.
#
# The result of an operation on DNDarays will in most cases preserve the `split` or distribution axis of the respective operands. However, in some cases the split axis might change. For example, a transpose of a Heat array will equally transpose the split axis. Furthermore, a reduction operations, e.g. `sum()` that is performed across the split axis, might remove data partitions entirely. The respective function behaviors can be found in Heat's documentation.


# transpose
print(dndarray.T)


# reduction operation along the distribution axis
print(dndarray.sum(axis=0))


other_dndarray = ht.arange(60, 120, split=0).reshape(5, 4, 3)  # distributed reshape

# element-wise multiplication
print(dndarray * other_dndarray)


# As we saw earlier, because the underlying data objects are PyTorch tensors, we can easily create DNDarrays on GPUs or move DNDarrays to GPUs. This allows us to perform distributed array operations on multi-GPU systems.
#
# So far we have demostrated small, easy-to-parallelize arithmetical operations. Let's move to linear algebra. Heat's `linalg` module supports a wide range of linear algebra operations, including matrix multiplication. Matrix multiplication is a very common operation data analysis, it is computationally intensive, and not trivial to parallelize.
#
# With Heat, you can perform matrix multiplication on distributed DNDarrays, and the operation will be parallelized across the MPI processes. Here on 4 GPUs:


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

# ### Interoperability
#
# We can easily create DNDarrays from PyTorch tensors and numpy ndarrays. We can also convert DNDarrays to PyTorch tensors and numpy ndarrays. This makes it easy to integrate Heat into existing PyTorch and numpy workflows. Here a basic example with xarrays:


import xarray as xr

local_xr = xr.DataArray(dndarray.larray, dims=("z", "y", "x"))
# proceed with local xarray operations
print(local_xr)


# **NOTE:** this is not a distributed `xarray`, but local xarray objects on each rank.
# Work on [expanding xarray support](https://github.com/helmholtz-analytics/heat/pull/1183) is ongoing.
#

# Heat will try to reuse the memory of the original array as much as possible. If you would prefer a copy with different memory, the ```copy``` keyword argument can be used when creating a DNDArray from other libraries.


import torch

torch_array = torch.arange(5)
heat_array = ht.array(torch_array, copy=False)
heat_array[0] = -1
print(torch_array)

torch_array = torch.arange(5)
heat_array = ht.array(torch_array, copy=True)
heat_array[0] = -1
print(torch_array)


# Interoperability is a key feature of Heat, and we are constantly working to increase Heat's compliance to the [Python array API standard](https://data-apis.org/array-api/latest/). As usual, please [let us know](tinyurl.com/demoissues) if you encounter any issues or have any feature requests.
