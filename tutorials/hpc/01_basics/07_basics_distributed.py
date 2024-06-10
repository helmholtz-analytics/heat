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
    print("3-dimensional DNDarray of integers ranging from 0 to 59:")
print(dndarray)


# Notice the additional metadata printed with the DNDarray. With respect to a numpy ndarray, the DNDarray has additional information on the device (in this case, the CPU) and the `split` axis. In the example above, the split axis is `None`, meaning that the DNDarray is not distributed and each MPI process has a full copy of the data.
#
# Let's experiment with a distributed DNDarray: we'll split the same DNDarray as above, but distributed along the major axis.


dndarray = ht.arange(60, split=0).reshape(5, 4, 3)
if dndarray.comm.rank == 0:
    print("3-dimensional DNDarray of integers ranging from 0 to 59:")
print(dndarray)


# The `split` axis is now 0, meaning that the DNDarray is distributed along the first axis. Each MPI process has a slice of the data along the first axis. In order to see the data on each process, we can print the "local array" via the `larray` attribute.

print(f"data on process no {dndarray.comm.rank}: {dndarray.larray}")


# Note that the `larray` is a `torch.Tensor` object. This is the underlying tensor that holds the data. The `dndarray` object is an MPI-aware wrapper around these process-local tensors, providing memory-distributed functionality and information.

# The DNDarray can be distributed along any axis. Modify the `split` attribute when creating the DNDarray in the cell above, to distribute it along a different axis, and see how the `larray`s change. You'll notice that the distributed arrays are always load-balanced, meaning that the data are distributed as evenly as possible across the MPI processes.

# The `DNDarray` object has a number of methods and attributes that are useful for distributed computing. In particular, it keeps track of its global and local (on a given process) shape through distributed operations and array manipulations. The DNDarray is also associated to a `comm` object, the MPI communicator.
#
# (In MPI, the *communicator* is a group of processes that can communicate with each other. The `comm` object is a `MPI.COMM_WORLD` communicator, which is the default communicator that includes all the processes. The `comm` object is used to perform collective operations, such as reductions, scatter, gather, and broadcast. The `comm` object is also used to perform point-to-point communication between processes.)


print(f"Global shape on rank {dndarray.comm.rank}: {dndarray.shape}")
print(f"Local shape on rank: {dndarray.comm.rank}: {dndarray.lshape}")
print(f"Local device on rank: {dndarray.comm.rank}: {dndarray.device}")
