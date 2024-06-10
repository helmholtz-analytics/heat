import heat as ht
import torch

# # Heat as infrastructure for MPI applications
#
# In this section, we'll go through some Heat-specific functionalities that simplify the implementation of a data-parallel application in Python. We'll demonstrate them on small arrays and 4 processes on a single cluster node, but the functionalities are indeed meant for a multi-node set up with huge arrays that cannot be processed on a single node.


# We already mentioned that the DNDarray object is "MPI-aware". Each DNDarray is associated to an MPI communicator, it is aware of the number of processes in the communicator, and it knows the rank of the process that owns it.
#

a = ht.random.randn(7, 4, 3, split=1)
if a.comm.rank == 0:
    print(f"a.comm gets the communicator {a.comm} associated with DNDarray a")

# MPI size = total number of processes
size = a.comm.size

if a.comm.rank == 0:
    print(f"a is distributed over {size} processes")
    print(f"a is a distributed {a.ndim}-dimensional array with global shape {a.shape}")


# MPI rank = rank of each process
rank = a.comm.rank
# Local shape = shape of the data on each process
local_shape = a.lshape
print(f"Rank {rank} holds a slice of a with local shape {local_shape}")


# ### Distribution map
#
# In many occasions, when building a memory-distributed pipeline it will be convenient for each rank to have information on what ranks holds which slice of the distributed array.
#
# The `lshape_map` attribute of a DNDarray gathers (or, if possible, calculates) this info from all processes and stores it as metadata of the DNDarray. Because it is meant for internal use, it is stored in a torch tensor, not a DNDarray.
#
# The `lshape_map` tensor is a 2D tensor, where the first dimension is the number of processes and the second dimension is the number of dimensions of the array. Each row of the tensor contains the local shape of the array on a process.


lshape_map = a.lshape_map
if a.comm.rank == 0:
    print(f"lshape_map available on any process: {lshape_map}")

# Go back to where we created the DNDarray and and create `a` with a different split axis. See how the `lshape_map` changes.
