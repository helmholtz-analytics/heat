""" Test script for MPI & Heat installation """

import heat as ht

x = ht.arange(10, split=0)
if x.comm.rank == 0:
    print("x is distributed: ", x.is_distributed())
print("Global DNDarray x: ", x)
print("Local torch tensor on rank ", x.comm.rank, ": ", x.larray)
