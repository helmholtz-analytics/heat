from mpi4py import MPI

comm = MPI.COMM_WORLD

import heat.cw4heat as ht


with ht.cw4h() as cw:
    if cw.controller():
        a = ht.arange(8, split=0)
        b = ht.ones(8, split=0)
        c = a @ b
        # assert hasattr(c, "__partitioned__")
        print(type(c))
        p = c.__partitioned__()
        print(c.shape, c, p)
        for k, v in p["partitions"].items():
            print(k, p["get"](v["data"]))

print("hello")

with ht.cw4h() as cw:
    if cw.controller():
        a = ht.arange(8, split=0)
        b = ht.ones(8, split=0)
        c = a @ b
        # assert hasattr(c, "__partitioned__")
        p = c.__partitioned__()
        print(c.shape, c, p)
        for k, v in p["partitions"].items():
            print(k, p["get"](v["data"]))
    else:
        p = None

p = comm.bcast(p, 0)
for v in p["partitions"].values():
    if v["location"] == comm.rank:
        print("My part:", p["get"](v["data"]))

print("bye")
