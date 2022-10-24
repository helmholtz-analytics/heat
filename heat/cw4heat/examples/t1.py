import pickle
import heat.cw4heat as ht

ht.init()

a = ht.arange(8, split=0)
b = ht.ones(8, split=0)
c = a @ b
# assert hasattr(c, "__partitioned__")
print(type(c))
p = a.__partitioned__()
print(a.shape, a, p)
for k, v in p["partitions"].items():
    print(33)
    print(k, p["get"](v["data"]))
print("kkkkkk")
ht.fini()
