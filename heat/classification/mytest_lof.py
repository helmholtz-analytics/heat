"""Tests during the implementation of the Local Outlier Factor (LOF) algorithm"""

import heat as ht

a = ht.array([10, 20, 2, 17, 8], split=0)
b = ht.sort(a)[0]
c = b[-1]
anomaly = ht.where(a >= 10, 1, -1)
# print(f"a={a}, \n b={b}, \n c={c}")
print(f"anomaly={anomaly}")
