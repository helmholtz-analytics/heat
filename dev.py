"""
remove before merge
"""

import heat as ht

x = ht.random.rand(10, 10, split=0)

print(x)

q = ht.linalg.qr(x)
print(q)
