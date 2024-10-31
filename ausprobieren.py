"""
Nur für Testzwecke ... später löschen!
"""

import heat as ht


x = ht.random.randn(ht.MPI_WORLD.size * 10 + 3, 9, dtype=ht.float32, split=0)
q, r = ht.linalg.qr(x, mode="reduced")

print("Non-batched case:")
print(ht.allclose(q @ r, x, atol=1e-5, rtol=1e-5))
print(ht.allclose(q.transpose() @ q, ht.eye(q.shape[1], dtype=ht.float32), atol=1e-5, rtol=1e-5))

x = ht.random.randn(8, ht.MPI_WORLD.size * 10 + 3, 9, dtype=ht.float32, split=1)
q, r = ht.linalg.qr(x, mode="reduced", procs_to_merge=3)

print("batched case:")
batched_id = ht.stack([ht.eye(q.shape[2], dtype=ht.float32) for _ in range(q.shape[0])])
print(ht.allclose(q.transpose([0, 2, 1]) @ q, batched_id, atol=1e-6, rtol=1e-6))
print(ht.allclose(q @ r, x, atol=1e-6, rtol=1e-6))
