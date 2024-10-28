"""
for debugging will be removed later
"""

import heat as ht
import torch
import heat.decomposition as htd


r = 6
A_red = ht.array(
    [
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
    ],
    split=None,
    dtype=ht.float32,
)
x0_reds = ht.random.randn(r, 1, split=None, dtype=ht.float32)
m, n = 25 * ht.MPI_WORLD.size, 20
X = ht.hstack(
    [(ht.array(torch.linalg.matrix_power(A_red.larray, i) @ x0_reds.larray)) for i in range(n)]
)
U = ht.random.randn(m, r, split=0)
U, _ = ht.linalg.qr(U)
X = U @ X

dmd = htd.DMD(svd_solver="full", svd_rank=r)
dmd.fit(X)
print(dmd.rom_basis_.shape, dmd.rom_basis_.split)
dmd.rom_basis_.resplit_(None)

# check batch prediction
X_new = X[:, :15]
X_new.resplit_(None)
Y = dmd.predict(X_new, 5)
print(Y.shape)
print(Y.split)
for i in range(4):
    print(ht.allclose(Y[i, :, :5], X[:, i : i + 5], atol=1e-2, rtol=1e-2))

Z = dmd.predict(ht.random.rand(m, 10, split=1), [-1, 1, 2])
print(Z.shape)
