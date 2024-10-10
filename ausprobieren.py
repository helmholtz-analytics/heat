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
x0_red = ht.random.randn(r, 1, split=None)
m, n = 25 * ht.MPI_WORLD.size, 15
X = ht.hstack(
    [(ht.array(torch.linalg.matrix_power(A_red.larray, i) @ x0_red.larray)) for i in range(n)]
)
U = ht.random.randn(m, r, split=0)
U, _ = ht.linalg.qr(U)
X = U @ X


dmd = htd.DMD(svd_solver="full", svd_rank=r)
dmd.fit(X)
print(dmd.rom_basis_.shape, dmd.rom_basis_.split)
dmd.rom_basis_.resplit_(None)

# check prediction of next states
Y = dmd.predict_next(X)
print(ht.allclose(Y[:, : n - 1], X[:, 1:], atol=1e-4, rtol=1e-4))

# check batch prediction
Y = dmd.predict(X[:, 0].resplit_(None), n)
print(Y.shape)
print(X - Y)
