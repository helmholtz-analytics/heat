import heat as ht

"""
The following variables can be changed:
- N_ELEMENTS_PER_PROC: number of elements per process
- TS_FACTOR_loc: tall-skinny factor for each process (long dimension of local array in tall-skinny matrix is TS_FACTOR_loc times larger than the short dimension)
- vTS_FACTOR_loc: very tall-skinny factor for each process (same as before, but for "very" tall-skinny matrices)
"""
N_ELEMENTS_PER_PROC = 2**26
TS_FACTOR_loc = 2
vTS_FACTOR_loc = 4

"""
all other variables are calculated based on the number of elements per process
shape of a 2D square array: (GSIZE_SQ, GSIZE_SQ)
shape of a 3D cube array: (GSIZE_CB, GSIZE_CB, GSIZE_CB)
shape of a 2D tall-skinny array: (GSIZE_TS_L, GSIZE_TS_S)
shape of a 2D very tall-skinny array: (GSIZE_vTS_L, GSIZE_vTS_S)
similar for short-fat and very short-fat arrays...
"""
n_procs = ht.MPI_WORLD.size
N_ELEMENTS_TOTAL = N_ELEMENTS_PER_PROC * n_procs

GSIZE_SQ = int(N_ELEMENTS_TOTAL**0.5)
TS_FACTOR_GLOB = TS_FACTOR_loc * n_procs  # global tall-skinny factor
GSIZE_TS_S = int(
    (N_ELEMENTS_TOTAL / TS_FACTOR_GLOB) ** 0.5
)  # short dimension of tall-skinny matrix
GSIZE_TS_L = GSIZE_TS_S * TS_FACTOR_GLOB + 1  # long dimension of tall-skinny matrix

vTS_FACTOR_GLOB = vTS_FACTOR_loc * n_procs  # global tall-skinny factor
GSIZE_vTS_S = int(
    (N_ELEMENTS_TOTAL / vTS_FACTOR_GLOB) ** 0.5
)  # short dimension of very tall-skinny matrix
GSIZE_vTS_L = GSIZE_vTS_S * vTS_FACTOR_GLOB + 1  # long dimension of very tall-skinny matrix

GSIZE_CB = int(N_ELEMENTS_TOTAL ** (1 / 3))  # dimension of a cube array

"""
Exceptions needed for the moment:
"""
LANCZOS_SIZE = 2**9
