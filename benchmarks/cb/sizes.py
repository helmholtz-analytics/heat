import heat as ht

"""
The following variables can be changed:
- N_ELEMENTS_PER_PROC: number of elements per process
- TS_FACTOR_loc: tall-skinny factor for each process (long dimension of local array in tall-skinny matrix is TS_FACTOR_loc times larger than the short dimension)
- vTS_FACTOR_loc: very tall-skinny factor for each process (same as before, but for "very" tall-skinny matrices)
"""
N_ELEMENTS_PER_PROC = 2**30
TS_FACTOR_loc = 2
vTS_FACTOR_loc = 4

# all other variables are calculated based on the number of elements per process
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
GSIZE_vTS_L = GSIZE_TS_S * vTS_FACTOR_GLOB + 1  # long dimension of very tall-skinny matrix
