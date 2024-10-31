"""
Nur für Testzwecke - später löschen!
"""

import heat as ht

A = ht.random.randn(
    25 * ht.MPI_WORLD.size + 2, 25 * ht.MPI_WORLD.size + 1, split=1, dtype=ht.float64
)
