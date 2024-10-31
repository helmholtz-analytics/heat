import heat as ht

# Create a DNDarray with split=1 and dtype=ht.float64
x = ht.random.randn(100, 100, split=1, dtype=ht.float64)

# Call _estimate_largest_singularvalue with an invalid algorithm
est = ht.linalg._estimate_largest_singularvalue(x, algorithm="invalid")
