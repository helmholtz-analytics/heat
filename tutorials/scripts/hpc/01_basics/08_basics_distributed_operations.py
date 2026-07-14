import heat as ht

dndarray = ht.arange(60, split=0).reshape(5, 4, 3)

# You can perform a vast number of operations on DNDarrays distributed over multi-node and/or multi-GPU resources. Check out our [Numpy coverage tables](https://github.com/helmholtz-analytics/heat/blob/main/coverage_tables.md) to see what operations are already supported.
#
# The result of an operation on DNDarays will in most cases preserve the `split` or distribution axis of the respective operands. However, in some cases the split axis might change. For example, a transpose of a Heat array will equally transpose the split axis. Furthermore, a reduction operations, e.g. `sum()` that is performed across the split axis, might remove data partitions entirely. The respective function behaviors can be found in Heat's documentation.


# transpose
print(dndarray.T)


# reduction operation along the distribution axis
print(dndarray.sum(axis=0))

# min / max etc.
print(ht.sin(dndarray).min(axis=0))


other_dndarray = ht.arange(60, 120, split=0).reshape(5, 4, 3)  # distributed reshape

# element-wise multiplication
print(dndarray * other_dndarray)
