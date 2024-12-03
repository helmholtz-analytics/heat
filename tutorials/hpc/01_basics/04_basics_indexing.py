import heat as ht

# ## Indexing

# Heat allows the indexing of arrays, and thereby, the extraction of a partial view of the elements in an array. It is possible to obtain single values as well as entire chunks, i.e. slices.

a = ht.arange(10)

print(a[3])
print(a[1:7])
print(a[::2])

# **NOTE:** Indexing in Heat is undergoing a [major overhaul](https://github.com/helmholtz-analytics/heat/pull/938), to increase interoperability with NumPy/PyTorch indexing, and to provide a fully distributed item setting functionality. Stay tuned for this feature in the next release.
