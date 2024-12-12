import heat as ht

# ### DNDarrays
#
#
# Similar to a NumPy `ndarray`, a Heat `dndarray`  (we'll get to the `d` later) is a grid of values of a single (one particular) type. The number of dimensions is the number of axes of the array, while the shape of an array is a tuple of integers giving the number of elements of the array along each dimension.
#
# Heat emulates NumPy's API as closely as possible, allowing for the use of well-known **array creation functions**.


a = ht.array([1, 2, 3])
print("array creation with values [1,2,3] with the heat array method:")
print(a)

a = ht.ones((4, 5))
#     (
#         4,
#         5,
#     )
# )
print("array creation of shape = (4, 5) example with the heat ones method:")
print(a)

a = ht.arange(10)
print("array creation with [0,1,...,9] example with the heat arange method:")
print(a)

a = ht.full(
    (
        3,
        2,
    ),
    fill_value=9,
)
print("array creation with ones and shape = (3, 2) with the heat full method:")
print(a)
