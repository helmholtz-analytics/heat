import heat as ht

# ### Operations
#
# Heat supports many mathematical operations, ranging from simple element-wise functions, binary arithmetic operations, and linear algebra, to more powerful reductions. Operations are by default performed on the entire array or they can be performed along one or more of its dimensions when available. Most relevant for data-intensive applications is that **all Heat functionalities support memory-distributed computation and GPU acceleration**. This holds for all operations, including reductions, statistics, linear algebra, and high-level algorithms.
#
# You can try out the few simple examples below if you want, but we will skip to the [Parallel Processing](#Parallel-Processing) section to see memory-distributed operations in action.

a = ht.full((3, 4), 8)
b = ht.ones((3, 4))
c = a + b
print("matrix addition a + b:")
print(c)


c = ht.sub(a, b)
print("matrix substraction a - b:")
print(c)

c = ht.arange(5).sin()
print("application of sin() elementwise:")
print(c)

c = a.T
print("transpose operation:")
print(c)

c = b.sum(axis=1)
print("summation of array elements:")
print(c)
