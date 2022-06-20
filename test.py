import heat as ht

print("Input array:")
b = ht.array([
    [[1,2,0,0], [2,0,0,5]],
    [[1,2,0,0], [2,0,0,5]]
    ])
print(b)
b = ht.sparse_matrix(b)