import heat as ht

x = ht.arange(16)
y = x.reshape(4,4)

x.resplit_(0)
y.resplit_(0)

print("No flip")
print(f"-- x: {x}")
print(f"-- y: {y}")

# flip with positive integer
xp = ht.flip(x,0)
yp = ht.flip(y,[0,1])

print("Postive index flip")
print(f"-- x: {xp}")
print(f"-- y: {yp}")

# flip with negative integer
xn = ht.flip(x,[-1])
yn = ht.flip(y,[-2,-1])

print("Negative index flip")
print(f"-- x: {xn}")
print(f"-- y: {yn}")
