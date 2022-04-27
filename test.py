import heat as ht 

v = ht.ones((3, 3))
a = ht.ones((10, 10), split=1)


aa=ht.convolve2d(a, v, mode='full',boundary='fill', fillvalue=2)



print('aa', aa.sum())

