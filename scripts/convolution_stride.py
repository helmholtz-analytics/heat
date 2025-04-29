import heat as ht
from heat import manipulations

full_even_stride2 = ht.array([0, 3, 10, 18, 26, 34, 42, 50, 42, 15]).astype(ht.int)
full_odd_stride2 = ht.array([0, 3, 9, 15, 21, 27, 33, 39, 29]).astype(ht.int)
valid_even_stride2 = ht.array([6, 14, 22, 30, 38, 46, 54]).astype(ht.int)

dis_signal = ht.arange(0, 16, split=0).astype(ht.int)
signal = ht.arange(0, 16).astype(ht.int)
full_ones = ht.ones(7, split=0).astype(ht.int)
kernel_odd = ht.ones(3).astype(ht.int)
kernel_even = [1, 1, 1, 1]
dis_kernel_odd = ht.ones(3, split=0).astype(ht.int)
dis_kernel_even = ht.ones(4, split=0).astype(ht.int)

modes = ["full", "same", "valid"]
for i, mode in enumerate(modes):
    print(mode)
    if mode != "same":
        # odd kernel
        print("Odd kernel")
        conv = ht.convolve(dis_signal, kernel_odd, mode=mode, stride=2)
        gathered = manipulations.resplit(conv, axis=None)
        print(ht.equal(full_odd_stride2[i // 2 : len(full_odd_stride2) - i // 2], gathered))

        conv = ht.convolve(dis_signal, dis_kernel_odd, mode=mode, stride=2)
        gathered = manipulations.resplit(conv, axis=None)
        print(ht.equal(full_odd_stride2[i // 2 : len(full_odd_stride2) - i // 2], gathered))

        conv = ht.convolve(signal, dis_kernel_odd, mode=mode, stride=2)
        gathered = manipulations.resplit(conv, axis=None)
        print(ht.equal(full_odd_stride2[i // 2 : len(full_odd_stride2) - i // 2], gathered))

        # even kernel
        print("Even kernel")
        conv_stride2 = ht.convolve(dis_signal, kernel_even, mode=mode, stride=2)
        dis_conv_stride2 = ht.convolve(dis_signal, dis_kernel_even, mode=mode, stride=2)
        gathered_stride2 = manipulations.resplit(conv_stride2, axis=None)
        dis_gathered_stride2 = manipulations.resplit(dis_conv_stride2, axis=None)

        if mode == "full":
            print(ht.equal(full_even_stride2, gathered_stride2))
            print(ht.equal(full_even_stride2, dis_gathered_stride2))
        else:
            print(ht.equal(valid_even_stride2, gathered_stride2))
            print(ht.equal(valid_even_stride2, dis_gathered_stride2))
