import heat as ht

# ---
# Heat implements the same broadcasting rules (implicit repetion of an operation when the rank/shape of the operands do not match) as NumPy does, e.g.:

a = ht.arange(10) + 3
print(f"broadcast example of adding single value 3 to [0, 1, ..., 9]: {a}")


a = ht.ones(
    (
        3,
        4,
    )
)
b = ht.arange(4)
print(
    f"broadcasting across the first dimension of  {a} with shape = (3, 4) and {b} with shape = (4): {a + b}"
)
