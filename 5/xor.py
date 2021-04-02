#%%
def AND(x1, x2):
    return x1 and x2

def NAND(x1, x2):
    if not(x1 and x2):
        return 1
    else:
        return 0

def OR(x1, x2):
    return x1 or x2

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
# %%
print('AND')
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
print()

print('NAND')
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))
print()

print('OR')
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
print()

print('XOR')
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
print()
# %%
