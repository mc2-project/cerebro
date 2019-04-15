leaf = 1000

data = []

for i in range(10):
    v = 2 ** i
    for j in range(v):
        if i == 9:
            data += [leaf, j - v + 1, leaf, leaf]
        else:
            data += [i, 2, (v + j + 1) * 2 + 1, (v + j + 1) * 2 + 2]

test_features = [5] * 10
data += test_features
