leaf = 1000
data = []
for i in range(6):
	v = 2 ** i
	for j in range(v):
		if i == 5:
			data += [leaf, j - v + 1, leaf]
		else:
			data += [i, 2, (v + j + 1) * 2 + 1 - (2 ** (i + 1)) - 1]
test_features = [5] * 6
data += test_features
