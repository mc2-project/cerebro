leaf = 1000
data = []
for i in range(10):
	v = 2 ** i
	for j in range(v):
		if i == 9:
			data += [leaf, 5, leaf]
		else:
			data += [i, 5, (2 ** i - 1 + j) * 2 + 1 - (2 ** (i + 1) - 1)]
test_features = [6] * 10
data += test_features
data = [d * (2 ** 32) for d in data]
