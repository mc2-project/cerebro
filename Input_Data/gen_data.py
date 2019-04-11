import struct

f = open("./f0", 'w')
leaf = 1000
data = [0, 0, 1, 2, 1, 1, 3, 4, 2, 2, 5, 6, leaf, 0, leaf, leaf, leaf, 1, leaf, leaf, leaf, 2, leaf, leaf, leaf, 3, leaf, leaf]
test_features = [5] * 10
data += test_features
for d in data:
    output = struct.pack("L", d)
    f.write(output)
print len(data)
f.close()
