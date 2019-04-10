import struct

f = open("./f0", 'w')

data = [123, 234]
for d in data:
    output = struct.pack("L", d)
    f.write(output)

f.close()
