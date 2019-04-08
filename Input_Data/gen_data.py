import struct

f = open("./f0", 'w')

data = [123]
for d in data:
    output = struct.pack("I", d)
    f.write(output)

f.close()
