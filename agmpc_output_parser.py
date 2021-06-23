import json
import argparse
import pprint

class BitReader(object):
    def __init__(self, data):
        self.data = data
        self.total_bits = len(data) * 8
        self.bit_counter = 0

    def __iter__(self):
        return self

    def read_bit(self):
        if self.bit_counter == self.total_bits:
            raise StopIteration
        
        d = self.data[self.bit_counter / 8]
        d = ord(d)
        v = int(d & (1 << (self.bit_counter % 8)) > 0)
        self.bit_counter += 1
        return v

    def next(self):
        return self.read_bit()

def parse(reader, obj, level=0):
    if obj["type"] == "bits":
        bit = reader.read_bit()
        if level == 0:
            print "bit:{} = {}".format(obj["name"], bit)
        else:
            return bit
    elif obj["type"] == "cbits":
        return int(obj["value"])
    elif obj["type"] == "int_gc":
        num_wires = len(obj["value"])
        int_value = 0
        for i in range(num_wires):
            bit = parse(reader, obj["value"][i], level=level+1)
            int_value += (bit << (i))
            
        k = int(obj["k"])
        if int_value > (1L << (k-1)):
            int_value = (1L << k) - int_value
            int_value = -1 * int_value

        if level == 0:
            print "integer:{} = {}".format(obj["name"], int_value)
        else:
            return int_value
    elif obj["type"] == "sfix_gc":
        int_value = parse(reader, obj["value"][0], level=level+1)
        # TODO(ryan): Changed this k from an f.
        value = int_value * 1.0 / (2 ** int(obj["k"]))
        if level == 0:
            print "fixed_point:{} = {}".format(obj["name"], value)
        else:
            return value
    elif obj["type"] == "ArrayGC":
        values = []
        for o in obj["value"]:
            values.append(parse(reader, o, level=level+1))
        if level == 0:
            print "Array {}".format(obj["name"])
            for v in values:
                print v
        else:
            return values
    elif obj["type"] == "MatrixGC":
        values = []
        for o in obj["value"]:
            row = parse(reader, o, level=level+1)
            values.append(row)
        if level == 0:
            print "Matrix {} ".format(obj["name"])
            for row in values:
                for v in row:
                    print v ,
                print ""
        else:
            return values
    else:
        print obj
        raise NotImplementedError
                
def main():
    parser = argparse.ArgumentParser(description="An AG-MPC output parser")
    parser.add_argument("output_data_file", type=str)
    parser.add_argument("circuit_directory", type=str)

    args = parser.parse_args()

    f = open(args.output_data_file, 'r')
    data = f.read()
    f.close()

    f = open(args.circuit_directory+"/agmpc.txt.output", 'r')
    data_format = f.read()
    data_format = json.loads(data_format)
    f.close()

    reader = BitReader(data)
    for obj in data_format:
        parse(reader, obj)

main()
