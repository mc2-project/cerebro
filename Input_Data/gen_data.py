import struct
import argparse

def write_spdz(input_folder, data):
    f = open(input_folder + "/f0", 'w')
    for d in data:
        output = struct.pack("L", d)
        f.write(output)
    f.close()

def write_agmpc(input_folder, data):
    f = open(input_folder + "/agmpc.input", 'w')
    for d in data:
        output = struct.pack("L", d)
        f.write(output)
    f.close()

def main():
    parser = argparse.ArgumentParser(description="An MC2 input parser")
    parser.add_argument("input_folder", type=str)
    args = parser.parse_args()
    
    # leaf = 1000
    # data = [0, 0, 1, 2, 1, 1, 3, 4, 2, 2, 5, 6, leaf, 0, leaf, leaf, leaf, 1, leaf, leaf, leaf, 2, leaf, leaf, leaf, 3, leaf, leaf]
    # test_features = [5] * 10
    # data += test_features
    
    data = [123]
    write_spdz(args.input_folder, data)
    write_agmpc(args.input_folder, data)

main()
