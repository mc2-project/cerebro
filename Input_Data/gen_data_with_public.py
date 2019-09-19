import struct
import argparse

def write_spdz(input_folder, data):
    f = open(input_folder + "/f0", 'w')
    for d in data[0]:
        sign = d < 0
        output = struct.pack("?", sign)
        f.write(output)
        output = struct.pack("Q", abs(int(d)))
        f.write(output)
    f.close()

    f = open(input_folder + "/f1", 'w')
    for d in data[1]:
        sign = d < 0
        output = struct.pack("?", sign)
        f.write(output)
        output = struct.pack("Q", abs(int(d)))
        f.write(output)
    f.close()

def main():
    parser = argparse.ArgumentParser(description="An MC2 input parser")
    parser.add_argument("input_folder", type=str)
    parser.add_argument("data_source", type=str)
    args = parser.parse_args()

    import importlib
    data = importlib.import_module(args.data_source).data
    write_spdz(args.input_folder, data)

main()
