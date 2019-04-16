import struct
import argparse

def write_spdz(input_folder, data):
    f = open(input_folder + "/f0", 'w')
    for d in data:
        sign = d < 0
        output = struct.pack("?", sign)
        f.write(output)
        output = struct.pack("Q", abs(int(d)))
        f.write(output)
    f.close()

# When writing out data for AG-MPC, we need to make sure that the format is [MSB ... LSB]
def write_agmpc(input_folder, data):
    data_rev = data[::-1]
    f = open(input_folder + "/agmpc.input", 'w')
    for d in data_rev:
        output = struct.pack(">q", int(d))
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
    write_agmpc(args.input_folder, data)

main()
