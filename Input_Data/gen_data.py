import struct
import argparse

def write_spdz(input_folder, data):
    f = open(input_folder + "/f0", 'w')
    for d in data:
        sign = d < 0
        output = struct.pack("?", sign)
        f.write(output)
        output = struct.pack("Q", abs(d))
        f.write(output)
    f.close()

# When writing out data for AG-MPC, we need to make sure that the format is [MSB ... LSB]
def write_agmpc(input_folder, data):
    f = open(input_folder + "/agmpc.input", 'w')
    for d in data:
        output = struct.pack(">q", d)
        f.write(output)
    f.close()

def main():
    parser = argparse.ArgumentParser(description="An MC2 input parser")
    parser.add_argument("input_folder", type=str)
    parser.add_argument("data_source", type=str)
    args = parser.parse_args()

    import args.data_source.data as data
    write_spdz(args.input_folder, data)
    write_agmpc(args.input_folder, data)

main()
