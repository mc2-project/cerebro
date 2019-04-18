#!/usr/bin/env python

import struct
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="An MC2 machine configuration loader")
    parser.add_argument("num_layer", type=int)
    args = parser.parse_args()
    
    f = open("./Programs/test_dt/test_dt.mpc", 'w')
    f.write("Params.set_params(int_precision=32, f=16, k=32)\n")
    f.write("dim = " + str(args.num_layer) + "\n")
    f.write("LEVELS = dim\n")
    f.write("TOTAL_NODES = (2 ** LEVELS) - 1\n")
    
    f.write("tree = s_fix_mat.read_input(TOTAL_NODES, 3, 0)\n")
    f.write("w = tree[0]\n")
    f.write("x = s_fix_array.read_input(dim, 0)\n")

    for i in range(args.num_layer - 1):
        f.write("index = w[0]\n")
        f.write("split = w[1]\n")
        f.write("left_child = w[2]\n")
        f.write("right_child = left_child + 1\n")
        f.write("f = x[index]\n")
        f.write("cond = (f < split)\n")
        layer_start = 2 ** (i + 1) - 1
        layer_end = 2 ** (i + 2) - 1
        layer_size = layer_end - layer_start
        f.write("tree_cur = sfixMatrix(" + str(layer_size) + ", 3)\n")
        f.write("for j in range(" + str(layer_start) + ", " + str(layer_end) +"):\n")
        f.write("\tfor k in range(3):\n")
        f.write("\t\ttree_cur[j][k] = tree[j + " + str(layer_start) +"][k]\n")
        f.write("w_res = array_index_secret_load_if(cond, tree_cur, left_child, right_child)\n")
        f.write("mat_assign(w, w_res)\n")

    f.write("reveal_all(w[1], \"Final prediction class\")\n")
    f.close()

    f = open("./Input_Data/gen_data_dt.py", 'w')
    f.write("leaf = 1000\n")
    f.write("data = []\n")
    f.write("for i in range(" + str(args.num_layer) + "):\n")
    f.write("\tv = 2 ** i\n")
    f.write("\tfor j in range(v):\n")
    f.write("\t\tif i == " + str(args.num_layer - 1) + ":\n")
    f.write("\t\t\tdata += [leaf, j - v + 1, leaf]\n")
    f.write("\t\telse:\n")
    f.write("\t\t\tdata += [i, 2, (v + j + 1) * 2 + 1 - (2 ** (i + 1)) - 1]\n")
    f.write("test_features = [5] * " + str(args.num_layer) + "\n")
    f.write("data += test_features\n")
    f.close()

    os.system("cd ./SCALE-MAMBA && python compile.py a Programs/test_dt/")
    os.system("cd ./Input_Data && python gen_data.py . gen_data_dt")

main()

