#!/usr/bin/env python

import struct
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="An MC2 machine configuration loader")
    parser.add_argument("config_file", type=str)
    parser.add_argument("num_party", type=int)
    args = parser.parse_args()
    
    # open the server list and read the list of servers
    f_server = open(args.config_file, 'r')
    server_list = f_server.read()
    f_server.close()

    server_list = server_list.splitlines()
    server_list = server_list[0:args.num_party]

    f = open("./emp-agmpc/emp-agmpc/cmpc_config.h", 'w')
    f.write("#ifndef __CMPC_CONFIG\n")
    f.write("#define __CMPC_CONFIG\n")
    f.write("const static int abit_block_size = 1024;\n")
    f.write("const static int fpre_threads = 1;\n")
    f.write("#define NUM_PARTY_FOR_RUNNING 3\n")
    f.write("const static char *IP[] = { \"\",\n")

    for server_ip in server_list:
        f.write("\t\"" + server_ip + "\",\n")

    f.write("\t\"\"};\n");
    f.write("const static bool lan_network = false;\n")
    f.write("#endif// __CMPC_CONFIG\n")
    f.close()

    f = open("./SCALE-MAMBA/Data/NetworkData.txt", 'w')
    f.write("RootCA.crt\n")
    f.write(str(args.num_party) + "\n")

    counter = 0
    for server_ip in server_list:
        f.write(str(counter) + " " + server_ip + " Player" + str(counter + 1) + ".crt P" + str(counter + 1) + "\n")
        counter = counter + 1

    f.write("1\n")
    f.write("1\n")

    f.close()

    os.system("cd ./emp-agmpc && cmake . && make")

main()

