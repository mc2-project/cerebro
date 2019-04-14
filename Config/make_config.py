#!/usr/bin/env python

import struct
import argparse

def main():
    parser = argparse.ArgumentParser(description="An MC2 machine configuration loader")
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    
    # open the server list and read the list of servers
    f_server = open(args.config_file, 'r')
    server_list = f_server.read()
    f_server.close()

    server_list = server_list.splitlines()


    f = open("./emp-agmpc/emp-agmpc/cmpc_config.h", 'w')
    f.write("#ifndef __CMPC_CONFIG\n");
    f.write("#define __CMPC_CONFIG\n");
    f.write("const static int abit_block_size = 1024;\n")
    f.write("const static int fpre_threads = 1;\n")
    f.write("const static char *IP[] = { \"\",\n")

    for server_ip in server_list:
        f.write("\t\"" + server_ip + "\",\n")

    f.write("\t\"\"};\n");
    f.write("const static bool lan_network = false;\n");
    f.write("#endif// __CMPC_CONFIG\n");
main()

