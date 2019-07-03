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

    os.system("cd ./Certificates/ && openssl genrsa -out ca.key 4096")
    os.system("cd ./Certificates/ && openssl req -new -x509 -days 3600 -key ca.key -out ca.pem -subj \"/C=US/ST=California/L=Berkeley/O=RISELab/OU=MC2/CN=CA/emailAddress=w.k@berkeley.edu\"")
   

    serial_num = 0
    for server_ip in server_list:
        serial_num = serial_num + 1
        os.system("cd ./Certificates/ && openssl genrsa -out " + server_ip +".key 2048")
        os.system("cd ./Certificates/ && openssl req -new -key " + server_ip + ".key -out " + server_ip + "_presign.pem -subj \"/C=US/ST=California/L=Berkeley/O=RISELab/OU=MC2/CN=" + server_ip + "/emailAddress=w.k@berkeley.edu\"")
        os.system("cd ./Certificates/ && openssl x509 -req -days 3000 -in " + server_ip + "_presign.pem -CA ca.pem -set_serial " + str(serial_num) + " -CAkey ca.key -out " + server_ip + ".pem -sha256")

main()

