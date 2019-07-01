#!/usr/bin/env python

import struct
import argparse
import os

def main():
    server_ip = os.environ['MY_PARTY_IP']

    if not server_ip:
        print "Environment variable MY_PARTY_IP is not set."
        exit(1)

    cwd = os.getcwd()

    os.system("cd ./emp-tool/ && rm -rf CMakeFiles && rm -rf CMakeCache.txt")
    os.system("cd ./emp-tool/ && cmake . -DNETIO_USE_TLS=on -DNETIO_CA_CERTIFICATE=\"" + cwd + "/Certificates/ca.pem\" -DNETIO_MY_CERTIFICATE=\"" + cwd + "/Certificates/" + server_ip + ".pem\"") 
    os.system("cd ./emp-tool/ && make && sudo make install")

    os.system("cd ./emp-ot/ && rm -rf CMakeFiles && rm -rf CMakeCache.txt")
    os.system("cd ./emp-ot/ && cmake . -DNETIO_USE_TLS=on -DNETIO_CA_CERTIFICATE=\"" + cwd + "/Certificates/ca.pem\" -DNETIO_MY_CERTIFICATE=\"" + cwd + "/Certificates/" + server_ip + ".pem\"")
    os.system("cd ./emp-ot/ && make && sudo make install")

    os.system("cd ./emp-agmpc/ && rm -rf CMakeFiles && rm -rf CMakeCache.txt")
    os.system("cd ./emp-agmpc/ && cmake . -DNETIO_USE_TLS=on -DNETIO_CA_CERTIFICATE=\"" + cwd + "/Certificates/ca.pem\" -DNETIO_MY_CERTIFICATE=\"" + cwd + "/Certificates/" + server_ip + ".pem\"")
    os.system("cd ./emp-agmpc/ && make") 

main()

