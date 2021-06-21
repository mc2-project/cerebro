#!/usr/bin/env python

import struct
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="An MC2 machine configuration loader", add_help=True)
    parser.add_argument("num_party", type=int, help="Number of parties")
    args = parser.parse_args()
    
    # Generate certificates according to SCALE-MAMBA documentation
    os.system("cd ./Certificates/ && openssl genrsa -out RootCA.key 4096")
    os.system("cd ./Certificates/ && openssl req -new -x509 -days 3600 -key RootCA.key -out RootCA.crt") 
    for party_num in range(args.num_party):
        os.system("cd ./Certificates/ && openssl genrsa -out " + "Player{0}.key 2048".format(party_num))
        os.system("cd ./Certificates/ && openssl req -new -key Player{0}.key -out Player{0}.csr".format(party_num))
        os.system("cd ./Certificates/ && openssl x509 -req -days 1000 -in Player{0}.csr -CA RootCA.crt -CAkey RootCA.key -set_serial 0101 -out Player{0}.crt -sha256".format(party_num))


    # Copy data into SCALE-MAMBA Cert-Store
    os.system("cp ./Certificates/* ./crypto_backend/SCALE-MAMBA/Cert-Store/")
main()

