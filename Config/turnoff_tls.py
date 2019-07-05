#!/usr/bin/env python

import struct
import argparse
import os

def main():
    param = "-DNETIO_USE_TLS=off"

    os.system("cd ./emp-tool/ && rm -rf CMakeFiles && rm -rf CMakeCache.txt")
    os.system("cd ./emp-tool/ && cmake . " + param) 
    os.system("cd ./emp-tool/ && make && sudo make install")

    os.system("cd ./emp-ot/ && rm -rf CMakeFiles && rm -rf CMakeCache.txt")
    os.system("cd ./emp-ot/ && cmake . " + param)
    os.system("cd ./emp-ot/ && make && sudo make install")

    os.system("cd ./emp-agmpc/ && rm -rf CMakeFiles && rm -rf CMakeCache.txt")
    os.system("cd ./emp-agmpc/ && cmake . " + param)
    os.system("cd ./emp-agmpc/ && make") 

main()

