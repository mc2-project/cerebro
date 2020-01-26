#!/usr/bin/env python


#     ===== Compiler usage instructions =====
# 
# See documentation for details on the Compiler package


import argparse
import subprocess, shlex, os

import Compiler
import Compiler.planning as planning

def process_input(input_filename):
    def f(s, x, start=0):
        pos = s.find(x, start)
        if pos == -1:
            return start-1
        return f(s, x, pos+1)

    split = f(input_filename, "/")
    directory = input_filename[:split]
    filename = input_filename[split+1:-3]
    cmd = "python Input_Data/gen_data.py {} {}".format(directory, filename)
    subprocess.Popen(shlex.split(cmd))
    return 

def execute_framework(decision, party_id, program_name):
    root_dir = os.getcwd()
    if decision == "arithmetic":
        os.chdir("./crypto_backend/SCALE-MAMBA/")
        subprocess.call("./Player {} {}".format(party_id, program_name))
    elif decision == "boolean":
        os.chdir("./crypto_backend/emp-toolkit/emp-agmpc/")
        subprocess.call("./bin/run_circuit {} 2000 {} {} {} ".format(party_id, program_name, program_name, program_name))
    else:
        raise ValueError("Framework {} is not supported as a valid execution backend".format(decision))
    os.chdir(root_dir)

def main():
    parser = argparse.ArgumentParser(description="A compiler for generating arithmetic or GC circuits from .mpc files")
    parser.add_argument("constants_file", type=str, help="Path to the file for the constants in the cost model")
    parser.add_argument("party", type=str, help="Party number (e.g., 0, 1, etc.)")
    parser.add_argument('program', type=str, help="Name of the .mpc program; file should be placed in Programs")
    parser.add_argument("input_file", type=str, help="Name of the file with inputs (file should be written in Python and under Input_Data; see documentation for further explanation")

    # Temporarily disable splitting since it is a work-in-progress
    parser.add_argument("-sp", "--split", action="store_false", default=False, dest="split", help="Whether or not to split the program")
    parser.add_argument("-ur", "--unroll", action="store_true", default=True, dest="unroll", help="Whether or not to unroll a loop")
    parser.add_argument("-in", "--inline", action="store_true", default=False, dest="inline", help="Whether or not to inline functions")

    args = parser.parse_args()
    options = args
    party_id = options.party
    constants_file = options.constants_file
    program_name = options.program
    input_filename = options.input_file
    decision = Compiler.plan(program_name, constants_file, options)

    process_input(input_filename)
    #execute_framework(decision, party_id, program_name)

if __name__ == '__main__':
    main()
