#!/usr/bin/env python


#     ===== Compiler usage instructions =====
# 
# See documentation for details on the Compiler package


import argparse
import Compiler
import Compiler.planning as planning

def main():
    parser = argparse.ArgumentParser(description="A compiler for generating arithmetic or GC circuits from .mpc files")
    parser.add_argument('filename', type=str, help="File name of the .mpc program")

    # Temporarily disable splitting since it is a work-in-progress
    parser.add_argument("-sp", "--split", action="store_false", default=False, dest="split", help="Whether or not to split the program")
    parser.add_argument("-ur", "--unroll", action="store_true", default=True, dest="unroll", help="Whether or not to unroll a loop")
    parser.add_argument("-in", "--inline", action="store_true", default=False, dest="inline", help="Whether or not to inline functions")

    # Add argument for constants file
    parser.add_argument("-cf", "--constant_file", dest="constant_file", default="", help="File for the constants")
    parser.add_argument("-p", "--party", default=0, dest="party", help="party number")

    args = parser.parse_args()

    print args 
    options = args
    #options.mpc_type = "b"
    prog = Compiler.plan(options.filename, options.constant_file, options)

if __name__ == '__main__':
    main()
