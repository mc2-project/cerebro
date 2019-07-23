#!/usr/bin/env python


#     ===== Compiler usage instructions =====
# 
# See documentation for details on the Compiler package


import argparse
import Compiler
import Compiler.planning as planning
import subprocess
from Compiler.program import Program
from Compiler.config import *
from Compiler.exceptions import *
from Compiler import instructions, instructions_base, types, comparison, library
from Compiler.compilerLib import VARS

from Compiler import interface
from Compiler.interface import ASTParser as ASTParser

import inspect
import copy
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="A compiler for generating arithmetic or GC circuits from .mpc files")
    parser.add_argument('filename', type=str, help="File name of the .mpc program")
    parser.add_argument("-p", "--party", dest="party", help="party number")
    parser.add_argument("-n", "--nomerge",
                      action="store_false", dest="merge_opens", default=True,
                      help="don't attempt to merge open instructions")
    parser.add_argument("-o", "--output", dest="outfile",
                      help="specify output file")
    parser.add_argument("-a", "--asm-output", dest="asmoutfile",
                      help="asm output file for debugging")
    parser.add_argument("-d", "--debug", action="store_true", dest="debug",
                      help="keep track of trace for debugging")
    parser.add_argument("-c", "--comparison", dest="comparison", default="log",
                      help="comparison variant: log|plain")
    parser.add_argument("-D", "--dead-code-elimination", action="store_true",
                      dest="dead_code_elimination", default=False,
                      help="eliminate instructions with unused result")
    parser.add_argument("-r", "--noreorder", dest="reorder_between_opens",
                      action="store_false", default=True,
                      help="don't attempt to place instructions between start/stop opens")
    parser.add_argument("-M", "--preserve-mem-order", action="store_true",
                      dest="preserve_mem_order", default=False,
                      help="preserve order of memory instructions; possible efficiency loss")
    parser.add_argument("-u", "--noreallocate", action="store_true", dest="noreallocate",
                      default=False, help="don't reallocate")
    parser.add_argument("-m", "--max-parallel-open", dest="max_parallel_open",
                      default=False, help="restrict number of parallel opens")
    parser.add_argument("-P", "--profile", action="store_true", dest="profile",
                      help="profile compilation")
    parser.add_argument("-C", "--continous", action="store_true", dest="continuous",
                      help="continuous computation")
    parser.add_argument("-s", "--stop", action="store_true", dest="stop",
                      help="stop on register errors")
    parser.add_argument("-f", "--fdflag", action="store_false",
                      dest="fdflag", default=True,
                      help="de-activates under-over flow check for sfloats")
    parser.add_argument("-l", "--local", action="store_true", dest="local", default=True, help="True means run local computation")
    args = parser.parse_args()

    options = args


    if options.local:
      # Setup start, copied from compile.py and compilerLib.py
      args = [options.filename]


      local_path = os.path.join(options.filename, "local.py")
      party = options.party

      
      interface.mpc_type = interface.SPDZ

      _interface = [t[1] for t in inspect.getmembers(interface, inspect.isclass)]
      for op in _interface:
          VARS[op.__name__] = op    


      param = -1
      prog = Program(args, options, param)
      instructions.program = prog
      instructions_base.program = prog
      types.program = prog
      comparison.program = prog
      prog.DEBUG = options.debug
      
      VARS['program'] = prog
      comparison.set_variant(options)
      
      print 'Compiling file', prog.infile
      
      if instructions_base.Instruction.count != 0:
          print 'instructions count', instructions_base.Instruction.count
          instructions_base.Instruction.count = 0
      prog.FIRST_PASS = False
      prog.reset_values()
      # make compiler modules directly accessible
      sys.path.insert(0, 'Compiler')

      # Setup end

      local_env = {}
      f = open(local_path, "r")
      local_program = f.read()
      temp = interface.mpc_type
      interface.mpc_type = interface.LOCAL 
      exec(local_program, VARS, local_env)
      interface.mpc_type = temp
      f.close()

    cmd = "./Player.x {0} {1}".format(party, options.filename) + " -max 10,10,10"
    subprocess.call(cmd, shell=True) 



if __name__ == '__main__':
    main()
