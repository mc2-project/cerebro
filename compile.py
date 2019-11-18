#!/usr/bin/env python


#     ===== Compiler usage instructions =====
# 
# See documentation for details on the Compiler package


import argparse
import Compiler
import Compiler.planning as planning

def main():
    parser = argparse.ArgumentParser(description="A compiler for generating arithmetic or GC circuits from .mpc files")
    parser.add_argument('mpc_type', type=str, help="Use 'a' for arithmetic, and 'b' for garbled circuit")
    parser.add_argument('filename', type=str, help="File name of the .mpc program")
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


    parser.add_argument("-sp", "--split", action="store_true", dest="split", help="Whether or not to split the program")
    parser.add_argument("-ur", "--unroll", action="store_true", default=True, dest="unroll", help="Whether or not to unroll a loop")
    parser.add_argument("-in", "--inline", action="store_true", default=False, dest="inline", help="Whether or not to inline functions")

    # Add argument for constants file
    parser.add_argument("-cf", "--constant_file", dest="constant_file", default="", help="File for the constants")
    parser.add_argument("-p", "--party", default=0, dest="party", help="party number")

    args = parser.parse_args()

    print args 
    options = args
    args = [options.mpc_type, options.filename]
    
    def compilation():
        prog = Compiler.run(args, options,
                            merge_opens=options.merge_opens, 
                            debug=options.debug)
        prog.write_bytes(options.outfile)
        if options.asmoutfile:
            for tape in prog.tapes:
                tape.write_str(options.asmoutfile + '-' + tape.name)
        
    if options.profile:
        import cProfile
        p = cProfile.Profile().runctx('compilation()', globals(), locals())
        p.dump_stats(args[0] + '.prof')
        p.print_stats(2)
    else:
        compilation()

if __name__ == '__main__':
    main()
