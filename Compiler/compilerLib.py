import random
import time
import sys
import planning
import ast
import types
from collections import deque
import os


def run_arithmetic(args, options, param=-1, merge_opens=True, \
                   reallocate=True, debug=False):
    
    from Compiler.program import Program
    from Compiler.config import *
    from Compiler.exceptions import *
    import instructions, instructions_base, types, comparison, library

    import interface
    from interface import ASTParser as ASTParser
    import inspect
    import copy
    interface.mpc_type = interface.SPDZ

    _interface = [t[1] for t in inspect.getmembers(interface, inspect.isclass)]
    for op in _interface:
        VARS[op.__name__] = op    
    
    """ Compile a file and output a Program object.
    
    If merge_opens is set to True, will attempt to merge any parallelisable open
    instructions. """
    
    prog = Program(args, options, param)
    instructions.program = prog
    instructions_base.program = prog
    types.program = prog
    comparison.program = prog
    prog.DEBUG = debug
    
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
    # create the tapes
    print 'Compiling file', prog.infile
    party = options.party

    debug = True
    a = ASTParser(prog.infile, party, debug, options)
    vectorized_calls, local_program = a.parse(options.split, options.unroll, options.inline)

    
    if local_program:
        local_file = open(os.path.join(options.filename, "local.py"), "w")
        local_file.write(local_program)
        local_file.close()

    a.execute(VARS)

    # optimize the tapes
    for tape in prog.tapes:
        tape.optimize(options)
    
    if prog.main_thread_running:
        prog.update_req(prog.curr_tape)
    print 'Program requires:', repr(prog.req_num)
    print 'Memory size:', prog.allocated_mem
    print 'Program requires {0} rounds of communication in total.'.format(prog.rounds)
    print 'Program requires {0} invocations in total.'.format(prog.invocations)
    print 'Matmul calls: {0}'.format(vectorized_calls)
    for k in vectorized_calls.keys():
        prog.req_num[k] = vectorized_calls[k]



    num_vectorized_triples, num_vectorized_bits = sub_vectorized_triples(vectorized_calls, prog.req_num)
    print "Due to vectorized triples, Reduced triple count by: {0}. Reduced bit count by: {1}.".format(num_vectorized_triples, num_vectorized_bits)
    #prog.req_num[('modp', 'triple')] -= num_vectorized_triples
    #prog.req_num[('modp', 'bit')] -= num_vectorized_bits
    # Don't want negative triples/bits
    assert(prog.req_num[('modp', 'triple')] >= num_vectorized_triples)
    assert(prog.req_num[('modp', 'bit')] >= num_vectorized_bits)



    # finalize the memory
    prog.finalize_memory()


    # Write file to output 
    prog.write_bytes(options.outfile)
    return prog



# Helper method to remove the triples used for matrix multiplication
def sub_vectorized_triples(vectorized_calls, req_num):
    num_triples = 0
    num_bits = 0
    for dims in vectorized_calls:
        left_rows = dims[0]
        left_cols = dims[1]
        right_rows = dims[2]
        right_cols = dims[3]
        one_matmul_triple_cost = left_rows * left_cols * right_cols
        if dims[-1] == "cfix" or dims[-1] == "cint":
            pass
        elif dims[-1] == "sfix":
            # 2k + kappa bits needed per multiplication of sfix.
            one_matmul_bit_cost = one_matmul_triple_cost * (2 * types.sfix.k + types.sfix.kappa)
            num_bits += one_matmul_bit_cost * vectorized_calls[dims]
            num_triples += one_matmul_triple_cost * vectorized_calls[dims]

        elif dims[-1] == "sint":
            # Multiply one matmul with the number of matmuls needed.
            num_triples += one_matmul_triple_cost * vectorized_calls[dims]
        else:
            raise ValueError("Unrecognized type when processing matmul/finding vectorized triples: {0}".format(dims[-1]))


    return num_triples, num_bits


# Similar 
def run_gc(args, options, param=-1, merge_opens=True, \
           reallocate=True, debug=False):

    from Compiler.program_gc import ProgramGC
    import instructions_gc, types_gc
    import interface
    from interface import ASTParser as ASTParser
    import inspect

    interface.mpc_type = interface.GC

    _interface = [t[1] for t in inspect.getmembers(interface, inspect.isclass)]
    for op in _interface:
        VARS[op.__name__] = op
    
    prog = ProgramGC(args, options, param)
    instructions_gc.program_gc = prog
    types_gc.program_gc = prog
    interface.program_gc = prog
    VARS['program_gc'] = prog

    print 'Compiling file', prog.infile
    party = options.party
    a = ASTParser(prog.infile, party, debug=True)
    a.parse(options.split, options.unroll, options.inline)
    a.execute(VARS)

    # Write output
    # prog.write_bytes(prog.outfile)

    return prog


def run(args, options, param=-1, merge_opens=True, \
        reallocate=True, debug=False):
    
    # Do planning
    if options.constant_file:
        arithmetic_prog = run_arithmetic(args[1:], options, param, merge_opens=merge_opens, debug=debug)
        boolean_prog = run_gc(args[1:], options, param, merge_opens=merge_opens, debug=debug)
        run_planner(arithmetic_prog, boolean_prog, options.constant_file)

    else:
        if args[0] == 'a':
            return run_arithmetic(args[1:], options, param, merge_opens=merge_opens, debug=debug)

        elif args[0] == 'b':
            return run_gc(args[1:], options, param, merge_opens=merge_opens, debug=debug)
        else:
            raise ValueError("Must choose either arithmetic (a) or GC (b)")
            print "---------------------------------------------"
            print "NO CONSTANTS FILE"
            print "---------------------------------------------" 


def run_planner(arithmetic_prog, boolean_prog, constant_file):
    d_b = planning.agmpc_cost(boolean_prog, constant_file)
    print "AG-MPC cost: ", d_b['total_cost']
    d_a = planning.spdz_cost(arithmetic_prog, constant_file)
    print "SPDZ cost: ", d_a['total_cost']
    if d_b['total_cost'] < d_a['total_cost']:
        print "BOOLEAN"
        print "---------------------------------------------"
        print "Decision: ", d_b['decision']
        print "Cost: ", d_b['total_cost']
        print "---------------------------------------------"

    else:
        print "---------------------------------------------"
        print "ARITHMETIC"
        print "Decision: ", d_a['decision']
        print "Cost: ", d_a['total_cost']
        print "---------------------------------------------"
               
