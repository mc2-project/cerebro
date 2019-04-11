# (C) 2017 University of Bristol. See License.txt

from Compiler.config import *
from Compiler.exceptions import *
import compilerLib
import allocator as al
import random
import time
import sys, os, errno
import inspect
from collections import defaultdict
import itertools
import math
import Compiler.instructions_gc
import types_gc

def find_ranges(l):
    sorted_l = sorted(l)

    low = sorted_l[0]
    prev = sorted_l[0]
    for v in sorted_l:
        if (v - prev) > 1:
            yield (low, prev)
            low = v
            prev = v
        else:
            prev = v

    yield (low, prev)
        
class ProgramGC(object):
    
    class BasicBlock(object):
        def __init__(self):
            self.instructions = []
        
    def __init__(self, args, options, param=-1):
        self.activeblock = ProgramGC.BasicBlock()
        self.init_names(args)
        self.total_wires = 0
        self.total_inputs = 0
        self.input_wires = {}
        self.output_wires = []

    def init_names(self, args):
        self.programs_dir=args[0];
        print 'Compiling program in', self.programs_dir
        if  self.programs_dir.endswith('/'):
           self.programs_dir = self.programs_dir[:-1]
        progname = self.programs_dir.split('/')[-1]
        
        if progname.endswith('.mpc'):
            progname = progname[:-4]
        
        self.infile = self.programs_dir + '/' + progname + '.mpc'
        print progname
        """
        self.name is input file name (minus extension) + any optional arguments.
        Used to generate output filenames
        """
        self.name = progname
        if len(args) > 1:
            self.name += '-' + '-'.join(args[1:])

        self.outfile = self.programs_dir + '/' + progname + '.agmpc.txt'

    @property
    def curr_block(self):
        return self.activeblock

    def write_bytes(self, outfile=None):
        fname = self.outfile
        if outfile is not None:
            fname = outfile

        print "[GC compilation] Writing to file"

        f = open(fname, 'w')

        f.write("{} {}\n".format(len(self.activeblock.instructions) + 1 + len(self.output_wires), self.total_wires + 1 + len(self.output_wires)))
        f.write("{} {} {}\n".format(self.total_inputs, 0, len(self.output_wires)))
        f.write("\n")

        # Output all instructions
        for i in self.activeblock.instructions:
            f.write("{}\n".format(i))

        # Output the output wires
        f.write("2 1 {} {} {} XOR\n".format(0, 0, self.total_wires))
        wire_count = self.total_wires
        for b in self.output_wires:
            wire_count += 1
            f.write("2 1 {} {} {} XOR\n".format(self.total_wires, b, wire_count))

        f.close()

        # Output input wire configuration for each party
        # format = party, wire start, wire end
        f = open(fname + ".input", 'w')
        num_parties = len(set(self.input_wires.keys()))
        # First output the total number of input wires
        f.write("{} {}\n".format(self.total_inputs, len(self.output_wires)))
        for party, wires in self.input_wires.iteritems():
            for v in find_ranges(wires):
                f.write("{} {} {}\n".format(party, v[0], v[1]))
            
        f.close()
