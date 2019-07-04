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
import json

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

    MAX_INSTS = 500000
    
    def __init__(self, args, options, param=-1):
        self.instructions = [None] * ProgramGC.MAX_INSTS
        self.inst_counter = 0

        self.init_names(args)
        self.num_instructions = 0
        self.total_wires = 0
        self.total_inputs = 0
        self.input_wires = {}
        self.output_wires = []
        self.output_objects = []

        print "[GC compilation] Writing to file"
        self.f = open(self.outfile, 'w')
        
        # Prepend a bunch of bytes to save room
        for i in range(2 * 100):
            self.f.write(" ")

    def add_instruction(self, inst):
        self.num_instructions += 1
        self.instructions[self.inst_counter] = inst
        self.inst_counter += 1
        if self.inst_counter >= ProgramGC.MAX_INSTS:
            self.flush_instructions()
            self.inst_counter = 0
            
    def flush_instructions(self):
        for i in range(self.inst_counter):
            self.f.write("\n{}".format(self.instructions[i]))
            self.f.flush()
        print "Processed {} instructions".format(self.num_instructions)

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

        self.outfile = self.programs_dir + '/agmpc.txt'

    def write_bytes(self, outfile=None):
        fname = self.outfile

        # Finish outputting all instructions
        self.flush_instructions()

        self.f.write("\n")

        # Output the output wires
        self.f.write("2 1 {} {} {} XOR\n".format(0, 0, self.total_wires))
        wire_count = self.total_wires
        for b in self.output_wires:
            wire_count += 1
            self.f.write("2 1 {} {} {} XOR\n".format(self.total_wires, b, wire_count))

        # Seek to the beginning and rewrite
        self.f.seek(0, 0)
        self.f.write("{} {}\n".format(self.num_instructions + 1 + len(self.output_wires), self.total_wires + 1 + len(self.output_wires)))
        self.f.write("{} {} {}\n".format(self.total_inputs, 0, len(self.output_wires)))
        self.f.write("\n")

        #self.f.close()

        # Write input wire configuration for each party
        # format = party, wire start, wire end
        f = open(fname + ".input", 'w')
        num_parties = len(set(self.input_wires.keys()))
        # First output the total number of input wires
        f.write("{} {}\n".format(self.total_inputs, len(self.output_wires)))
        for party, wires in self.input_wires.iteritems():
            for v in find_ranges(wires):
                f.write("{} {} {}\n".format(party, v[0], v[1]))
            
        f.close()

        # Write output parsing
        f = open(fname + ".output", 'w')
        f.write(json.dumps(self.output_objects))
        f.close()
